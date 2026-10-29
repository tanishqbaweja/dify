from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Never, cast, override

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models.knowledge_fs import (
    KnowledgeFSAuthorizationRevision,
    KnowledgeFSCapabilityClaimsSummary,
    KnowledgeFSCapabilityIssuanceAudit,
    KnowledgeFSCapabilityIssuanceReservation,
    KnowledgeFSCapabilityReservationSummary,
    KnowledgeFSControlSpace,
    KnowledgeFSControlSpaceState,
    KnowledgeFSLifecycleOutbox,
    KnowledgeFSLifecycleOutboxStatus,
    KnowledgeFSRevokeCommandPayload,
)
from services.knowledge_fs.lifecycle_port import (
    KnowledgeFSCapabilityGrantRevokeAck,
    KnowledgeFSCapabilityGrantRevokeRequest,
    KnowledgeFSIntegratedDeletionRequest,
    KnowledgeFSIntegratedProvisionRequest,
    KnowledgeFSRemoteSpace,
)
from services.knowledge_fs.lifecycle_saga import KnowledgeFSLifecycleSagaRunner
from services.knowledge_fs.observability import (
    KnowledgeFSBatchStatusMetric,
    KnowledgeFSCapabilityIssuanceMetric,
    KnowledgeFSControlSpaceStateMetric,
    KnowledgeFSLifecycleTaskMetric,
)
from services.knowledge_fs.revocation_commands import (
    KnowledgeFSRevocationCommandError,
    KnowledgeFSRevocationCommandProducer,
)
from services.knowledge_fs.revocation_reconciler import KnowledgeFSRevocationReconciler
from tests.helpers.unit.services.knowledge_fs_fakes import UnexpectedLifecycleRemote, revoke_payload

_SPACE_ID = "10000000-0000-4000-8000-000000000001"
_OTHER_SPACE_ID = "10000000-0000-4000-8000-000000000002"
_GRANT_A = "20000000-0000-4000-8000-000000000001"
_GRANT_B = "20000000-0000-4000-8000-000000000002"


class FakeRemote(UnexpectedLifecycleRemote):
    def __init__(self) -> None:
        self.revoke_requests: list[KnowledgeFSCapabilityGrantRevokeRequest] = []
        self.revoke_acks: deque[KnowledgeFSCapabilityGrantRevokeAck] = deque()

    @override
    def revoke_capability_grant(
        self, request: KnowledgeFSCapabilityGrantRevokeRequest
    ) -> KnowledgeFSCapabilityGrantRevokeAck:
        self.revoke_requests.append(request)
        return self.revoke_acks.popleft()

    @override
    def provision_integrated_space(self, request: KnowledgeFSIntegratedProvisionRequest) -> KnowledgeFSRemoteSpace:
        raise AssertionError(request)

    @override
    def request_integrated_deletion(self, request: KnowledgeFSIntegratedDeletionRequest) -> Never:
        raise AssertionError(request)

    @override
    def find_by_provisioning_key(self, *, provisioning_key: str, control_space_id: str) -> Never:
        raise AssertionError((provisioning_key, control_space_id))

    @override
    def list_spaces(self, *, namespace_id: str, control_space_id: str) -> Never:
        raise AssertionError((namespace_id, control_space_id))


@dataclass
class RecordingMetrics:
    lifecycle_tasks: list[KnowledgeFSLifecycleTaskMetric] = field(default_factory=list)

    def record_batch_status(self, event: KnowledgeFSBatchStatusMetric) -> None:
        raise AssertionError(event)

    def record_capability_issuance(self, event: KnowledgeFSCapabilityIssuanceMetric) -> None:
        raise AssertionError(event)

    def record_control_space_state(self, event: KnowledgeFSControlSpaceStateMetric) -> None:
        raise AssertionError(event)

    def record_lifecycle_task(self, event: KnowledgeFSLifecycleTaskMetric) -> None:
        self.lifecycle_tasks.append(event)

    def register_control_space_state_gauge(self, read_counts: Callable[[], object]) -> None:
        raise AssertionError(read_counts)


def _unused_session() -> Session:
    return Session()


def _grant(number: int) -> str:
    return f"20000000-0000-4000-8000-{number:012d}"


def _control_space(
    *,
    tenant_id: str = "tenant-1",
    provisioning_key: str = "provision-1",
    knowledge_space_id: str | None = _SPACE_ID,
    state: KnowledgeFSControlSpaceState = KnowledgeFSControlSpaceState.ACTIVE,
) -> KnowledgeFSControlSpace:
    return KnowledgeFSControlSpace(
        tenant_id=tenant_id,
        owner_account_id="owner-1",
        provisioning_key=provisioning_key,
        knowledge_space_id=knowledge_space_id,
        state=state,
    )


def _revision(control_space: KnowledgeFSControlSpace) -> KnowledgeFSAuthorizationRevision:
    return KnowledgeFSAuthorizationRevision(tenant_id=control_space.tenant_id, control_space_id=control_space.id)


def _claims_audit(claims: dict[str, object], *, control_space_id: str) -> KnowledgeFSCapabilityIssuanceAudit:
    return KnowledgeFSCapabilityIssuanceAudit(
        tenant_id="tenant-1",
        control_space_id=control_space_id,
        trace_id="trace-1",
        jti_hash="sha256:" + "a" * 64,
        claims_summary=cast(KnowledgeFSCapabilityClaimsSummary, claims),
    )


def _reservation(
    *,
    control_space_id: str,
    grant_id: str = _GRANT_A,
    subject: str = "dify-account:member-1",
    caller_kind: str = "interactive",
    summary: dict[str, object] | None = None,
) -> KnowledgeFSCapabilityIssuanceReservation:
    if summary is None:
        summary = {"caller_kind": caller_kind, "grant_id": grant_id, "subject": subject}
    return KnowledgeFSCapabilityIssuanceReservation(
        tenant_id="tenant-1",
        control_space_id=control_space_id,
        grant_id=grant_id,
        trace_id="trace-1",
        subject=subject,
        caller_kind=caller_kind,
        request_summary=cast(KnowledgeFSCapabilityReservationSummary, summary),
    )


def _maker(sqlite_session: Session) -> sessionmaker[Session]:
    return sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False)


def _audit(
    *,
    control_space_id: str,
    grant_id: str,
    subject: str = "dify-account:member-1",
    caller_kind: str = "interactive",
    jti: str,
) -> KnowledgeFSCapabilityIssuanceAudit:
    return KnowledgeFSCapabilityIssuanceAudit(
        tenant_id="tenant-1",
        control_space_id=control_space_id,
        trace_id=f"trace-{jti}",
        jti_hash=f"sha256:{jti * 64}"[:71],
        claims_summary=cast(
            KnowledgeFSCapabilityClaimsSummary,
            {
                "caller_kind": caller_kind,
                "grant_id": grant_id,
                "subject": subject,
            },
        ),
    )


def test_producer_strictly_binds_and_deduplicates_grants_with_a_monotonic_space_sequence(
    sqlite_session: Session,
) -> None:
    space = KnowledgeFSControlSpace(
        tenant_id="tenant-1",
        owner_account_id="owner-1",
        provisioning_key="provision-1",
        knowledge_space_id=_SPACE_ID,
        state=KnowledgeFSControlSpaceState.ACTIVE,
    )
    sqlite_session.add_all(
        [
            space,
            KnowledgeFSAuthorizationRevision(tenant_id="tenant-1", control_space_id=space.id),
            _audit(control_space_id=space.id, grant_id=_GRANT_A, jti="a"),
            _audit(control_space_id=space.id, grant_id=_GRANT_A, jti="b"),
            _audit(control_space_id=space.id, grant_id=_GRANT_B, jti="c"),
            _audit(
                control_space_id=space.id,
                grant_id="20000000-0000-4000-8000-000000000003",
                subject="dify-account:other",
                jti="d",
            ),
            _audit(
                control_space_id=space.id,
                grant_id="20000000-0000-4000-8000-000000000004",
                caller_kind="service",
                jti="e",
            ),
        ]
    )
    sqlite_session.commit()

    with _maker(sqlite_session).begin() as session:
        commands = KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            subject="dify-account:member-1",
            reason_code="permission_revoked",
            caller_kinds=("interactive",),
        )

    with _maker(sqlite_session)() as session:
        revision = session.scalar(
            select(KnowledgeFSAuthorizationRevision).where(
                KnowledgeFSAuthorizationRevision.control_space_id == space.id
            )
        )
        persisted = tuple(session.scalars(select(KnowledgeFSLifecycleOutbox)))
    persisted_payloads = sorted(
        (cast(KnowledgeFSRevokeCommandPayload, command.command_payload) for command in persisted),
        key=itemgetter("revoke_sequence"),
    )
    assert revision is not None
    assert revision.revoke_sequence == 2
    assert len(commands) == len(persisted) == 2
    assert [payload["grant_id"] for payload in persisted_payloads] == [_GRANT_A, _GRANT_B]
    assert [payload["revoke_sequence"] for payload in persisted_payloads] == [1, 2]
    assert all(payload["knowledge_space_id"] == _SPACE_ID for payload in persisted_payloads)


def test_principal_revocation_includes_reserved_grants_and_ignores_other_principals_callers_and_spaces(
    sqlite_session: Session,
) -> None:
    space = _control_space()
    other_space = _control_space(provisioning_key="provision-2", knowledge_space_id=_OTHER_SPACE_ID)
    sqlite_session.add_all(
        [
            space,
            _revision(space),
            other_space,
            _revision(other_space),
            _audit(control_space_id=space.id, grant_id=_GRANT_A, jti="a"),
            _reservation(control_space_id=space.id, grant_id=_GRANT_A),
            # Reserved before signing; no issuance audit exists yet.
            _reservation(control_space_id=space.id, grant_id=_GRANT_B),
            _reservation(control_space_id=space.id, grant_id=_grant(3), subject="dify-account:other"),
            _reservation(control_space_id=space.id, grant_id=_grant(4), caller_kind="service"),
            _reservation(control_space_id=other_space.id, grant_id=_grant(5)),
            _audit(control_space_id=other_space.id, grant_id=_grant(6), jti="b"),
        ]
    )
    sqlite_session.commit()

    with _maker(sqlite_session).begin() as session:
        commands = KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            subject="dify-account:member-1",
            reason_code="permission_revoked",
            caller_kinds=("interactive",),
        )

    assert [revoke_payload(command)["grant_id"] for command in commands] == [_GRANT_A, _GRANT_B]


def test_revoke_consumer_accepts_idempotent_or_higher_watermark_and_reconciliation_replays(
    sqlite_session: Session,
) -> None:
    space = KnowledgeFSControlSpace(
        tenant_id="tenant-1",
        owner_account_id="owner-1",
        provisioning_key="provision-1",
        knowledge_space_id=_SPACE_ID,
        state=KnowledgeFSControlSpaceState.ACTIVE,
    )
    sqlite_session.add_all(
        [
            space,
            KnowledgeFSAuthorizationRevision(tenant_id="tenant-1", control_space_id=space.id),
            _audit(control_space_id=space.id, grant_id=_GRANT_A, jti="f"),
        ]
    )
    sqlite_session.commit()
    maker = _maker(sqlite_session)
    with maker.begin() as session:
        (command,) = KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            subject="dify-account:member-1",
            reason_code="permission_revoked",
            caller_kinds=("interactive",),
        )
        command.created_at = datetime(2026, 7, 21, 11, 59, 35)

    remote = FakeRemote()
    remote.revoke_acks.append(KnowledgeFSCapabilityGrantRevokeAck(False, 7, "revoked"))
    metrics = RecordingMetrics()
    dispatched = KnowledgeFSLifecycleSagaRunner(maker, remote, metrics=metrics).dispatch_one(
        worker_id="worker-1",
        now=datetime(2026, 7, 21, 12, 0),
        lease_duration=timedelta(minutes=1),
        product_enabled=True,
    )

    with maker() as session:
        persisted = session.get(KnowledgeFSLifecycleOutbox, command.id)
    assert dispatched.completed is True
    assert persisted is not None
    assert persisted.status is KnowledgeFSLifecycleOutboxStatus.SUCCEEDED
    command_payload = cast(KnowledgeFSRevokeCommandPayload, command.command_payload)
    assert remote.revoke_requests[0].event_id == command_payload["event_id"]
    assert remote.revoke_requests[0].revoke_sequence == 1
    assert metrics.lifecycle_tasks[-1] == KnowledgeFSLifecycleTaskMetric(25.0, "revoke", "succeeded")
    assert "tenant-1" not in str(metrics.lifecycle_tasks)
    assert command.id not in str(metrics.lifecycle_tasks)

    remote.revoke_acks.append(KnowledgeFSCapabilityGrantRevokeAck(True, 1, "revoked"))
    repaired = KnowledgeFSRevocationReconciler(maker, remote).reconcile(
        tenant_id="tenant-1",
        control_space_id=space.id,
    )
    assert repaired.checked == 1
    assert repaired.repaired == 1
    assert repaired.issues == ()

    remote.revoke_acks.append(KnowledgeFSCapabilityGrantRevokeAck(False, 0, "active"))
    lag = KnowledgeFSRevocationReconciler(maker, remote).reconcile(
        tenant_id="tenant-1",
        control_space_id=space.id,
    )
    assert lag.checked == 1
    assert lag.repaired == 0
    assert lag.issues[0].error_code == "REMOTE_REVOKE_WATERMARK_LAG"


@pytest.mark.parametrize(
    ("subject", "reason", "message"),
    [
        ("  ", "permission_revoked", "subject"),
        ("dify-account:member-1", "INVALID REASON", "reason code"),
    ],
)
def test_principal_revocation_rejects_unbound_operator_input(subject: str, reason: str, message: str) -> None:
    with pytest.raises(KnowledgeFSRevocationCommandError, match=message):
        KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=_unused_session(),
            tenant_id="tenant-1",
            control_space_id="control-1",
            subject=subject,
            reason_code=reason,
        )


def test_principal_revocation_requires_control_space_revision_and_registration(sqlite_session: Session) -> None:
    other_tenant_space = _control_space(tenant_id="tenant-2")
    unrevised_space = _control_space(provisioning_key="provision-2", knowledge_space_id=_OTHER_SPACE_ID)
    registered_space = _control_space(provisioning_key="provision-3")
    unregistered_space = _control_space(
        provisioning_key="provision-4",
        knowledge_space_id=None,
        state=KnowledgeFSControlSpaceState.PROVISIONING,
    )
    sqlite_session.add_all(
        [
            other_tenant_space,
            _revision(other_tenant_space),
            unrevised_space,
            registered_space,
            _revision(registered_space),
            unregistered_space,
            _revision(unregistered_space),
            _audit(control_space_id=unregistered_space.id, grant_id=_GRANT_A, jti="a"),
        ]
    )
    sqlite_session.commit()
    producer = KnowledgeFSRevocationCommandProducer()

    def revoke(control_space_id: str) -> tuple[KnowledgeFSLifecycleOutbox, ...]:
        return producer.enqueue_principal_grants(
            session=sqlite_session,
            tenant_id="tenant-1",
            control_space_id=control_space_id,
            subject="dify-account:member-1",
            reason_code="permission_revoked",
        )

    with pytest.raises(KnowledgeFSRevocationCommandError, match="control-space"):
        revoke(other_tenant_space.id)
    with pytest.raises(KnowledgeFSRevocationCommandError, match="revision"):
        revoke(unrevised_space.id)
    assert revoke(registered_space.id) == ()
    with pytest.raises(KnowledgeFSRevocationCommandError, match="registered"):
        revoke(unregistered_space.id)


@pytest.mark.parametrize(
    ("claims", "message"),
    [
        ({"subject": "member", "caller_kind": "interactive", "grant_id": 7}, "grant id is invalid"),
        ({"subject": "member", "caller_kind": "interactive", "grant_id": "not-a-uuid"}, "not a UUID"),
    ],
)
def test_principal_revocation_revalidates_persisted_audit_bindings(
    sqlite_session: Session,
    claims: dict[str, object],
    message: str,
) -> None:
    space = _control_space()
    sqlite_session.add_all([space, _revision(space), _claims_audit(claims, control_space_id=space.id)])
    sqlite_session.commit()

    with pytest.raises(KnowledgeFSRevocationCommandError, match=message):
        KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=sqlite_session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            subject="member",
            reason_code="permission_revoked",
            caller_kinds=("interactive",),
        )


@pytest.mark.parametrize(
    ("summary", "subject", "caller_kind", "grant_id", "message"),
    [
        (
            {"subject": "other", "caller_kind": "interactive", "grant_id": _GRANT_A},
            "member",
            "interactive",
            _GRANT_A,
            "subject binding",
        ),
        (
            {"subject": "member", "caller_kind": "service", "grant_id": _GRANT_A},
            "member",
            "interactive",
            _GRANT_A,
            "caller binding",
        ),
        (
            {"subject": "member", "caller_kind": "interactive", "grant_id": _GRANT_B},
            "member",
            "interactive",
            _GRANT_A,
            "grant binding",
        ),
        (
            {"subject": "member", "caller_kind": "interactive", "grant_id": "not-a-uuid"},
            "member",
            "interactive",
            "not-a-uuid",
            "not a UUID",
        ),
    ],
)
def test_principal_revocation_revalidates_persisted_reservation_bindings(
    sqlite_session: Session,
    summary: dict[str, object],
    subject: str,
    caller_kind: str,
    grant_id: str,
    message: str,
) -> None:
    space = _control_space()
    reservation = _reservation(
        control_space_id=space.id,
        grant_id=grant_id,
        subject=subject,
        caller_kind=caller_kind,
        summary=summary,
    )
    sqlite_session.add_all([space, _revision(space), reservation])
    sqlite_session.commit()

    with pytest.raises(KnowledgeFSRevocationCommandError, match=message):
        KnowledgeFSRevocationCommandProducer().enqueue_principal_grants(
            session=sqlite_session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            subject="member",
            reason_code="permission_revoked",
            caller_kinds=("interactive",),
        )


def test_control_space_revocation_validates_callers_and_deduplicates_grants(
    sqlite_session: Session,
) -> None:
    producer = KnowledgeFSRevocationCommandProducer()
    with pytest.raises(KnowledgeFSRevocationCommandError, match="caller kind"):
        producer.enqueue_control_space_grants(
            session=_unused_session(),
            tenant_id="tenant-1",
            control_space_id="control-1",
            reason_code="permission_revoked",
            caller_kinds=(),
        )
    with pytest.raises(KnowledgeFSRevocationCommandError, match="reason code"):
        producer.enqueue_control_space_grants(
            session=_unused_session(),
            tenant_id="tenant-1",
            control_space_id="control-1",
            reason_code="INVALID REASON",
            caller_kinds=("interactive",),
        )

    space = _control_space()
    sqlite_session.add_all(
        [
            space,
            _revision(space),
            _audit(control_space_id=space.id, grant_id=_GRANT_B, subject="member-b", jti="a"),
            _audit(control_space_id=space.id, grant_id=_grant(3), subject="excluded", jti="b"),
            _audit(control_space_id=space.id, grant_id=_grant(4), subject="api-key", caller_kind="service", jti="c"),
            _reservation(control_space_id=space.id, grant_id=_GRANT_A, subject="member-a"),
            _reservation(control_space_id=space.id, grant_id=_GRANT_B, subject="member-b"),
            _reservation(control_space_id=space.id, grant_id=_grant(5), subject="api-key", caller_kind="service"),
        ]
    )
    sqlite_session.commit()

    with _maker(sqlite_session).begin() as session:
        commands = producer.enqueue_control_space_grants(
            session=session,
            tenant_id="tenant-1",
            control_space_id=space.id,
            reason_code="permission_revoked",
            caller_kinds=("interactive", "interactive"),
            excluded_subjects=("excluded",),
        )

    payloads = [revoke_payload(command) for command in commands]
    assert [(payload["principal"], payload["grant_id"]) for payload in payloads] == [
        ("member-a", _GRANT_A),
        ("member-b", _GRANT_B),
    ]
