from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    CompletedThreadTurn,
    ThreadStateLimits,
    ThreadStatePolicyViolationError,
    ThreadStateSummary,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
    PRODUCTION_MAX_COMPLETED_TURNS,
    PRODUCTION_MAX_RETAINED_BYTES,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    ITERATIVE_DECISION_SCHEMA_ID,
    ClarificationKind,
    IterativeAgentDecision,
    IterativeDecisionAction,
    IterativeDecisionPhase,
    IterativeDecisionRouterPort,
    IterativeRouterInvocationMetrics,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    PrimeQAHybridSidecarAgentRun,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    SidecarObservationBundle,
)
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

BOUNDED_ITERATIVE_AGENT_PROTOCOL_ID = "primeqa_hybrid_bounded_iterative_agent_runtime_v1"
BOUNDED_ITERATIVE_AGENT_GRAPH_ID = "primeqa_hybrid_inspect_then_clarify_graph_v1"
ALTERNATE_EVIDENCE_DEPTH = 10
ITERATIVE_ALLOWED_TERMINAL_STATES = ("complete", "clarify", "refuse")

SYSTEM_CLARIFICATION_RESPONSES = {
    ClarificationKind.PRODUCT_OR_COMPONENT.value: (
        "Which product or component are you asking about?"
    ),
    ClarificationKind.VERSION_OR_BUILD.value: ("Which product version or build are you using?"),
    ClarificationKind.ERROR_CODE_OR_LOG.value: (
        "What exact error code or relevant log message do you see?"
    ),
    ClarificationKind.ENVIRONMENT_OR_PLATFORM.value: (
        "Which operating system, platform, or deployment environment are you using?"
    ),
    ClarificationKind.REQUESTED_OUTCOME.value: ("What result are you trying to achieve?"),
    ClarificationKind.REPRODUCTION_STEPS.value: ("What steps reproduce the problem?"),
}

_TOOL_IDS = (
    "retrieve_candidate_pool",
    "inspect_alternate_evidence",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
)
_NODE_IDS = (
    "validate_request",
    "retrieve_candidate_pool",
    "prepare_initial_context",
    "select_initial_action",
    "inspect_alternate_evidence",
    "select_final_action",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
    "finalize_verified_response",
    "finalize_clarification",
    "finalize_refusal",
)


class IterativeAgentState(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    RETRIEVED = "retrieved"
    INITIAL_CONTEXT_PREPARED = "initial_context_prepared"
    INITIAL_DECIDED_COMPOSE = "initial_decided_compose"
    INITIAL_DECIDED_INSPECT = "initial_decided_inspect"
    INITIAL_DECIDED_CLARIFY = "initial_decided_clarify"
    INITIAL_DECIDED_REFUSE = "initial_decided_refuse"
    ALTERNATE_EVIDENCE_INSPECTED = "alternate_evidence_inspected"
    FINAL_DECIDED_COMPOSE = "final_decided_compose"
    FINAL_DECIDED_CLARIFY = "final_decided_clarify"
    FINAL_DECIDED_REFUSE = "final_decided_refuse"
    ANSWERED = "answered"
    VERIFIED = "verified"
    OBSERVED = "observed"
    COMPLETE = "complete"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class IterativeAgentAction(str, Enum):
    VALIDATE = "validate"
    RETRIEVE = "retrieve"
    PREPARE = "prepare"
    INITIAL_COMPOSE = "initial_compose"
    INITIAL_INSPECT = "initial_inspect"
    INITIAL_CLARIFY = "initial_clarify"
    INITIAL_REFUSE = "initial_refuse"
    INSPECT = "inspect"
    FINAL_COMPOSE = "final_compose"
    FINAL_CLARIFY = "final_clarify"
    FINAL_REFUSE = "final_refuse"
    COMPOSE = "compose"
    VERIFY = "verify"
    OBSERVE = "observe"
    COMPLETE = "complete"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class InvalidIterativeAgentTransitionError(ValueError):
    """Raised before state mutation when the v2 graph order is violated."""


class AlternateEvidenceInspectorPort(Protocol):
    def inspect(
        self,
        *,
        candidate_pool_results: Sequence[RetrievalResult],
        initial_evidence_results: Sequence[RetrievalResult],
    ) -> tuple[RetrievalResult, ...]: ...


class OriginalRrfAlternateEvidenceInspector:
    """Expose original-RRF Top10 from the existing pool without retrieval."""

    def __init__(self, *, depth: int = ALTERNATE_EVIDENCE_DEPTH) -> None:
        if depth != ALTERNATE_EVIDENCE_DEPTH:
            raise ValueError("Stage168 alternate evidence depth must remain 10")
        self._depth = depth

    def inspect(
        self,
        *,
        candidate_pool_results: Sequence[RetrievalResult],
        initial_evidence_results: Sequence[RetrievalResult],
    ) -> tuple[RetrievalResult, ...]:
        _ = initial_evidence_results
        ordered = tuple(sorted(candidate_pool_results, key=lambda result: result.rank))
        alternate = ordered[: self._depth]
        if len(alternate) != self._depth:
            raise ValueError("alternate evidence inspector requires at least ten candidates")
        if tuple(result.rank for result in alternate) != tuple(range(1, self._depth + 1)):
            raise ValueError("alternate evidence inspector requires contiguous original ranks")
        return alternate


@dataclass(frozen=True)
class IterativeAgentTerminalResult:
    verified_answer: GeneratedAnswer
    terminal_state: str
    clarification_kind: str | None
    original_answer: GeneratedAnswer | None
    verification_result: AnswerVerificationResult | None
    diagnostic_run: PrimeQAHybridSidecarAgentRun | None


class IterativeAgentPrivateState(TypedDict):
    runtime_query: PrimeQAQuery
    completed_turns: tuple[CompletedThreadTurn, ...]
    candidate_pool_results: tuple[RetrievalResult, ...]
    initial_evidence_results: tuple[RetrievalResult, ...]
    alternate_evidence_results: tuple[RetrievalResult, ...]
    composition_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    sidecar_observation_bundle: SidecarObservationBundle | None
    decisions: tuple[IterativeAgentDecision, ...]
    router_metrics: tuple[IterativeRouterInvocationMetrics, ...]
    original_answer: GeneratedAnswer | None
    verification_result: AnswerVerificationResult | None
    terminal_result: IterativeAgentTerminalResult | None
    current_state: IterativeAgentState
    visited_states: tuple[str, ...]
    tool_call_counts: dict[str, int]
    model_decision_count: int
    failure_stage: str | None


class FrozenIterativeTransitionPolicy:
    """Acyclic two-decision state policy with one authorized inspection."""

    _TRANSITIONS = {
        (IterativeAgentState.RECEIVED, IterativeAgentAction.VALIDATE): (
            IterativeAgentState.VALIDATED
        ),
        (IterativeAgentState.VALIDATED, IterativeAgentAction.RETRIEVE): (
            IterativeAgentState.RETRIEVED
        ),
        (IterativeAgentState.RETRIEVED, IterativeAgentAction.PREPARE): (
            IterativeAgentState.INITIAL_CONTEXT_PREPARED
        ),
        (IterativeAgentState.INITIAL_CONTEXT_PREPARED, IterativeAgentAction.INITIAL_COMPOSE): (
            IterativeAgentState.INITIAL_DECIDED_COMPOSE
        ),
        (IterativeAgentState.INITIAL_CONTEXT_PREPARED, IterativeAgentAction.INITIAL_INSPECT): (
            IterativeAgentState.INITIAL_DECIDED_INSPECT
        ),
        (IterativeAgentState.INITIAL_CONTEXT_PREPARED, IterativeAgentAction.INITIAL_CLARIFY): (
            IterativeAgentState.INITIAL_DECIDED_CLARIFY
        ),
        (IterativeAgentState.INITIAL_CONTEXT_PREPARED, IterativeAgentAction.INITIAL_REFUSE): (
            IterativeAgentState.INITIAL_DECIDED_REFUSE
        ),
        (IterativeAgentState.INITIAL_DECIDED_INSPECT, IterativeAgentAction.INSPECT): (
            IterativeAgentState.ALTERNATE_EVIDENCE_INSPECTED
        ),
        (IterativeAgentState.ALTERNATE_EVIDENCE_INSPECTED, IterativeAgentAction.FINAL_COMPOSE): (
            IterativeAgentState.FINAL_DECIDED_COMPOSE
        ),
        (IterativeAgentState.ALTERNATE_EVIDENCE_INSPECTED, IterativeAgentAction.FINAL_CLARIFY): (
            IterativeAgentState.FINAL_DECIDED_CLARIFY
        ),
        (IterativeAgentState.ALTERNATE_EVIDENCE_INSPECTED, IterativeAgentAction.FINAL_REFUSE): (
            IterativeAgentState.FINAL_DECIDED_REFUSE
        ),
        (IterativeAgentState.INITIAL_DECIDED_COMPOSE, IterativeAgentAction.COMPOSE): (
            IterativeAgentState.ANSWERED
        ),
        (IterativeAgentState.FINAL_DECIDED_COMPOSE, IterativeAgentAction.COMPOSE): (
            IterativeAgentState.ANSWERED
        ),
        (IterativeAgentState.ANSWERED, IterativeAgentAction.VERIFY): IterativeAgentState.VERIFIED,
        (IterativeAgentState.VERIFIED, IterativeAgentAction.OBSERVE): IterativeAgentState.OBSERVED,
        (IterativeAgentState.OBSERVED, IterativeAgentAction.COMPLETE): IterativeAgentState.COMPLETE,
        (IterativeAgentState.OBSERVED, IterativeAgentAction.REFUSE): IterativeAgentState.REFUSE,
        (IterativeAgentState.INITIAL_DECIDED_CLARIFY, IterativeAgentAction.CLARIFY): (
            IterativeAgentState.CLARIFY
        ),
        (IterativeAgentState.FINAL_DECIDED_CLARIFY, IterativeAgentAction.CLARIFY): (
            IterativeAgentState.CLARIFY
        ),
        (IterativeAgentState.INITIAL_DECIDED_REFUSE, IterativeAgentAction.REFUSE): (
            IterativeAgentState.REFUSE
        ),
        (IterativeAgentState.FINAL_DECIDED_REFUSE, IterativeAgentAction.REFUSE): (
            IterativeAgentState.REFUSE
        ),
    }

    def next_state(
        self,
        *,
        current_state: IterativeAgentState,
        action: IterativeAgentAction,
    ) -> IterativeAgentState:
        try:
            return self._TRANSITIONS[(current_state, action)]
        except KeyError:
            raise InvalidIterativeAgentTransitionError(
                f"action {action.value!r} is not allowed from {current_state.value!r}"
            ) from None


class IterativeAgentNodeExecutor:
    """Node semantics for one retrieval, optional inspection, and clarification fallback."""

    def __init__(
        self,
        *,
        toolset: PrimeQAHybridAgentToolset,
        decision_router: IterativeDecisionRouterPort,
        evidence_inspector: AlternateEvidenceInspectorPort,
    ) -> None:
        self._toolset = toolset
        self._decision_router = decision_router
        self._evidence_inspector = evidence_inspector
        self._policy = FrozenIterativeTransitionPolicy()
        self._last_snapshot: ContextVar[IterativeAgentPrivateState | None] = ContextVar(
            f"iterative_agent_snapshot_{id(self)}", default=None
        )
        self._failure_lock = Lock()
        self._failure_snapshots: dict[BaseException, IterativeAgentPrivateState] = {}

    @property
    def last_snapshot(self) -> IterativeAgentPrivateState | None:
        return self._last_snapshot.get()

    def begin(self, state: IterativeAgentPrivateState) -> None:
        self._last_snapshot.set(_copy_state(state))

    def end(self) -> None:
        self._last_snapshot.set(None)

    def consume_failure_snapshot(self, error: BaseException) -> IterativeAgentPrivateState | None:
        with self._failure_lock:
            return self._failure_snapshots.pop(error, None)

    def execute(self, node_id: str, state: IterativeAgentPrivateState) -> dict[str, Any]:
        handlers = {
            "validate_request": self.validate_request,
            "retrieve_candidate_pool": self.retrieve_candidate_pool,
            "prepare_initial_context": self.prepare_initial_context,
            "select_initial_action": self.select_initial_action,
            "inspect_alternate_evidence": self.inspect_alternate_evidence,
            "select_final_action": self.select_final_action,
            "compose_grounded_answer": self.compose_grounded_answer,
            "verify_grounded_answer": self.verify_grounded_answer,
            "observe_diagnostics": self.observe_diagnostics,
            "finalize_verified_response": self.finalize_verified_response,
            "finalize_clarification": self.finalize_clarification,
            "finalize_refusal": self.finalize_refusal,
        }
        try:
            return handlers[node_id](state)
        except BaseException as error:
            snapshot = _copy_state(self._last_snapshot.get() or state)
            snapshot["failure_stage"] = node_id
            self._last_snapshot.set(snapshot)
            with self._failure_lock:
                self._failure_snapshots[error] = snapshot
            raise

    def validate_request(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        query = state["runtime_query"]
        if not query.id.strip() or not query.text.strip():
            raise ValueError("runtime query id and text must be non-empty")
        return self._transition(state, IterativeAgentAction.VALIDATE)

    def retrieve_candidate_pool(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        counts = _increment_tool_count(state, "retrieve_candidate_pool")
        results = self._toolset.retrieve_candidate_pool(state["runtime_query"])
        return self._transition(
            state,
            IterativeAgentAction.RETRIEVE,
            candidate_pool_results=results,
            tool_call_counts=counts,
        )

    def prepare_initial_context(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        bundle, generation, verification = self._toolset.prepare_context(
            question=state["runtime_query"],
            candidate_pool_results=state["candidate_pool_results"],
        )
        return self._transition(
            state,
            IterativeAgentAction.PREPARE,
            initial_evidence_results=generation,
            composition_context_results=generation,
            verification_context_results=verification,
            sidecar_observation_bundle=bundle,
        )

    def select_initial_action(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        decision = self._decide(state, IterativeDecisionPhase.INITIAL)
        actions = {
            IterativeDecisionAction.COMPOSE.value: IterativeAgentAction.INITIAL_COMPOSE,
            IterativeDecisionAction.INSPECT.value: IterativeAgentAction.INITIAL_INSPECT,
            IterativeDecisionAction.CLARIFY.value: IterativeAgentAction.INITIAL_CLARIFY,
            IterativeDecisionAction.REFUSE.value: IterativeAgentAction.INITIAL_REFUSE,
        }
        return self._transition(
            state, actions[decision.action], **self._decision_updates(state, decision)
        )

    def route_initial_action(
        self, state: IterativeAgentPrivateState
    ) -> Literal["compose", "inspect", "clarify", "refuse"]:
        decision = _last_decision(state)
        return {
            IterativeDecisionAction.COMPOSE.value: "compose",
            IterativeDecisionAction.INSPECT.value: "inspect",
            IterativeDecisionAction.CLARIFY.value: "clarify",
            IterativeDecisionAction.REFUSE.value: "refuse",
        }[decision.action]

    def inspect_alternate_evidence(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        counts = _increment_tool_count(state, "inspect_alternate_evidence")
        alternate = self._evidence_inspector.inspect(
            candidate_pool_results=state["candidate_pool_results"],
            initial_evidence_results=state["initial_evidence_results"],
        )
        combined = _deduplicated_context(
            state["initial_evidence_results"],
            alternate,
        )
        return self._transition(
            state,
            IterativeAgentAction.INSPECT,
            alternate_evidence_results=alternate,
            composition_context_results=combined,
            tool_call_counts=counts,
        )

    def select_final_action(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        decision = self._decide(state, IterativeDecisionPhase.FINAL_AFTER_INSPECTION)
        actions = {
            IterativeDecisionAction.COMPOSE.value: IterativeAgentAction.FINAL_COMPOSE,
            IterativeDecisionAction.CLARIFY.value: IterativeAgentAction.FINAL_CLARIFY,
            IterativeDecisionAction.REFUSE.value: IterativeAgentAction.FINAL_REFUSE,
        }
        return self._transition(
            state, actions[decision.action], **self._decision_updates(state, decision)
        )

    def route_final_action(
        self, state: IterativeAgentPrivateState
    ) -> Literal["compose", "clarify", "refuse"]:
        decision = _last_decision(state)
        return {
            IterativeDecisionAction.COMPOSE.value: "compose",
            IterativeDecisionAction.CLARIFY.value: "clarify",
            IterativeDecisionAction.REFUSE.value: "refuse",
        }[decision.action]

    def compose_grounded_answer(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        counts = _increment_tool_count(state, "compose_grounded_answer")
        answer = self._toolset.compose_grounded_answer(
            question=state["runtime_query"],
            generation_context_results=state["composition_context_results"],
        )
        return self._transition(
            state,
            IterativeAgentAction.COMPOSE,
            original_answer=answer,
            tool_call_counts=counts,
        )

    def verify_grounded_answer(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        counts = _increment_tool_count(state, "verify_grounded_answer")
        answer = state["original_answer"]
        if answer is None:
            raise RuntimeError("verification requires a composed answer")
        verification = self._toolset.verify_grounded_answer(
            answer=answer,
            verification_context_results=state["verification_context_results"],
        )
        return self._transition(
            state,
            IterativeAgentAction.VERIFY,
            verification_result=verification,
            tool_call_counts=counts,
        )

    def observe_diagnostics(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        counts = _increment_tool_count(state, "observe_diagnostics")
        bundle = state["sidecar_observation_bundle"]
        answer = state["original_answer"]
        verification = state["verification_result"]
        if bundle is None or answer is None or verification is None:
            raise RuntimeError("diagnostics require completed answer verification")
        diagnostic = self._toolset.observe_diagnostics(
            bundle=bundle,
            verification_context_results=state["verification_context_results"],
            original_answer=answer,
            verification=verification,
        )
        terminal = IterativeAgentTerminalResult(
            verified_answer=diagnostic.verified_answer,
            terminal_state=("refuse" if diagnostic.verified_answer.refused else "complete"),
            clarification_kind=None,
            original_answer=answer,
            verification_result=verification,
            diagnostic_run=diagnostic,
        )
        return self._transition(
            state,
            IterativeAgentAction.OBSERVE,
            terminal_result=terminal,
            tool_call_counts=counts,
        )

    def finalize_verified_response(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        terminal = state["terminal_result"]
        if terminal is None:
            raise RuntimeError("verified finalization requires a terminal result")
        action = (
            IterativeAgentAction.REFUSE
            if terminal.verified_answer.refused
            else IterativeAgentAction.COMPLETE
        )
        return self._transition(state, action)

    def finalize_clarification(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        decision = _last_decision(state)
        kind = decision.clarification_kind
        if kind is None:
            raise RuntimeError("clarification finalization requires a kind")
        answer = GeneratedAnswer(
            question_id=state["runtime_query"].id,
            answer=SYSTEM_CLARIFICATION_RESPONSES[kind],
            citations=[],
            refused=True,
        )
        terminal = IterativeAgentTerminalResult(
            verified_answer=answer,
            terminal_state="clarify",
            clarification_kind=kind,
            original_answer=None,
            verification_result=None,
            diagnostic_run=None,
        )
        return self._transition(
            state,
            IterativeAgentAction.CLARIFY,
            terminal_result=terminal,
        )

    def finalize_refusal(self, state: IterativeAgentPrivateState) -> dict[str, Any]:
        answer = GeneratedAnswer(
            question_id=state["runtime_query"].id,
            answer=FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
            citations=[],
            refused=True,
        )
        terminal = IterativeAgentTerminalResult(
            verified_answer=answer,
            terminal_state="refuse",
            clarification_kind=None,
            original_answer=None,
            verification_result=None,
            diagnostic_run=None,
        )
        return self._transition(
            state,
            IterativeAgentAction.REFUSE,
            terminal_result=terminal,
        )

    def _decide(
        self, state: IterativeAgentPrivateState, phase: IterativeDecisionPhase
    ) -> IterativeAgentDecision:
        expected_count = 1 if phase is IterativeDecisionPhase.INITIAL else 2
        if state["model_decision_count"] + 1 != expected_count:
            raise RuntimeError("iterative Agent model decision order is invalid")
        return self._decision_router.decide(
            phase=phase,
            question=state["runtime_query"],
            initial_evidence_results=state["initial_evidence_results"],
            alternate_evidence_results=state["alternate_evidence_results"],
            completed_turns=state["completed_turns"],
        )

    def _decision_updates(
        self,
        state: IterativeAgentPrivateState,
        decision: IterativeAgentDecision,
    ) -> dict[str, Any]:
        metrics = self._decision_router.last_metrics
        return {
            "decisions": (*state["decisions"], decision),
            "router_metrics": (*state["router_metrics"], *((metrics,) if metrics else ())),
            "model_decision_count": state["model_decision_count"] + 1,
        }

    def _transition(
        self,
        state: IterativeAgentPrivateState,
        action: IterativeAgentAction,
        **updates: Any,
    ) -> dict[str, Any]:
        next_state = self._policy.next_state(current_state=state["current_state"], action=action)
        result = {
            **updates,
            "current_state": next_state,
            "visited_states": (*state["visited_states"], next_state.value),
            "failure_stage": None,
        }
        self._last_snapshot.set(_copy_state({**state, **result}))
        return result


@dataclass(frozen=True)
class IterativeWorkflowRun:
    final_state: IterativeAgentPrivateState

    @property
    def verified_answer(self) -> GeneratedAnswer:
        terminal = self.final_state["terminal_result"]
        if terminal is None:
            raise RuntimeError("iterative workflow ended without a terminal result")
        return terminal.verified_answer

    @property
    def terminal_state(self) -> str:
        terminal = self.final_state["terminal_result"]
        if terminal is None:
            raise RuntimeError("iterative workflow ended without a terminal result")
        return terminal.terminal_state


class BoundedIterativeLangGraphWorkflow:
    """Compile the Stage168 acyclic graph once and invoke it request-locally."""

    def __init__(self, *, executor: IterativeAgentNodeExecutor) -> None:
        self._executor = executor
        self._compiled = self._compile()
        self._last_execution_snapshot: ContextVar[IterativeAgentPrivateState | None] = ContextVar(
            f"iterative_workflow_snapshot_{id(self)}", default=None
        )

    @property
    def last_execution_snapshot(self) -> IterativeAgentPrivateState | None:
        return self._last_execution_snapshot.get()

    def run(
        self,
        *,
        question: PrimeQAQuery,
        completed_turns: Sequence[CompletedThreadTurn],
    ) -> IterativeWorkflowRun:
        state = _initial_state(question, completed_turns)
        self._last_execution_snapshot.set(None)
        self._executor.begin(state)
        try:
            final = self._compiled.invoke(state)
            snapshot = _copy_state(final)
            self._last_execution_snapshot.set(snapshot)
            return IterativeWorkflowRun(snapshot)
        except BaseException as error:
            snapshot = self._executor.consume_failure_snapshot(error)
            self._last_execution_snapshot.set(snapshot or self._executor.last_snapshot or state)
            raise
        finally:
            self._executor.end()

    def topology(self) -> dict[str, Any]:
        graph = self._compiled.get_graph()
        return {
            "node_count": len(_NODE_IDS),
            "node_ids": list(_NODE_IDS),
            "edge_count": len(graph.edges),
            "conditional_edge_count": 2,
            "compile_count": 1,
            "checkpointer_attached": False,
            "cache_attached": False,
        }

    def _compile(self) -> Any:
        builder = StateGraph(IterativeAgentPrivateState)
        for node_id in _NODE_IDS:
            builder.add_node(
                node_id, lambda state, node_id=node_id: self._executor.execute(node_id, state)
            )
        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "retrieve_candidate_pool")
        builder.add_edge("retrieve_candidate_pool", "prepare_initial_context")
        builder.add_edge("prepare_initial_context", "select_initial_action")
        builder.add_conditional_edges(
            "select_initial_action",
            self._executor.route_initial_action,
            {
                "compose": "compose_grounded_answer",
                "inspect": "inspect_alternate_evidence",
                "clarify": "finalize_clarification",
                "refuse": "finalize_refusal",
            },
        )
        builder.add_edge("inspect_alternate_evidence", "select_final_action")
        builder.add_conditional_edges(
            "select_final_action",
            self._executor.route_final_action,
            {
                "compose": "compose_grounded_answer",
                "clarify": "finalize_clarification",
                "refuse": "finalize_refusal",
            },
        )
        builder.add_edge("compose_grounded_answer", "verify_grounded_answer")
        builder.add_edge("verify_grounded_answer", "observe_diagnostics")
        builder.add_edge("observe_diagnostics", "finalize_verified_response")
        builder.add_edge("finalize_verified_response", END)
        builder.add_edge("finalize_clarification", END)
        builder.add_edge("finalize_refusal", END)
        return builder.compile()


@dataclass(frozen=True)
class PublicSafeIterativeAgentTrace:
    protocol_id: str
    decision_schema_id: str
    terminal_state: str
    completed_turn_count: int
    retained_state_bytes: int
    model_decision_count: int
    selected_actions: tuple[str, ...]
    retrieval_call_count: int
    evidence_inspection_count: int
    composition_call_count: int
    verification_call_count: int
    diagnostic_observation_count: int
    initial_evidence_count: int
    alternate_evidence_count: int
    composition_context_count: int
    clarification_fallback_count: int
    clarification_kind: str | None
    thread_state_opened: bool
    state_limit_rejected: bool
    failure_stage: str | None
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedIterativeAgentRuntimeRun:
    workflow_run: IterativeWorkflowRun
    public_safe_trace: PublicSafeIterativeAgentTrace

    @property
    def candidate_pool_results(self) -> tuple[RetrievalResult, ...]:
        return self.workflow_run.final_state["candidate_pool_results"]

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.workflow_run.verified_answer


class PrimeQAHybridBoundedIterativeAgentRuntime:
    """Process-local v2 runtime with one inspection and clarification fallback."""

    def __init__(
        self,
        *,
        workflow: BoundedIterativeLangGraphWorkflow,
        thread_ledger: VolatileThreadStateLedger,
    ) -> None:
        self._workflow = workflow
        self._thread_ledger = thread_ledger
        self._last_public_trace: ContextVar[PublicSafeIterativeAgentTrace | None] = ContextVar(
            f"iterative_runtime_trace_{id(self)}", default=None
        )

    @property
    def last_public_trace(self) -> PublicSafeIterativeAgentTrace | None:
        return self._last_public_trace.get()

    def topology(self) -> dict[str, Any]:
        return self._workflow.topology()

    def open_thread(self, handle: str) -> ThreadStateSummary:
        self._thread_ledger.open_thread(handle)
        return self._thread_ledger.public_summary(handle)

    def close_thread(self, handle: str) -> ThreadStateSummary:
        return self._thread_ledger.close_thread(handle)

    def thread_summary(self, handle: str) -> ThreadStateSummary:
        return self._thread_ledger.public_summary(handle)

    def run_turn(
        self,
        *,
        opaque_thread_handle: str,
        question: PrimeQAQuery,
    ) -> BoundedIterativeAgentRuntimeRun:
        self._last_public_trace.set(None)
        try:
            history = self._thread_ledger.private_history(opaque_thread_handle)
            before = self._thread_ledger.public_summary(opaque_thread_handle)
        except ThreadStatePolicyViolationError:
            self._last_public_trace.set(_empty_trace("load_thread_state"))
            raise
        try:
            workflow_run = self._workflow.run(question=question, completed_turns=history)
        except BaseException:
            self._last_public_trace.set(
                _trace_from_state(self._workflow.last_execution_snapshot, before, False)
            )
            raise
        terminal = workflow_run.final_state["terminal_result"]
        if terminal is None:
            raise RuntimeError("iterative runtime cannot commit without terminal result")
        turn = CompletedThreadTurn(
            sequence_number=len(history) + 1,
            user_turn_input=question.full_question,
            verified_terminal_response=terminal.verified_answer.answer,
            terminal_state=terminal.terminal_state,
        )
        try:
            after = self._thread_ledger.append_completed_turn(opaque_thread_handle, turn)
        except ThreadStatePolicyViolationError:
            state = _copy_state(workflow_run.final_state)
            state["failure_stage"] = "commit_thread_state"
            self._last_public_trace.set(_trace_from_state(state, before, True))
            raise
        trace = _trace_from_state(workflow_run.final_state, after, False)
        self._last_public_trace.set(trace)
        return BoundedIterativeAgentRuntimeRun(workflow_run=workflow_run, public_safe_trace=trace)


def create_bounded_iterative_agent_runtime_from_toolset(
    *,
    toolset: PrimeQAHybridAgentToolset,
    decision_router: IterativeDecisionRouterPort,
    evidence_inspector: AlternateEvidenceInspectorPort | None = None,
    thread_ledger: VolatileThreadStateLedger | None = None,
) -> PrimeQAHybridBoundedIterativeAgentRuntime:
    inspector = evidence_inspector or OriginalRrfAlternateEvidenceInspector()
    ledger = thread_ledger or VolatileThreadStateLedger(
        limits=ThreadStateLimits(
            max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
            max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            allowed_terminal_states=ITERATIVE_ALLOWED_TERMINAL_STATES,
        )
    )
    executor = IterativeAgentNodeExecutor(
        toolset=toolset,
        decision_router=decision_router,
        evidence_inspector=inspector,
    )
    return PrimeQAHybridBoundedIterativeAgentRuntime(
        workflow=BoundedIterativeLangGraphWorkflow(executor=executor),
        thread_ledger=ledger,
    )


def bounded_iterative_agent_runtime_contract() -> dict[str, Any]:
    return {
        "protocol_id": BOUNDED_ITERATIVE_AGENT_PROTOCOL_ID,
        "graph_id": BOUNDED_ITERATIVE_AGENT_GRAPH_ID,
        "decision_schema_id": ITERATIVE_DECISION_SCHEMA_ID,
        "node_ids": list(_NODE_IDS),
        "initial_actions": [action.value for action in IterativeDecisionAction],
        "final_actions": [
            IterativeDecisionAction.COMPOSE.value,
            IterativeDecisionAction.CLARIFY.value,
            IterativeDecisionAction.REFUSE.value,
        ],
        "maximum_model_decisions_per_turn": 2,
        "retrieval_call_count_per_turn": 1,
        "maximum_evidence_inspection_count_per_turn": 1,
        "alternate_evidence_source": "existing_candidate_pool_original_rrf_top10",
        "second_retrieval_enabled": False,
        "query_rewrite_enabled": False,
        "decision_loop_enabled": False,
        "clarification_fallback_user_authorized": True,
        "clarification_text_system_owned": True,
        "clarification_kinds": list(SYSTEM_CLARIFICATION_RESPONSES),
        "allowed_terminal_states": list(ITERATIVE_ALLOWED_TERMINAL_STATES),
        "thread_limits": {
            "max_completed_turns": PRODUCTION_MAX_COMPLETED_TURNS,
            "max_retained_bytes": PRODUCTION_MAX_RETAINED_BYTES,
        },
        "runtime_registered_as_default": False,
        "http_service_integrated": False,
        "test_gate_opened": False,
        "retry_actions_enabled": False,
        "unapproved_fallback_actions_enabled": False,
    }


def _initial_state(
    question: PrimeQAQuery,
    completed_turns: Sequence[CompletedThreadTurn],
) -> IterativeAgentPrivateState:
    return {
        "runtime_query": question,
        "completed_turns": tuple(completed_turns),
        "candidate_pool_results": (),
        "initial_evidence_results": (),
        "alternate_evidence_results": (),
        "composition_context_results": (),
        "verification_context_results": (),
        "sidecar_observation_bundle": None,
        "decisions": (),
        "router_metrics": (),
        "original_answer": None,
        "verification_result": None,
        "terminal_result": None,
        "current_state": IterativeAgentState.RECEIVED,
        "visited_states": (IterativeAgentState.RECEIVED.value,),
        "tool_call_counts": {tool_id: 0 for tool_id in _TOOL_IDS},
        "model_decision_count": 0,
        "failure_stage": None,
    }


def _increment_tool_count(state: IterativeAgentPrivateState, tool_id: str) -> dict[str, int]:
    counts = dict(state["tool_call_counts"])
    counts[tool_id] += 1
    if counts[tool_id] > 1:
        raise RuntimeError(f"tool {tool_id!r} exceeded its one-call turn budget")
    return counts


def _deduplicated_context(
    initial: Sequence[RetrievalResult], alternate: Sequence[RetrievalResult]
) -> tuple[RetrievalResult, ...]:
    seen: set[str] = set()
    combined = []
    for result in (*initial, *alternate):
        if result.document.id not in seen:
            combined.append(result)
            seen.add(result.document.id)
    return tuple(combined)


def _last_decision(state: IterativeAgentPrivateState) -> IterativeAgentDecision:
    if not state["decisions"]:
        raise RuntimeError("iterative routing requires a validated decision")
    return state["decisions"][-1]


def _trace_from_state(
    state: IterativeAgentPrivateState | None,
    summary: ThreadStateSummary,
    state_limit_rejected: bool,
) -> PublicSafeIterativeAgentTrace:
    if state is None:
        return _empty_trace("workflow")
    counts = state["tool_call_counts"]
    terminal = state["terminal_result"]
    clarification_count = int(terminal is not None and terminal.terminal_state == "clarify")
    return PublicSafeIterativeAgentTrace(
        protocol_id=BOUNDED_ITERATIVE_AGENT_PROTOCOL_ID,
        decision_schema_id=ITERATIVE_DECISION_SCHEMA_ID,
        terminal_state=(terminal.terminal_state if terminal else state["current_state"].value),
        completed_turn_count=summary.completed_turn_count,
        retained_state_bytes=summary.retained_state_bytes,
        model_decision_count=state["model_decision_count"],
        selected_actions=tuple(decision.action for decision in state["decisions"]),
        retrieval_call_count=counts["retrieve_candidate_pool"],
        evidence_inspection_count=counts["inspect_alternate_evidence"],
        composition_call_count=counts["compose_grounded_answer"],
        verification_call_count=counts["verify_grounded_answer"],
        diagnostic_observation_count=counts["observe_diagnostics"],
        initial_evidence_count=len(state["initial_evidence_results"]),
        alternate_evidence_count=len(state["alternate_evidence_results"]),
        composition_context_count=len(state["composition_context_results"]),
        clarification_fallback_count=clarification_count,
        clarification_kind=terminal.clarification_kind if terminal else None,
        thread_state_opened=summary.opened,
        state_limit_rejected=state_limit_rejected,
        failure_stage=state["failure_stage"],
        retry_action_count=0,
        fallback_action_count=clarification_count,
    )


def _empty_trace(failure_stage: str) -> PublicSafeIterativeAgentTrace:
    return PublicSafeIterativeAgentTrace(
        protocol_id=BOUNDED_ITERATIVE_AGENT_PROTOCOL_ID,
        decision_schema_id=ITERATIVE_DECISION_SCHEMA_ID,
        terminal_state="failed",
        completed_turn_count=0,
        retained_state_bytes=0,
        model_decision_count=0,
        selected_actions=(),
        retrieval_call_count=0,
        evidence_inspection_count=0,
        composition_call_count=0,
        verification_call_count=0,
        diagnostic_observation_count=0,
        initial_evidence_count=0,
        alternate_evidence_count=0,
        composition_context_count=0,
        clarification_fallback_count=0,
        clarification_kind=None,
        thread_state_opened=False,
        state_limit_rejected=False,
        failure_stage=failure_stage,
    )


def _copy_state(state: Mapping[str, Any]) -> IterativeAgentPrivateState:
    return {
        "runtime_query": state["runtime_query"],
        "completed_turns": tuple(state["completed_turns"]),
        "candidate_pool_results": tuple(state["candidate_pool_results"]),
        "initial_evidence_results": tuple(state["initial_evidence_results"]),
        "alternate_evidence_results": tuple(state["alternate_evidence_results"]),
        "composition_context_results": tuple(state["composition_context_results"]),
        "verification_context_results": tuple(state["verification_context_results"]),
        "sidecar_observation_bundle": state["sidecar_observation_bundle"],
        "decisions": tuple(state["decisions"]),
        "router_metrics": tuple(state["router_metrics"]),
        "original_answer": state["original_answer"],
        "verification_result": state["verification_result"],
        "terminal_result": state["terminal_result"],
        "current_state": state["current_state"],
        "visited_states": tuple(state["visited_states"]),
        "tool_call_counts": dict(state["tool_call_counts"]),
        "model_decision_count": int(state["model_decision_count"]),
        "failure_stage": state["failure_stage"],
    }
