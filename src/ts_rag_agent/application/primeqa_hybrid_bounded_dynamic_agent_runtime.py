from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from threading import Lock
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    CandidatePoolRetrieverPort,
    FrozenAnswerGeneratorFactory,
    FrozenAnswerVerifierFactory,
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    CompletedThreadTurn,
    DynamicDecisionAction,
    ThreadStateLimits,
    ThreadStatePolicyViolationError,
    ThreadStateSummary,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    PrimeQAHybridSidecarAgentRun,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    SidecarObservationBundle,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    STRUCTURED_DECISION_SCHEMA_ID,
    BoundedAnswerDecision,
    StructuredDecisionRouterPort,
    StructuredRouterInvocationMetrics,
)
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

BOUNDED_DYNAMIC_AGENT_PROTOCOL_ID = "primeqa_hybrid_bounded_dynamic_agent_runtime_v1"
BOUNDED_DYNAMIC_AGENT_GRAPH_ID = "primeqa_hybrid_bounded_answer_or_refuse_graph_v1"
PRODUCTION_MAX_COMPLETED_TURNS = 4
PRODUCTION_MAX_RETAINED_BYTES = 32 * 1024
FIXED_INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I do not have enough verified evidence to answer this question."
)
_TOOL_IDS = (
    "retrieve_candidate_pool",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
)
_NODE_IDS = (
    "validate_request",
    "retrieve_candidate_pool",
    "prepare_context",
    "select_action",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
    "finalize_verified_response",
    "finalize_insufficient_evidence_refusal",
)
_PUBLIC_TRACE_FIELDS = frozenset(
    {
        "protocol_id",
        "decision_schema_id",
        "terminal_state",
        "completed_turn_count",
        "retained_state_bytes",
        "model_decision_count",
        "selected_action",
        "retrieval_call_count",
        "composition_call_count",
        "verification_call_count",
        "diagnostic_observation_count",
        "thread_state_opened",
        "thread_state_closed",
        "state_limit_rejected",
        "failure_stage",
        "retry_action_count",
        "fallback_action_count",
    }
)


class BoundedDynamicAgentState(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    RETRIEVED = "retrieved"
    CONTEXT_PREPARED = "context_prepared"
    DECIDED_COMPOSE = "decided_compose"
    DECIDED_REFUSE = "decided_refuse"
    ANSWERED = "answered"
    VERIFIED = "verified"
    OBSERVED = "observed"
    COMPLETE = "complete"
    REFUSE = "refuse"


class BoundedDynamicAgentAction(str, Enum):
    VALIDATE_REQUEST = "validate_request"
    RETRIEVE_CANDIDATE_POOL = "retrieve_candidate_pool"
    PREPARE_CONTEXT = "prepare_context"
    SELECT_COMPOSE = "select_compose"
    SELECT_REFUSE = "select_refuse"
    COMPOSE_GROUNDED_ANSWER = "compose_grounded_answer"
    VERIFY_GROUNDED_ANSWER = "verify_grounded_answer"
    OBSERVE_DIAGNOSTICS = "observe_diagnostics"
    COMPLETE = "complete"
    REFUSE = "refuse"


class InvalidBoundedDynamicAgentTransitionError(ValueError):
    """Raised before mutation when the bounded graph order is violated."""


@dataclass(frozen=True)
class BoundedAgentTerminalResult:
    verified_answer: GeneratedAnswer
    original_answer: GeneratedAnswer | None
    verification_result: AnswerVerificationResult | None
    diagnostic_run: PrimeQAHybridSidecarAgentRun | None


class BoundedDynamicAgentPrivateState(TypedDict):
    runtime_query: PrimeQAQuery
    completed_turns: tuple[CompletedThreadTurn, ...]
    candidate_pool_results: tuple[RetrievalResult, ...]
    generation_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    sidecar_observation_bundle: SidecarObservationBundle | None
    decision: BoundedAnswerDecision | None
    router_metrics: StructuredRouterInvocationMetrics | None
    original_answer: GeneratedAnswer | None
    verification_result: AnswerVerificationResult | None
    terminal_result: BoundedAgentTerminalResult | None
    current_state: BoundedDynamicAgentState
    visited_states: tuple[str, ...]
    tool_call_counts: dict[str, int]
    model_decision_count: int
    failure_stage: str | None


class CompiledBoundedAgentGraphPort(Protocol):
    def invoke(self, input: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_graph(self) -> Any: ...


class FrozenBoundedDynamicTransitionPolicy:
    """Acyclic state policy with one model-selected branch and no recovery."""

    _TRANSITIONS = {
        (
            BoundedDynamicAgentState.RECEIVED,
            BoundedDynamicAgentAction.VALIDATE_REQUEST,
        ): BoundedDynamicAgentState.VALIDATED,
        (
            BoundedDynamicAgentState.VALIDATED,
            BoundedDynamicAgentAction.RETRIEVE_CANDIDATE_POOL,
        ): BoundedDynamicAgentState.RETRIEVED,
        (
            BoundedDynamicAgentState.RETRIEVED,
            BoundedDynamicAgentAction.PREPARE_CONTEXT,
        ): BoundedDynamicAgentState.CONTEXT_PREPARED,
        (
            BoundedDynamicAgentState.CONTEXT_PREPARED,
            BoundedDynamicAgentAction.SELECT_COMPOSE,
        ): BoundedDynamicAgentState.DECIDED_COMPOSE,
        (
            BoundedDynamicAgentState.CONTEXT_PREPARED,
            BoundedDynamicAgentAction.SELECT_REFUSE,
        ): BoundedDynamicAgentState.DECIDED_REFUSE,
        (
            BoundedDynamicAgentState.DECIDED_COMPOSE,
            BoundedDynamicAgentAction.COMPOSE_GROUNDED_ANSWER,
        ): BoundedDynamicAgentState.ANSWERED,
        (
            BoundedDynamicAgentState.ANSWERED,
            BoundedDynamicAgentAction.VERIFY_GROUNDED_ANSWER,
        ): BoundedDynamicAgentState.VERIFIED,
        (
            BoundedDynamicAgentState.VERIFIED,
            BoundedDynamicAgentAction.OBSERVE_DIAGNOSTICS,
        ): BoundedDynamicAgentState.OBSERVED,
        (
            BoundedDynamicAgentState.OBSERVED,
            BoundedDynamicAgentAction.COMPLETE,
        ): BoundedDynamicAgentState.COMPLETE,
        (
            BoundedDynamicAgentState.OBSERVED,
            BoundedDynamicAgentAction.REFUSE,
        ): BoundedDynamicAgentState.REFUSE,
        (
            BoundedDynamicAgentState.DECIDED_REFUSE,
            BoundedDynamicAgentAction.REFUSE,
        ): BoundedDynamicAgentState.REFUSE,
    }

    def next_state(
        self,
        *,
        current_state: BoundedDynamicAgentState,
        action: BoundedDynamicAgentAction,
    ) -> BoundedDynamicAgentState:
        try:
            return self._TRANSITIONS[(current_state, action)]
        except KeyError:
            raise InvalidBoundedDynamicAgentTransitionError(
                f"action {action.value!r} is not allowed from {current_state.value!r}"
            ) from None


class BoundedDynamicAgentNodeExecutor:
    """Request-local node semantics shared by the compiled graph."""

    def __init__(
        self,
        *,
        toolset: PrimeQAHybridAgentToolset,
        decision_router: StructuredDecisionRouterPort,
    ) -> None:
        self._toolset = toolset
        self._decision_router = decision_router
        self._transition_policy = FrozenBoundedDynamicTransitionPolicy()
        self._last_snapshot: ContextVar[BoundedDynamicAgentPrivateState | None] = ContextVar(
            f"bounded_agent_snapshot_{id(self)}",
            default=None,
        )
        self._failure_lock = Lock()
        self._failure_snapshots: dict[BaseException, BoundedDynamicAgentPrivateState] = {}

    @property
    def last_snapshot(self) -> BoundedDynamicAgentPrivateState | None:
        return self._last_snapshot.get()

    def begin(self, state: BoundedDynamicAgentPrivateState) -> None:
        if self._last_snapshot.get() is not None:
            raise RuntimeError("bounded Agent invocation context is already active")
        self._last_snapshot.set(_copy_state(state))

    def end(self) -> None:
        self._last_snapshot.set(None)

    def consume_failure_snapshot(
        self,
        error: BaseException,
    ) -> BoundedDynamicAgentPrivateState | None:
        with self._failure_lock:
            return self._failure_snapshots.pop(error, None)

    def execute(
        self,
        node_id: str,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        handlers = {
            "validate_request": self.validate_request,
            "retrieve_candidate_pool": self.retrieve_candidate_pool,
            "prepare_context": self.prepare_context,
            "select_action": self.select_action,
            "compose_grounded_answer": self.compose_grounded_answer,
            "verify_grounded_answer": self.verify_grounded_answer,
            "observe_diagnostics": self.observe_diagnostics,
            "finalize_verified_response": self.finalize_verified_response,
            "finalize_insufficient_evidence_refusal": (self.finalize_insufficient_evidence_refusal),
        }
        try:
            handler = handlers[node_id]
        except KeyError:
            raise ValueError("node is not part of the bounded Agent graph") from None
        try:
            return handler(state)
        except BaseException as error:
            snapshot = _copy_state(self._last_snapshot.get() or state)
            snapshot["failure_stage"] = node_id
            self._last_snapshot.set(snapshot)
            with self._failure_lock:
                self._failure_snapshots[error] = snapshot
            raise

    def validate_request(self, state: BoundedDynamicAgentPrivateState) -> dict[str, Any]:
        query = state["runtime_query"]
        if not query.id.strip() or not query.text.strip():
            raise ValueError("runtime query id and text must be non-empty")
        return self._transition(state, BoundedDynamicAgentAction.VALIDATE_REQUEST)

    def retrieve_candidate_pool(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        counts = _increment_tool_count(state, "retrieve_candidate_pool")
        try:
            results = self._toolset.retrieve_candidate_pool(state["runtime_query"])
        except BaseException:
            self._record_counts(state, counts)
            raise
        return self._transition(
            state,
            BoundedDynamicAgentAction.RETRIEVE_CANDIDATE_POOL,
            candidate_pool_results=results,
            tool_call_counts=counts,
        )

    def prepare_context(self, state: BoundedDynamicAgentPrivateState) -> dict[str, Any]:
        bundle, generation, verification = self._toolset.prepare_context(
            question=state["runtime_query"],
            candidate_pool_results=state["candidate_pool_results"],
        )
        return self._transition(
            state,
            BoundedDynamicAgentAction.PREPARE_CONTEXT,
            sidecar_observation_bundle=bundle,
            generation_context_results=generation,
            verification_context_results=verification,
        )

    def select_action(self, state: BoundedDynamicAgentPrivateState) -> dict[str, Any]:
        decision_count = state["model_decision_count"] + 1
        if decision_count != 1:
            raise RuntimeError("bounded Agent permits exactly one model decision")
        try:
            decision = self._decision_router.decide(
                question=state["runtime_query"],
                generation_context_results=state["generation_context_results"],
                completed_turns=state["completed_turns"],
            )
        except BaseException:
            snapshot = _copy_state(state)
            snapshot["model_decision_count"] = decision_count
            self._last_snapshot.set(snapshot)
            raise
        action = (
            BoundedDynamicAgentAction.SELECT_COMPOSE
            if decision.action == DynamicDecisionAction.COMPOSE_GROUNDED_ANSWER.value
            else BoundedDynamicAgentAction.SELECT_REFUSE
        )
        return self._transition(
            state,
            action,
            decision=decision,
            router_metrics=self._decision_router.last_metrics,
            model_decision_count=decision_count,
        )

    def route_selected_action(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> Literal["compose", "refuse"]:
        decision = state["decision"]
        if decision is None:
            raise RuntimeError("bounded routing requires one validated decision")
        return (
            "compose"
            if decision.action == DynamicDecisionAction.COMPOSE_GROUNDED_ANSWER.value
            else "refuse"
        )

    def compose_grounded_answer(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        counts = _increment_tool_count(state, "compose_grounded_answer")
        try:
            answer = self._toolset.compose_grounded_answer(
                question=state["runtime_query"],
                generation_context_results=state["generation_context_results"],
            )
        except BaseException:
            self._record_counts(state, counts)
            raise
        return self._transition(
            state,
            BoundedDynamicAgentAction.COMPOSE_GROUNDED_ANSWER,
            original_answer=answer,
            tool_call_counts=counts,
        )

    def verify_grounded_answer(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        counts = _increment_tool_count(state, "verify_grounded_answer")
        answer = state["original_answer"]
        if answer is None:
            self._record_counts(state, counts)
            raise RuntimeError("verification requires a composed answer")
        try:
            verification = self._toolset.verify_grounded_answer(
                answer=answer,
                verification_context_results=state["verification_context_results"],
            )
        except BaseException:
            self._record_counts(state, counts)
            raise
        return self._transition(
            state,
            BoundedDynamicAgentAction.VERIFY_GROUNDED_ANSWER,
            verification_result=verification,
            tool_call_counts=counts,
        )

    def observe_diagnostics(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        counts = _increment_tool_count(state, "observe_diagnostics")
        bundle = state["sidecar_observation_bundle"]
        answer = state["original_answer"]
        verification = state["verification_result"]
        if bundle is None or answer is None or verification is None:
            self._record_counts(state, counts)
            raise RuntimeError("diagnostics require completed answer verification")
        try:
            diagnostic_run = self._toolset.observe_diagnostics(
                bundle=bundle,
                verification_context_results=state["verification_context_results"],
                original_answer=answer,
                verification=verification,
            )
        except BaseException:
            self._record_counts(state, counts)
            raise
        terminal = BoundedAgentTerminalResult(
            verified_answer=diagnostic_run.verified_answer,
            original_answer=answer,
            verification_result=verification,
            diagnostic_run=diagnostic_run,
        )
        return self._transition(
            state,
            BoundedDynamicAgentAction.OBSERVE_DIAGNOSTICS,
            terminal_result=terminal,
            tool_call_counts=counts,
        )

    def finalize_verified_response(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        terminal = state["terminal_result"]
        if terminal is None:
            raise RuntimeError("verified finalization requires a terminal result")
        action = (
            BoundedDynamicAgentAction.REFUSE
            if terminal.verified_answer.refused
            else BoundedDynamicAgentAction.COMPLETE
        )
        return self._transition(state, action)

    def finalize_insufficient_evidence_refusal(
        self,
        state: BoundedDynamicAgentPrivateState,
    ) -> dict[str, Any]:
        refusal = GeneratedAnswer(
            question_id=state["runtime_query"].id,
            answer=FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
            citations=[],
            refused=True,
        )
        terminal = BoundedAgentTerminalResult(
            verified_answer=refusal,
            original_answer=None,
            verification_result=None,
            diagnostic_run=None,
        )
        return self._transition(
            state,
            BoundedDynamicAgentAction.REFUSE,
            terminal_result=terminal,
        )

    def _transition(
        self,
        state: BoundedDynamicAgentPrivateState,
        action: BoundedDynamicAgentAction,
        **updates: Any,
    ) -> dict[str, Any]:
        next_state = self._transition_policy.next_state(
            current_state=state["current_state"],
            action=action,
        )
        transition_updates = {
            **updates,
            "current_state": next_state,
            "visited_states": (*state["visited_states"], next_state.value),
            "failure_stage": None,
        }
        self._last_snapshot.set(_copy_state({**state, **transition_updates}))
        return transition_updates

    def _record_counts(
        self,
        state: BoundedDynamicAgentPrivateState,
        counts: dict[str, int],
    ) -> None:
        snapshot = _copy_state(state)
        snapshot["tool_call_counts"] = counts
        self._last_snapshot.set(snapshot)


@dataclass(frozen=True)
class BoundedDynamicWorkflowRun:
    final_state: BoundedDynamicAgentPrivateState

    @property
    def candidate_pool_results(self) -> tuple[RetrievalResult, ...]:
        return self.final_state["candidate_pool_results"]

    @property
    def verified_answer(self) -> GeneratedAnswer:
        terminal = self.final_state["terminal_result"]
        if terminal is None:
            raise RuntimeError("bounded workflow has no terminal answer")
        return terminal.verified_answer

    @property
    def router_metrics(self) -> StructuredRouterInvocationMetrics | None:
        return self.final_state["router_metrics"]


class BoundedDynamicLangGraphWorkflow:
    """One compiled conditional graph with no checkpointer, cache, or loop."""

    def __init__(self, *, executor: BoundedDynamicAgentNodeExecutor) -> None:
        self._executor = executor
        self._graph = self._compile()
        self.compile_count = 1
        self._last_execution_snapshot: ContextVar[BoundedDynamicAgentPrivateState | None] = (
            ContextVar(f"bounded_workflow_result_{id(self)}", default=None)
        )

    @property
    def last_execution_snapshot(self) -> BoundedDynamicAgentPrivateState | None:
        return self._last_execution_snapshot.get()

    def run(
        self,
        *,
        question: PrimeQAQuery,
        completed_turns: Sequence[CompletedThreadTurn],
    ) -> BoundedDynamicWorkflowRun:
        state = _initial_state(question=question, completed_turns=completed_turns)
        self._last_execution_snapshot.set(None)
        self._executor.begin(state)
        try:
            final_state = _copy_state(self._graph.invoke(state))
            if final_state["terminal_result"] is None:
                raise RuntimeError("bounded graph completed without a terminal result")
            self._last_execution_snapshot.set(final_state)
            return BoundedDynamicWorkflowRun(final_state=final_state)
        except BaseException as error:
            snapshot = (
                self._executor.consume_failure_snapshot(error)
                or self._executor.last_snapshot
                or state
            )
            self._last_execution_snapshot.set(_copy_state(snapshot))
            raise
        finally:
            self._executor.end()

    def topology(self) -> dict[str, Any]:
        graph = self._graph.get_graph()
        node_ids = sorted(
            node_id
            for node_id in graph.nodes
            if node_id not in {START, END, "__start__", "__end__"}
        )
        conditional_edges = [
            edge for edge in graph.edges if bool(getattr(edge, "conditional", False))
        ]
        return {
            "node_ids": node_ids,
            "node_count": len(node_ids),
            "edge_count_including_start_end": len(graph.edges),
            "conditional_edge_count": len(
                {getattr(edge, "source", None) for edge in conditional_edges}
            ),
            "conditional_target_edge_count": len(conditional_edges),
            "checkpointer_attached": getattr(self._graph, "checkpointer", None) is not None,
            "cache_attached": getattr(self._graph, "cache", None) is not None,
            "compile_count": self.compile_count,
        }

    def _compile(self) -> CompiledBoundedAgentGraphPort:
        builder = StateGraph(BoundedDynamicAgentPrivateState)
        for node_id in _NODE_IDS:
            builder.add_node(node_id, partial(self._executor.execute, node_id))
        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "retrieve_candidate_pool")
        builder.add_edge("retrieve_candidate_pool", "prepare_context")
        builder.add_edge("prepare_context", "select_action")
        builder.add_conditional_edges(
            "select_action",
            self._executor.route_selected_action,
            {
                "compose": "compose_grounded_answer",
                "refuse": "finalize_insufficient_evidence_refusal",
            },
        )
        builder.add_edge("compose_grounded_answer", "verify_grounded_answer")
        builder.add_edge("verify_grounded_answer", "observe_diagnostics")
        builder.add_edge("observe_diagnostics", "finalize_verified_response")
        builder.add_edge("finalize_verified_response", END)
        builder.add_edge("finalize_insufficient_evidence_refusal", END)
        return builder.compile()


@dataclass(frozen=True)
class PublicSafeBoundedDynamicAgentTrace:
    protocol_id: str
    decision_schema_id: str
    terminal_state: str
    completed_turn_count: int
    retained_state_bytes: int
    model_decision_count: int
    selected_action: str | None
    retrieval_call_count: int
    composition_call_count: int
    verification_call_count: int
    diagnostic_observation_count: int
    thread_state_opened: bool
    thread_state_closed: bool
    state_limit_rejected: bool
    failure_stage: str | None
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _PUBLIC_TRACE_FIELDS:
            raise ValueError("bounded Agent trace does not match the Stage156 allowlist")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"bounded Agent trace contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class BoundedDynamicAgentRuntimeRun:
    workflow_run: BoundedDynamicWorkflowRun
    public_safe_trace: PublicSafeBoundedDynamicAgentTrace

    @property
    def candidate_pool_results(self) -> tuple[RetrievalResult, ...]:
        return self.workflow_run.candidate_pool_results

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.workflow_run.verified_answer


class PrimeQAHybridBoundedDynamicAgentRuntime:
    """Explicit process-local multi-turn runtime over the bounded graph."""

    def __init__(
        self,
        *,
        workflow: BoundedDynamicLangGraphWorkflow,
        thread_ledger: VolatileThreadStateLedger,
    ) -> None:
        self._workflow = workflow
        self._thread_ledger = thread_ledger
        self._last_public_trace: ContextVar[PublicSafeBoundedDynamicAgentTrace | None] = ContextVar(
            f"bounded_runtime_trace_{id(self)}", default=None
        )

    @property
    def last_public_trace(self) -> PublicSafeBoundedDynamicAgentTrace | None:
        return self._last_public_trace.get()

    def topology(self) -> dict[str, Any]:
        return self._workflow.topology()

    def open_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        self._thread_ledger.open_thread(opaque_thread_handle)
        return self._thread_ledger.public_summary(opaque_thread_handle)

    def close_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        return self._thread_ledger.close_thread(opaque_thread_handle)

    def thread_summary(self, opaque_thread_handle: str) -> ThreadStateSummary:
        return self._thread_ledger.public_summary(opaque_thread_handle)

    def run_turn(
        self,
        *,
        opaque_thread_handle: str,
        question: PrimeQAQuery,
    ) -> BoundedDynamicAgentRuntimeRun:
        self._last_public_trace.set(None)
        try:
            history = self._thread_ledger.private_history(opaque_thread_handle)
            before = self._thread_ledger.public_summary(opaque_thread_handle)
        except ThreadStatePolicyViolationError:
            trace = _trace_without_workflow(
                failure_stage="load_thread_state",
                thread_state_opened=False,
            )
            self._last_public_trace.set(trace)
            raise
        try:
            workflow_run = self._workflow.run(question=question, completed_turns=history)
        except BaseException:
            trace = _trace_from_state(
                state=self._workflow.last_execution_snapshot,
                summary=before,
                state_limit_rejected=False,
            )
            self._last_public_trace.set(trace)
            raise
        answer = workflow_run.verified_answer
        terminal_state = "refuse" if answer.refused else "complete"
        turn = CompletedThreadTurn(
            sequence_number=len(history) + 1,
            user_turn_input=question.full_question,
            verified_terminal_response=answer.answer,
            terminal_state=terminal_state,
        )
        try:
            after = self._thread_ledger.append_completed_turn(opaque_thread_handle, turn)
        except ThreadStatePolicyViolationError:
            state = _copy_state(workflow_run.final_state)
            state["failure_stage"] = "commit_thread_state"
            trace = _trace_from_state(
                state=state,
                summary=before,
                state_limit_rejected=True,
            )
            self._last_public_trace.set(trace)
            raise
        trace = _trace_from_state(
            state=workflow_run.final_state,
            summary=after,
            state_limit_rejected=False,
        )
        self._last_public_trace.set(trace)
        return BoundedDynamicAgentRuntimeRun(
            workflow_run=workflow_run,
            public_safe_trace=trace,
        )


def create_primeqa_hybrid_bounded_dynamic_agent_runtime(
    *,
    candidate_pool_retriever: CandidatePoolRetrieverPort,
    decision_router: StructuredDecisionRouterPort,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> PrimeQAHybridBoundedDynamicAgentRuntime:
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=FrozenAnswerGeneratorFactory(
            evidence_selector_name=evidence_selector_name,
            max_candidates_per_document=max_candidates_per_document,
            composition_policy_name=composition_policy_name,
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
        ),
        answer_verifier_factory=FrozenAnswerVerifierFactory(
            min_evidence_score=min_evidence_score,
        ),
    )
    return create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset(
        toolset=toolset,
        decision_router=decision_router,
        thread_ledger=VolatileThreadStateLedger(
            limits=ThreadStateLimits(
                max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
                max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            )
        ),
    )


def create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset(
    *,
    toolset: PrimeQAHybridAgentToolset,
    decision_router: StructuredDecisionRouterPort,
    thread_ledger: VolatileThreadStateLedger,
) -> PrimeQAHybridBoundedDynamicAgentRuntime:
    executor = BoundedDynamicAgentNodeExecutor(
        toolset=toolset,
        decision_router=decision_router,
    )
    return PrimeQAHybridBoundedDynamicAgentRuntime(
        workflow=BoundedDynamicLangGraphWorkflow(executor=executor),
        thread_ledger=thread_ledger,
    )


def bounded_dynamic_agent_runtime_contract() -> dict[str, Any]:
    return {
        "protocol_id": BOUNDED_DYNAMIC_AGENT_PROTOCOL_ID,
        "graph_id": BOUNDED_DYNAMIC_AGENT_GRAPH_ID,
        "decision_schema_id": STRUCTURED_DECISION_SCHEMA_ID,
        "node_ids": list(_NODE_IDS),
        "model_decision_position": "after_prepare_context",
        "conditional_branch_source": "select_action",
        "allowed_model_actions": [action.value for action in DynamicDecisionAction],
        "compose_branch": [
            "compose_grounded_answer",
            "verify_grounded_answer",
            "observe_diagnostics",
            "finalize_verified_response",
        ],
        "refuse_branch": ["finalize_insufficient_evidence_refusal"],
        "retrieval_call_count_per_turn": 1,
        "model_decision_count_per_turn": 1,
        "thread_limits": {
            "max_completed_turns": PRODUCTION_MAX_COMPLETED_TURNS,
            "max_retained_bytes": PRODUCTION_MAX_RETAINED_BYTES,
        },
        "thread_storage": "process_local_volatile_memory_only",
        "checkpointer_attached": False,
        "persistent_store_attached": False,
        "implicit_thread_creation": False,
        "overflow_behavior": "reject_before_mutation",
        "fixed_refusal_text_system_owned": True,
        "runtime_registered_as_default": False,
        "http_service_integrated": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_actions_enabled": False,
    }


def _initial_state(
    *,
    question: PrimeQAQuery,
    completed_turns: Sequence[CompletedThreadTurn],
) -> BoundedDynamicAgentPrivateState:
    return {
        "runtime_query": question,
        "completed_turns": tuple(completed_turns),
        "candidate_pool_results": (),
        "generation_context_results": (),
        "verification_context_results": (),
        "sidecar_observation_bundle": None,
        "decision": None,
        "router_metrics": None,
        "original_answer": None,
        "verification_result": None,
        "terminal_result": None,
        "current_state": BoundedDynamicAgentState.RECEIVED,
        "visited_states": (BoundedDynamicAgentState.RECEIVED.value,),
        "tool_call_counts": {tool_id: 0 for tool_id in _TOOL_IDS},
        "model_decision_count": 0,
        "failure_stage": None,
    }


def _increment_tool_count(
    state: BoundedDynamicAgentPrivateState,
    tool_id: str,
) -> dict[str, int]:
    counts = dict(state["tool_call_counts"])
    counts[tool_id] += 1
    if counts[tool_id] > 1:
        raise RuntimeError(f"tool {tool_id!r} exceeded its one-call turn budget")
    return counts


def _trace_from_state(
    *,
    state: BoundedDynamicAgentPrivateState | None,
    summary: ThreadStateSummary,
    state_limit_rejected: bool,
) -> PublicSafeBoundedDynamicAgentTrace:
    if state is None:
        return _trace_without_workflow(
            failure_stage="workflow_state_unavailable",
            thread_state_opened=summary.opened,
            completed_turn_count=summary.completed_turn_count,
            retained_state_bytes=summary.retained_state_bytes,
        )
    decision = state["decision"]
    counts = state["tool_call_counts"]
    terminal = state["current_state"] in {
        BoundedDynamicAgentState.COMPLETE,
        BoundedDynamicAgentState.REFUSE,
    }
    trace = PublicSafeBoundedDynamicAgentTrace(
        protocol_id=BOUNDED_DYNAMIC_AGENT_PROTOCOL_ID,
        decision_schema_id=STRUCTURED_DECISION_SCHEMA_ID,
        terminal_state=state["current_state"].value if terminal else "failed",
        completed_turn_count=summary.completed_turn_count,
        retained_state_bytes=summary.retained_state_bytes,
        model_decision_count=state["model_decision_count"],
        selected_action=decision.action if decision is not None else None,
        retrieval_call_count=counts["retrieve_candidate_pool"],
        composition_call_count=counts["compose_grounded_answer"],
        verification_call_count=counts["verify_grounded_answer"],
        diagnostic_observation_count=counts["observe_diagnostics"],
        thread_state_opened=summary.opened,
        thread_state_closed=False,
        state_limit_rejected=state_limit_rejected,
        failure_stage=state["failure_stage"],
    )
    trace.to_public_dict()
    return trace


def _trace_without_workflow(
    *,
    failure_stage: str,
    thread_state_opened: bool,
    completed_turn_count: int = 0,
    retained_state_bytes: int = 0,
) -> PublicSafeBoundedDynamicAgentTrace:
    trace = PublicSafeBoundedDynamicAgentTrace(
        protocol_id=BOUNDED_DYNAMIC_AGENT_PROTOCOL_ID,
        decision_schema_id=STRUCTURED_DECISION_SCHEMA_ID,
        terminal_state="failed",
        completed_turn_count=completed_turn_count,
        retained_state_bytes=retained_state_bytes,
        model_decision_count=0,
        selected_action=None,
        retrieval_call_count=0,
        composition_call_count=0,
        verification_call_count=0,
        diagnostic_observation_count=0,
        thread_state_opened=thread_state_opened,
        thread_state_closed=False,
        state_limit_rejected=False,
        failure_stage=failure_stage,
    )
    trace.to_public_dict()
    return trace


def _copy_state(state: Mapping[str, Any]) -> BoundedDynamicAgentPrivateState:
    return {
        "runtime_query": state["runtime_query"],
        "completed_turns": tuple(state["completed_turns"]),
        "candidate_pool_results": tuple(state["candidate_pool_results"]),
        "generation_context_results": tuple(state["generation_context_results"]),
        "verification_context_results": tuple(state["verification_context_results"]),
        "sidecar_observation_bundle": state["sidecar_observation_bundle"],
        "decision": state["decision"],
        "router_metrics": state["router_metrics"],
        "original_answer": state["original_answer"],
        "verification_result": state["verification_result"],
        "terminal_result": state["terminal_result"],
        "current_state": state["current_state"],
        "visited_states": tuple(state["visited_states"]),
        "tool_call_counts": dict(state["tool_call_counts"]),
        "model_decision_count": int(state["model_decision_count"]),
        "failure_stage": state["failure_stage"],
    }
