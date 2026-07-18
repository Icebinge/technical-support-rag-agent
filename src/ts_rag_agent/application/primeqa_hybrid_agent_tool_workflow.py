from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import partial
from threading import Lock
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
    _answer_generator,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationCounters,
    AgentWorkflowObservationSink,
    JsonLineAgentWorkflowObservationSink,
    PrimeQAHybridAgentWorkflowObserver,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
    AGENT_TOOL_WORKFLOW_GRAPH_ID,
    AgentToolWorkflowAction,
    AgentToolWorkflowState,
    FrozenAgentToolWorkflowTransitionPolicy,
    agent_tool_workflow_state_contract,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    AnswerGeneratorPort,
    AnswerVerifierPort,
    PrimeQAHybridSidecarAgentRun,
    assemble_primeqa_hybrid_sidecar_agent_run,
    validate_primeqa_hybrid_candidate_pool,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    SidecarObservationBundle,
)
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

_IMPLEMENTATION_ID = "primeqa_hybrid_langgraph_agent_tool_workflow_v1"
_VERIFICATION_CONTEXT_MAX_RANK = 200
_TOOL_IDS = (
    "retrieve_candidate_pool",
    "compose_grounded_answer",
    "verify_grounded_answer",
)
_PUBLIC_TRACE_FIELDS = frozenset(agent_tool_workflow_state_contract()["public_trace_fields"])


class CandidatePoolRetrieverPort(Protocol):
    def retrieve(self, question: PrimeQAQuery) -> Sequence[RetrievalResult]: ...


class SidecarObservationAdapterPort(Protocol):
    def observe(
        self,
        *,
        question: PrimeQAQuery,
        candidate_pool_results: Sequence[RetrievalResult],
    ) -> SidecarObservationBundle: ...


class AnswerGeneratorFactoryPort(Protocol):
    def create(self) -> AnswerGeneratorPort: ...


class AnswerVerifierFactoryPort(Protocol):
    def create(self) -> AnswerVerifierPort: ...


class CompiledAgentToolGraphPort(Protocol):
    def invoke(self, input: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_graph(self) -> Any: ...


class AgentToolWorkflowPrivateState(TypedDict):
    request_handle: str
    runtime_query: PrimeQAQuery
    candidate_pool_results: tuple[RetrievalResult, ...]
    generation_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    sidecar_observation_bundle: SidecarObservationBundle | None
    original_answer: GeneratedAnswer | None
    verification_result: AnswerVerificationResult | None
    terminal_response: PrimeQAHybridSidecarAgentRun | None
    current_state: AgentToolWorkflowState
    visited_states: tuple[str, ...]
    tool_call_counts: dict[str, int]
    failure_stage: str | None


@dataclass(frozen=True)
class FrozenAnswerGeneratorFactory:
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY
    max_sentences: int = _DEFAULT_MAX_SENTENCES
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE

    def create(self) -> AnswerGeneratorPort:
        return _answer_generator(
            evidence_selector_name=self.evidence_selector_name,
            max_candidates_per_document=self.max_candidates_per_document,
            composition_policy_name=self.composition_policy_name,
            max_sentences=self.max_sentences,
            min_sentence_score=self.min_sentence_score,
        )


@dataclass(frozen=True)
class FrozenAnswerVerifierFactory:
    min_citations: int = 1
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE
    max_citation_rank: int = _VERIFICATION_CONTEXT_MAX_RANK

    def create(self) -> AnswerVerifierPort:
        return AnswerVerifier(
            min_citations=self.min_citations,
            min_evidence_score=self.min_evidence_score,
            max_citation_rank=self.max_citation_rank,
        )


class PrimeQAHybridAgentToolset:
    """Three tools plus deterministic context preparation and observation."""

    def __init__(
        self,
        *,
        candidate_pool_retriever: CandidatePoolRetrieverPort,
        observation_adapter: SidecarObservationAdapterPort,
        answer_generator_factory: AnswerGeneratorFactoryPort,
        answer_verifier_factory: AnswerVerifierFactoryPort,
    ) -> None:
        self._candidate_pool_retriever = candidate_pool_retriever
        self._observation_adapter = observation_adapter
        self._answer_generator_factory = answer_generator_factory
        self._answer_verifier_factory = answer_verifier_factory

    def retrieve_candidate_pool(
        self,
        question: PrimeQAQuery,
    ) -> tuple[RetrievalResult, ...]:
        results = tuple(self._candidate_pool_retriever.retrieve(question))
        validate_primeqa_hybrid_candidate_pool(results)
        return results

    def prepare_context(
        self,
        *,
        question: PrimeQAQuery,
        candidate_pool_results: tuple[RetrievalResult, ...],
    ) -> tuple[
        SidecarObservationBundle,
        tuple[RetrievalResult, ...],
        tuple[RetrievalResult, ...],
    ]:
        bundle = self._observation_adapter.observe(
            question=question,
            candidate_pool_results=candidate_pool_results,
        )
        generation_context = bundle.answer_context_for_generation()
        verification_context = tuple(
            sorted(
                (
                    result
                    for result in candidate_pool_results
                    if result.rank <= _VERIFICATION_CONTEXT_MAX_RANK
                ),
                key=lambda result: result.rank,
            )
        )
        return bundle, generation_context, verification_context

    def compose_grounded_answer(
        self,
        *,
        question: PrimeQAQuery,
        generation_context_results: tuple[RetrievalResult, ...],
    ) -> GeneratedAnswer:
        return self._answer_generator_factory.create().generate(
            question,
            generation_context_results,
        )

    def verify_grounded_answer(
        self,
        *,
        answer: GeneratedAnswer,
        verification_context_results: tuple[RetrievalResult, ...],
    ) -> AnswerVerificationResult:
        return self._answer_verifier_factory.create().verify(
            answer,
            verification_context_results,
        )

    def observe_diagnostics(
        self,
        *,
        bundle: SidecarObservationBundle,
        verification_context_results: tuple[RetrievalResult, ...],
        original_answer: GeneratedAnswer,
        verification: AnswerVerificationResult,
    ) -> PrimeQAHybridSidecarAgentRun:
        return assemble_primeqa_hybrid_sidecar_agent_run(
            bundle=bundle,
            verification_context=verification_context_results,
            original_answer=original_answer,
            verification=verification,
        )


class AgentToolWorkflowNodeExecutor:
    """Shared node semantics used by both execution engines."""

    def __init__(
        self,
        *,
        toolset: PrimeQAHybridAgentToolset,
        observer: PrimeQAHybridAgentWorkflowObserver | None = None,
    ) -> None:
        self._toolset = toolset
        self._observer = observer or PrimeQAHybridAgentWorkflowObserver(
            sink=JsonLineAgentWorkflowObservationSink()
        )
        self._transition_policy = FrozenAgentToolWorkflowTransitionPolicy()
        self._failure_lock = Lock()
        self._failure_snapshots: dict[
            tuple[object, BaseException], AgentToolWorkflowPrivateState
        ] = {}
        self._invocation_token: ContextVar[object | None] = ContextVar(
            f"primeqa_hybrid_agent_tool_invocation_{id(self)}",
            default=None,
        )
        self._last_snapshot: ContextVar[AgentToolWorkflowPrivateState | None] = ContextVar(
            f"primeqa_hybrid_agent_tool_snapshot_{id(self)}",
            default=None,
        )

    @property
    def last_snapshot(self) -> AgentToolWorkflowPrivateState | None:
        return self._last_snapshot.get()

    @property
    def observer(self) -> PrimeQAHybridAgentWorkflowObserver:
        return self._observer

    def execute_observed(
        self,
        node_id: str,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        handlers = {
            "validate_request": self.validate_request,
            "retrieve_candidate_pool": self.retrieve_candidate_pool,
            "prepare_context": self.prepare_context,
            "compose_grounded_answer": self.compose_grounded_answer,
            "verify_grounded_answer": self.verify_grounded_answer,
            "observe_diagnostics": self.observe_diagnostics,
            "finalize_response": self.finalize_response,
        }
        try:
            handler = handlers[node_id]
        except KeyError:
            raise ValueError("observed node is not in the frozen Agent graph") from None
        started_at = self._observer.node_started()
        try:
            updates = handler(state)
        except BaseException:
            snapshot = self._last_snapshot.get() or state
            self._observer.node_failed(
                node_id=node_id,
                state=snapshot,
                started_at=started_at,
            )
            raise
        self._observer.node_completed(
            node_id=node_id,
            state={**state, **updates},
            started_at=started_at,
        )
        return updates

    def begin(self, state: AgentToolWorkflowPrivateState) -> None:
        if self._invocation_token.get() is not None:
            raise RuntimeError("agent tool workflow invocation context is already active")
        self._invocation_token.set(object())
        self._last_snapshot.set(_copy_state(state))

    def end(self) -> None:
        self._invocation_token.set(None)
        self._last_snapshot.set(None)

    def consume_failure_snapshot(
        self,
        error: BaseException,
    ) -> AgentToolWorkflowPrivateState | None:
        token = self._invocation_token.get()
        if token is None:
            return None
        with self._failure_lock:
            return self._failure_snapshots.pop((token, error), None)

    def validate_request(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.VALIDATE_REQUEST)
        try:
            query = state["runtime_query"]
            if not query.id.strip():
                raise ValueError("runtime query id must be non-empty")
            if not query.text.strip():
                raise ValueError("runtime query text must be non-empty")
        except Exception as error:
            self._record_failure(state, "validate_request", error=error)
            raise
        return self._transition(state, AgentToolWorkflowAction.VALIDATE_REQUEST)

    def retrieve_candidate_pool(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL)
        counts = _increment_tool_count(state, "retrieve_candidate_pool")
        try:
            results = self._toolset.retrieve_candidate_pool(state["runtime_query"])
        except Exception as error:
            self._record_failure(
                state,
                "retrieve_candidate_pool",
                tool_call_counts=counts,
                error=error,
            )
            raise
        return self._transition(
            state,
            AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
            candidate_pool_results=results,
            tool_call_counts=counts,
        )

    def prepare_context(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.PREPARE_CONTEXT)
        try:
            bundle, generation_context, verification_context = self._toolset.prepare_context(
                question=state["runtime_query"],
                candidate_pool_results=state["candidate_pool_results"],
            )
        except Exception as error:
            self._record_failure(state, "prepare_context", error=error)
            raise
        return self._transition(
            state,
            AgentToolWorkflowAction.PREPARE_CONTEXT,
            sidecar_observation_bundle=bundle,
            generation_context_results=generation_context,
            verification_context_results=verification_context,
        )

    def compose_grounded_answer(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER)
        counts = _increment_tool_count(state, "compose_grounded_answer")
        try:
            answer = self._toolset.compose_grounded_answer(
                question=state["runtime_query"],
                generation_context_results=state["generation_context_results"],
            )
        except Exception as error:
            self._record_failure(
                state,
                "compose_grounded_answer",
                tool_call_counts=counts,
                error=error,
            )
            raise
        return self._transition(
            state,
            AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
            original_answer=answer,
            tool_call_counts=counts,
        )

    def verify_grounded_answer(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER)
        counts = _increment_tool_count(state, "verify_grounded_answer")
        answer = state["original_answer"]
        if answer is None:
            error = RuntimeError("verification requires an original answer")
            self._record_failure(
                state,
                "verify_grounded_answer",
                tool_call_counts=counts,
                error=error,
            )
            raise error
        try:
            verification = self._toolset.verify_grounded_answer(
                answer=answer,
                verification_context_results=state["verification_context_results"],
            )
        except Exception as error:
            self._record_failure(
                state,
                "verify_grounded_answer",
                tool_call_counts=counts,
                error=error,
            )
            raise
        return self._transition(
            state,
            AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER,
            verification_result=verification,
            tool_call_counts=counts,
        )

    def observe_diagnostics(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        self._assert_transition(state, AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS)
        bundle = state["sidecar_observation_bundle"]
        answer = state["original_answer"]
        verification = state["verification_result"]
        if bundle is None or answer is None or verification is None:
            error = RuntimeError("diagnostic observation requires completed answer verification")
            self._record_failure(state, "observe_diagnostics", error=error)
            raise error
        try:
            response = self._toolset.observe_diagnostics(
                bundle=bundle,
                verification_context_results=state["verification_context_results"],
                original_answer=answer,
                verification=verification,
            )
        except Exception as error:
            self._record_failure(state, "observe_diagnostics", error=error)
            raise
        return self._transition(
            state,
            AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS,
            terminal_response=response,
        )

    def route_terminal(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> Literal["complete", "refuse"]:
        response = state["terminal_response"]
        if response is None:
            error = RuntimeError("terminal routing requires an observed response")
            self._record_failure(state, "route_terminal", error=error)
            raise error
        route: Literal["complete", "refuse"] = (
            "refuse" if response.verified_answer.refused else "complete"
        )
        action = (
            AgentToolWorkflowAction.REFUSE
            if route == "refuse"
            else AgentToolWorkflowAction.COMPLETE
        )
        self._assert_transition(state, action)
        return route

    def finalize_response(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> dict[str, Any]:
        route = self.route_terminal(state)
        action = (
            AgentToolWorkflowAction.REFUSE
            if route == "refuse"
            else AgentToolWorkflowAction.COMPLETE
        )
        return self._transition(state, action)

    def _transition(
        self,
        state: AgentToolWorkflowPrivateState,
        action: AgentToolWorkflowAction,
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
        self._record_snapshot(state, transition_updates)
        return transition_updates

    def _assert_transition(
        self,
        state: AgentToolWorkflowPrivateState,
        action: AgentToolWorkflowAction,
    ) -> None:
        self._transition_policy.next_state(
            current_state=state["current_state"],
            action=action,
        )

    def _record_failure(
        self,
        state: AgentToolWorkflowPrivateState,
        stage: str,
        *,
        tool_call_counts: dict[str, int] | None = None,
        error: BaseException | None = None,
    ) -> None:
        updates: dict[str, Any] = {"failure_stage": stage}
        if tool_call_counts is not None:
            updates["tool_call_counts"] = tool_call_counts
        snapshot = _copy_state({**state, **updates})
        self._last_snapshot.set(snapshot)
        if error is not None:
            token = self._invocation_token.get()
            if token is None:
                raise RuntimeError("failure snapshot requires an active invocation token")
            with self._failure_lock:
                self._failure_snapshots[(token, error)] = snapshot

    def _record_snapshot(
        self,
        state: AgentToolWorkflowPrivateState,
        updates: Mapping[str, Any],
    ) -> None:
        self._last_snapshot.set(_copy_state({**state, **updates}))


class AgentToolWorkflowEngine(Protocol):
    engine_id: str
    compile_count: int

    def invoke(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> AgentToolWorkflowPrivateState: ...

    def topology(self) -> dict[str, Any]: ...


class DeterministicAgentToolWorkflowEngine:
    """Framework-neutral reference engine for Stage154 equivalence checks."""

    engine_id = "framework_neutral_sequential_reference"
    compile_count = 0

    def __init__(self, *, executor: AgentToolWorkflowNodeExecutor) -> None:
        self._executor = executor

    def invoke(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> AgentToolWorkflowPrivateState:
        current = _copy_state(state)
        for node_id in (
            "validate_request",
            "retrieve_candidate_pool",
            "prepare_context",
            "compose_grounded_answer",
            "verify_grounded_answer",
            "observe_diagnostics",
        ):
            current.update(self._executor.execute_observed(node_id, current))
        self._executor.route_terminal(current)
        current.update(self._executor.execute_observed("finalize_response", current))
        return current

    def topology(self) -> dict[str, Any]:
        contract = agent_tool_workflow_state_contract()
        return {
            "node_ids": list(contract["nodes"]),
            "node_count": len(contract["nodes"]),
            "conditional_edge_count": 1,
            "checkpointer_attached": False,
            "cache_attached": False,
        }


class LangGraphAgentToolWorkflowEngine:
    """Compile the Stage153 graph once without persistence, cache, or recovery."""

    engine_id = "langgraph_stategraph_1_2_9"

    def __init__(self, *, executor: AgentToolWorkflowNodeExecutor) -> None:
        self._executor = executor
        self._graph = self._compile_graph()
        self.compile_count = 1

    def invoke(
        self,
        state: AgentToolWorkflowPrivateState,
    ) -> AgentToolWorkflowPrivateState:
        return _copy_state(self._graph.invoke(state))

    def topology(self) -> dict[str, Any]:
        graph = self._graph.get_graph()
        node_ids = sorted(
            node_id
            for node_id in graph.nodes
            if node_id not in {START, END, "__start__", "__end__"}
        )
        conditional_edges = sum(bool(getattr(edge, "conditional", False)) for edge in graph.edges)
        return {
            "node_ids": node_ids,
            "node_count": len(node_ids),
            "edge_count_including_start_end": len(graph.edges),
            "conditional_edge_count": conditional_edges,
            "checkpointer_attached": getattr(self._graph, "checkpointer", None) is not None,
            "cache_attached": getattr(self._graph, "cache", None) is not None,
        }

    def _compile_graph(self) -> CompiledAgentToolGraphPort:
        builder = StateGraph(AgentToolWorkflowPrivateState)
        for node_id in agent_tool_workflow_state_contract()["nodes"]:
            builder.add_node(
                node_id,
                partial(self._executor.execute_observed, node_id),
            )
        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "retrieve_candidate_pool")
        builder.add_edge("retrieve_candidate_pool", "prepare_context")
        builder.add_edge("prepare_context", "compose_grounded_answer")
        builder.add_edge("compose_grounded_answer", "verify_grounded_answer")
        builder.add_edge("verify_grounded_answer", "observe_diagnostics")
        builder.add_conditional_edges(
            "observe_diagnostics",
            self._executor.route_terminal,
            {
                "complete": "finalize_response",
                "refuse": "finalize_response",
            },
        )
        builder.add_edge("finalize_response", END)
        return builder.compile()


@dataclass(frozen=True)
class PublicSafeAgentToolWorkflowTrace:
    protocol_id: str
    graph_id: str
    terminal_state: str
    transition_count: int
    tool_call_count: int
    retrieval_tool_call_count: int
    answer_tool_call_count: int
    verification_tool_call_count: int
    candidate_pool_depth: int
    generation_context_count: int
    verification_context_count: int
    sidecar_observation_count: int
    verified_refused: bool
    verified_citation_count: int
    citation_context_valid: bool
    diagnostics_observed: bool
    failure_stage: str | None
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _PUBLIC_TRACE_FIELDS:
            raise ValueError("workflow trace fields do not match the Stage153 allowlist")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"workflow trace contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class PrimeQAHybridAgentToolWorkflowRun:
    candidate_pool_results: tuple[RetrievalResult, ...]
    agent_run: PrimeQAHybridSidecarAgentRun
    generation_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    public_safe_trace: PublicSafeAgentToolWorkflowTrace

    @property
    def original_answer(self) -> GeneratedAnswer:
        return self.agent_run.original_answer

    @property
    def verification_result(self) -> AnswerVerificationResult:
        return self.agent_run.verification_result

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.agent_run.verified_answer


@dataclass(frozen=True)
class AgentToolWorkflowCounters:
    invocation_count: int
    completed_count: int
    refused_count: int
    failed_count: int
    current_in_flight: int
    max_observed_in_flight: int
    graph_compile_count: int


class PrimeQAHybridAgentToolWorkflow:
    """Request-isolated workflow facade over a selected execution engine."""

    def __init__(
        self,
        *,
        executor: AgentToolWorkflowNodeExecutor,
        engine: AgentToolWorkflowEngine,
    ) -> None:
        self._executor = executor
        self._engine = engine
        self._observer = executor.observer
        self._counter_lock = Lock()
        self._last_public_trace: ContextVar[PublicSafeAgentToolWorkflowTrace | None] = ContextVar(
            f"primeqa_hybrid_agent_tool_trace_{id(self)}",
            default=None,
        )
        self._invocation_count = 0
        self._completed_count = 0
        self._refused_count = 0
        self._failed_count = 0
        self._current_in_flight = 0
        self._max_observed_in_flight = 0

    @property
    def engine_id(self) -> str:
        return self._engine.engine_id

    @property
    def last_public_trace(self) -> PublicSafeAgentToolWorkflowTrace | None:
        return self._last_public_trace.get()

    def counters(self) -> AgentToolWorkflowCounters:
        with self._counter_lock:
            return AgentToolWorkflowCounters(
                invocation_count=self._invocation_count,
                completed_count=self._completed_count,
                refused_count=self._refused_count,
                failed_count=self._failed_count,
                current_in_flight=self._current_in_flight,
                max_observed_in_flight=self._max_observed_in_flight,
                graph_compile_count=self._engine.compile_count,
            )

    def topology(self) -> dict[str, Any]:
        return self._engine.topology()

    def observation_counters(self) -> AgentWorkflowObservationCounters:
        return self._observer.counters()

    def run(self, question: PrimeQAQuery) -> PrimeQAHybridAgentToolWorkflowRun:
        self._last_public_trace.set(None)
        state = _initial_state(question)
        with self._counter_lock:
            self._invocation_count += 1
            self._current_in_flight += 1
            self._max_observed_in_flight = max(
                self._max_observed_in_flight,
                self._current_in_flight,
            )
        executor_started = False
        observer_started = False
        final_state: AgentToolWorkflowPrivateState | None = None
        try:
            self._executor.begin(state)
            executor_started = True
            self._observer.begin(state)
            observer_started = True
            try:
                final_state = self._engine.invoke(state)
                response = final_state["terminal_response"]
                if response is None:
                    raise RuntimeError("workflow completed without a terminal response")
                trace = _public_trace(final_state)
                trace.to_public_dict()
                self._last_public_trace.set(trace)
                self._observer.complete(final_state)
                with self._counter_lock:
                    self._completed_count += 1
                    if response.verified_answer.refused:
                        self._refused_count += 1
                return PrimeQAHybridAgentToolWorkflowRun(
                    candidate_pool_results=final_state["candidate_pool_results"],
                    agent_run=response,
                    generation_context_results=final_state["generation_context_results"],
                    verification_context_results=final_state["verification_context_results"],
                    public_safe_trace=trace,
                )
            except Exception as error:
                snapshot = self._executor.consume_failure_snapshot(error)
                if snapshot is None:
                    snapshot = final_state or self._executor.last_snapshot or state
                trace = _public_trace(snapshot)
                self._last_public_trace.set(trace)
                if observer_started:
                    self._observer.fail(snapshot)
                with self._counter_lock:
                    self._failed_count += 1
                raise
        finally:
            if observer_started:
                self._observer.end()
            if executor_started:
                self._executor.end()
            with self._counter_lock:
                self._current_in_flight -= 1


def create_primeqa_hybrid_langgraph_agent_tool_workflow(
    *,
    candidate_pool_retriever: CandidatePoolRetrieverPort,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
    observation_sink: AgentWorkflowObservationSink | None = None,
) -> PrimeQAHybridAgentToolWorkflow:
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
    executor = _observed_executor(toolset=toolset, observation_sink=observation_sink)
    return PrimeQAHybridAgentToolWorkflow(
        executor=executor,
        engine=LangGraphAgentToolWorkflowEngine(executor=executor),
    )


def create_primeqa_hybrid_reference_agent_tool_workflow(
    *,
    toolset: PrimeQAHybridAgentToolset,
    observation_sink: AgentWorkflowObservationSink | None = None,
) -> PrimeQAHybridAgentToolWorkflow:
    executor = _observed_executor(toolset=toolset, observation_sink=observation_sink)
    return PrimeQAHybridAgentToolWorkflow(
        executor=executor,
        engine=DeterministicAgentToolWorkflowEngine(executor=executor),
    )


def create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
    *,
    toolset: PrimeQAHybridAgentToolset,
    observation_sink: AgentWorkflowObservationSink | None = None,
) -> PrimeQAHybridAgentToolWorkflow:
    executor = _observed_executor(toolset=toolset, observation_sink=observation_sink)
    return PrimeQAHybridAgentToolWorkflow(
        executor=executor,
        engine=LangGraphAgentToolWorkflowEngine(executor=executor),
    )


def agent_tool_workflow_implementation_contract() -> dict[str, Any]:
    """Describe the implemented adapter without activating a service or test gate."""

    return {
        "implementation_id": _IMPLEMENTATION_ID,
        "protocol_id": AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
        "graph_id": AGENT_TOOL_WORKFLOW_GRAPH_ID,
        "direct_dependency": "langgraph==1.2.9",
        "adapter": "langgraph.graph.StateGraph",
        "graph_compiled_once_per_workflow_instance": True,
        "new_private_state_per_request": True,
        "shared_node_executor_uses_context_local_snapshots": True,
        "operational_observability_always_attached": True,
        "operational_observability_delivery": "synchronous_validated_json_line",
        "node_latency_observed": True,
        "request_content_observed": False,
        "tool_ids": list(_TOOL_IDS),
        "successful_tool_call_count": 3,
        "conditional_route_source": "observe_diagnostics",
        "checkpointer_attached": False,
        "persistent_store_attached": False,
        "cache_attached": False,
        "tool_node_used": False,
        "llm_tool_router_used": False,
        "streaming_enabled": False,
        "human_interrupt_enabled": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
    }


def _observed_executor(
    *,
    toolset: PrimeQAHybridAgentToolset,
    observation_sink: AgentWorkflowObservationSink | None,
) -> AgentToolWorkflowNodeExecutor:
    sink = observation_sink or JsonLineAgentWorkflowObservationSink()
    return AgentToolWorkflowNodeExecutor(
        toolset=toolset,
        observer=PrimeQAHybridAgentWorkflowObserver(sink=sink),
    )


def _initial_state(question: PrimeQAQuery) -> AgentToolWorkflowPrivateState:
    return {
        "request_handle": question.id,
        "runtime_query": question,
        "candidate_pool_results": (),
        "generation_context_results": (),
        "verification_context_results": (),
        "sidecar_observation_bundle": None,
        "original_answer": None,
        "verification_result": None,
        "terminal_response": None,
        "current_state": AgentToolWorkflowState.RECEIVED,
        "visited_states": (AgentToolWorkflowState.RECEIVED.value,),
        "tool_call_counts": {tool_id: 0 for tool_id in _TOOL_IDS},
        "failure_stage": None,
    }


def _increment_tool_count(
    state: AgentToolWorkflowPrivateState,
    tool_id: str,
) -> dict[str, int]:
    counts = dict(state["tool_call_counts"])
    counts[tool_id] += 1
    if counts[tool_id] > 1:
        raise RuntimeError(f"tool {tool_id!r} exceeded its one-call request budget")
    return counts


def _public_trace(state: AgentToolWorkflowPrivateState) -> PublicSafeAgentToolWorkflowTrace:
    response = state["terminal_response"]
    bundle = state["sidecar_observation_bundle"]
    counts = state["tool_call_counts"]
    verified_answer = response.verified_answer if response is not None else None
    verification = state["verification_result"]
    return PublicSafeAgentToolWorkflowTrace(
        protocol_id=AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
        graph_id=AGENT_TOOL_WORKFLOW_GRAPH_ID,
        terminal_state=state["current_state"].value,
        transition_count=max(0, len(state["visited_states"]) - 1),
        tool_call_count=sum(counts.values()),
        retrieval_tool_call_count=counts["retrieve_candidate_pool"],
        answer_tool_call_count=counts["compose_grounded_answer"],
        verification_tool_call_count=counts["verify_grounded_answer"],
        candidate_pool_depth=len(state["candidate_pool_results"]),
        generation_context_count=len(state["generation_context_results"]),
        verification_context_count=len(state["verification_context_results"]),
        sidecar_observation_count=(len(bundle.sidecar_observations) if bundle is not None else 0),
        verified_refused=(verified_answer.refused if verified_answer is not None else False),
        verified_citation_count=(
            len(verified_answer.citations) if verified_answer is not None else 0
        ),
        citation_context_valid=(
            verification.citation_context_valid if verification is not None else False
        ),
        diagnostics_observed=response is not None,
        failure_stage=state["failure_stage"],
    )


def _copy_state(state: Mapping[str, Any]) -> AgentToolWorkflowPrivateState:
    return {
        "request_handle": state["request_handle"],
        "runtime_query": state["runtime_query"],
        "candidate_pool_results": tuple(state["candidate_pool_results"]),
        "generation_context_results": tuple(state["generation_context_results"]),
        "verification_context_results": tuple(state["verification_context_results"]),
        "sidecar_observation_bundle": state["sidecar_observation_bundle"],
        "original_answer": state["original_answer"],
        "verification_result": state["verification_result"],
        "terminal_response": state["terminal_response"],
        "current_state": state["current_state"],
        "visited_states": tuple(state["visited_states"]),
        "tool_call_counts": dict(state["tool_call_counts"]),
        "failure_stage": state["failure_stage"],
    }
