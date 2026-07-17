from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint import (
    AgentServiceExitCode,
    JsonLineAgentServiceTerminalEventSink,
    PrimeQAHybridLocalAgentServiceEntrypoint,
    activation_configuration_failure_event,
    cli_contract_failure_event,
    parse_exact_agent_service_cli,
)
from ts_rag_agent.config import ProjectSettings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict non-default local service entrypoint."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    sink = JsonLineAgentServiceTerminalEventSink()
    try:
        port = parse_exact_agent_service_cli(arguments)
    except ValueError:
        sink.emit(cli_contract_failure_event())
        return int(AgentServiceExitCode.CLI_CONTRACT_INVALID)

    try:
        settings = ProjectSettings()
    except ValidationError:
        sink.emit(activation_configuration_failure_event(port=port))
        return int(AgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED)

    result = PrimeQAHybridLocalAgentServiceEntrypoint(
        settings=settings,
        event_sink=sink,
    ).run(port=port)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
