from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    BoundedDynamicAgentServiceExitCode,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
    parse_exact_bounded_dynamic_agent_service_cli,
)
from ts_rag_agent.config import ProjectSettings


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        port = parse_exact_bounded_dynamic_agent_service_cli(arguments)
    except ValueError:
        return int(BoundedDynamicAgentServiceExitCode.CLI_CONTRACT_INVALID)
    try:
        settings = ProjectSettings()
    except ValidationError:
        return int(BoundedDynamicAgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED)
    result = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(settings=settings).run(port=port)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
