from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .primeqa_hybrid_train_history_isolation_protocol import (
    Stage165ArmObservation,
    Stage165PairedWorkloadPlan,
    Stage165SyntheticThread,
)

STAGE165_THREADS_PER_SHARD = 12
STAGE165_EXPECTED_SHARD_COUNT = 12
STAGE165_EXPECTED_FULL_SHARD_COUNT = 11
STAGE165_EXPECTED_FINAL_SHARD_THREAD_COUNT = 9
STAGE165_EXPECTED_FINAL_SHARD_PAIR_COUNT = 34
STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256 = (
    "cd543f5c1ddd19f58483cb7ba4763c226e601cfb04ac24abe71d8f7ea42dfc23"
)


@dataclass(frozen=True)
class Stage165ShardSpec:
    ordinal: int
    threads: tuple[Stage165SyntheticThread, ...]
    private_identity_assignment_sha256: str

    @property
    def start_thread_ordinal(self) -> int:
        return self.threads[0].ordinal

    @property
    def end_thread_ordinal(self) -> int:
        return self.threads[-1].ordinal

    @property
    def pair_count(self) -> int:
        return sum(len(thread.samples) for thread in self.threads)

    @property
    def arm_row_count(self) -> int:
        return self.pair_count * 2

    def public_summary(self) -> dict[str, Any]:
        return {
            "shard_ordinal": self.ordinal,
            "start_thread_ordinal": self.start_thread_ordinal,
            "end_thread_ordinal": self.end_thread_ordinal,
            "thread_count": len(self.threads),
            "pair_count": self.pair_count,
            "agent_turn_count": self.arm_row_count,
            "private_identity_assignment_sha256": (self.private_identity_assignment_sha256),
        }


@dataclass(frozen=True)
class Stage165ShardingPlan:
    shards: tuple[Stage165ShardSpec, ...]
    assignment_sha256: str

    @property
    def threads(self) -> tuple[Stage165SyntheticThread, ...]:
        return tuple(thread for shard in self.shards for thread in shard.threads)

    @property
    def pair_count(self) -> int:
        return sum(shard.pair_count for shard in self.shards)

    @property
    def arm_row_count(self) -> int:
        return sum(shard.arm_row_count for shard in self.shards)

    def shard(self, ordinal: int) -> Stage165ShardSpec:
        if not 1 <= ordinal <= len(self.shards):
            raise ValueError(f"Stage165 shard ordinal must be 1..{len(self.shards)}")
        return self.shards[ordinal - 1]

    def public_summary(self) -> dict[str, Any]:
        return {
            "design": "contiguous_synthetic_thread_process_sharding",
            "execution_order": "strictly_sequential_shard_ordinal",
            "threads_per_full_shard": STAGE165_THREADS_PER_SHARD,
            "shard_count": len(self.shards),
            "full_shard_count": sum(
                len(shard.threads) == STAGE165_THREADS_PER_SHARD for shard in self.shards
            ),
            "final_shard_thread_count": len(self.shards[-1].threads),
            "synthetic_thread_count": len(self.threads),
            "pair_count": self.pair_count,
            "agent_turn_count": self.arm_row_count,
            "process_count": len(self.shards),
            "model_load_count": len(self.shards),
            "warmup_generation_count": len(self.shards),
            "retry": False,
            "fallback": False,
            "cuda_empty_cache": False,
            "timeout": False,
            "continue_after_failed_shard": False,
            "split_boundary": "synthetic_thread_only",
            "assignment_sha256": self.assignment_sha256,
            "shards": [shard.public_summary() for shard in self.shards],
        }


def build_stage165_sharding_plan(
    workload: Stage165PairedWorkloadPlan,
) -> Stage165ShardingPlan:
    shards = []
    for offset in range(0, len(workload.threads), STAGE165_THREADS_PER_SHARD):
        threads = workload.threads[offset : offset + STAGE165_THREADS_PER_SHARD]
        identity_rows = [
            sample.private_identity_sha256 for thread in threads for sample in thread.samples
        ]
        shards.append(
            Stage165ShardSpec(
                ordinal=len(shards) + 1,
                threads=threads,
                private_identity_assignment_sha256=_canonical_json_sha256(identity_rows),
            )
        )
    assignment = [shard.public_summary() for shard in shards]
    return Stage165ShardingPlan(
        shards=tuple(shards),
        assignment_sha256=_canonical_json_sha256(assignment),
    )


def expected_stage165_shard_observation_keys(
    *,
    shard: Stage165ShardSpec,
    workload: Stage165PairedWorkloadPlan,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (sample.private_identity_sha256, arm)
        for thread in shard.threads
        for sample in thread.samples
        for arm in workload.arm_order(sample)
    )


def validate_stage165_shard_observations(
    *,
    shard: Stage165ShardSpec,
    workload: Stage165PairedWorkloadPlan,
    observations: Sequence[Stage165ArmObservation],
) -> None:
    actual = tuple(
        (observation.private_identity_sha256, observation.arm) for observation in observations
    )
    expected = expected_stage165_shard_observation_keys(
        shard=shard,
        workload=workload,
    )
    if actual != expected:
        raise ValueError("Stage165 shard observation sequence is not exact")
    expected_threads = {
        sample.private_identity_sha256: (thread.ordinal, position)
        for thread in shard.threads
        for position, sample in enumerate(thread.samples, start=1)
    }
    for observation in observations:
        thread_ordinal, turn_position = expected_threads[observation.private_identity_sha256]
        if (
            observation.synthetic_thread_ordinal != thread_ordinal
            or observation.synthetic_turn_position != turn_position
        ):
            raise ValueError("Stage165 shard thread position is not exact")


def stage165_observation_sequence_sha256(
    observations: Sequence[Stage165ArmObservation],
) -> str:
    return _canonical_json_sha256([observation.to_private_dict() for observation in observations])


def load_stage165_observation_jsonl(
    path: Path,
) -> tuple[Stage165ArmObservation, ...]:
    resolved = path.expanduser().resolve(strict=True)
    expected_fields = {field.name for field in fields(Stage165ArmObservation)}
    observations = []
    with resolved.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Stage165 shard JSONL contains blank line {line_number}")
            value = json.loads(stripped)
            if not isinstance(value, Mapping):
                raise ValueError("Stage165 shard JSONL row must be an object")
            if set(value) != expected_fields:
                raise ValueError("Stage165 shard JSONL row fields are not exact")
            arm = value.get("arm")
            if arm not in {"isolated", "synthetic_history"}:
                raise ValueError("Stage165 shard JSONL arm is invalid")
            observations.append(Stage165ArmObservation(**dict(value)))
    return tuple(observations)


def write_stage165_observation_jsonl_row(
    *,
    path: Path,
    observation: Stage165ArmObservation,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                observation.to_private_dict(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        handle.write("\n")
        handle.flush()


def file_sha256(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return _canonical_json_sha256(value)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
