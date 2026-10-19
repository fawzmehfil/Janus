"""Fresh gated commit-reveal release generation for hidden-v2."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from janus.core.config import MarketConfig
from janus.core.randomness import NamedRNG
from janus.core.replay import canonical_json
from janus.interventions.actions import InterventionConfig
from janus.scenarios.generator import CandidateGeneratorConfig, CandidateWorld, generate_candidates
from janus.scenarios.hidden import generate_ood_candidates
from janus.scenarios.matching import AmbiguityBundle, match_ambiguity_bundles
from janus.simulation.runner import DynamicsConfig
from janus.v2 import BENCHMARK_VERSION
from janus.v2.audit import development_gate_report
from janus.v2.contracts import decision_schema_bytes
from janus.v2.design import (
    audit_action_coverage,
    audit_matched_contrasts,
    audit_structural_ood_reversals,
)
from janus.v2.oracle import load_or_evaluate_v2_oracles, primary_oracle


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write(path: Path, payload: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    if private:
        path.chmod(0o600)


def _frozen_input_hashes(project_root: Path) -> dict[str, str]:
    fixed = (
        "uv.lock",
        "configs/experiments/hidden-v2.json",
        "configs/experiments/v2-development.json",
        "configs/experiments/v2-analysis-preregistration.json",
        "configs/models/v2-candidates.json",
        "configs/pricing/v2-pending.json",
        "configs/prompts/primary-v2.txt",
        "configs/schemas/decision-v2.json",
        "src/janus/evaluation/observations.py",
        "src/janus/interventions/actions.py",
        "src/janus/core/ledger.py",
        "src/janus/simulation/runner.py",
        "src/janus/v2/contracts.py",
        "src/janus/v2/providers.py",
        "src/janus/v2/harness.py",
        "src/janus/v2/collection.py",
        "src/janus/v2/oracle.py",
        "src/janus/v2/statistics.py",
        "src/janus/v2/unblind.py",
        "manifests/v2/provider-admission.json",
        "manifests/v2/prompt-review-approval.json",
        "reports/v2/precision.json",
        "reports/v2/action-coverage.json",
        "reports/v2/matched-contrasts.json",
        "reports/v2/structural-ood.json",
        "artifacts/robustness/report.json",
        "docs/v2/ROBUSTNESS_AND_FALSIFICATION.md",
    )
    paths = [project_root / value for value in fixed]
    paths.extend(sorted((project_root / "manifests/v2/baselines").glob("*")))
    missing = [str(path.relative_to(project_root)) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"v2 freeze inputs are incomplete: {missing}")
    return {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def prepare_v2_seed(
    project_root: Path,
    private_output: Path,
    commitment_path: Path,
    *,
    code_commit: str,
) -> str:
    gates = development_gate_report(project_root)
    if not gates["passed"]:
        failed = [row["name"] for row in gates["checks"] if not row["passed"]]
        raise ValueError(f"hidden-v2 seed is blocked by development gates: {failed}")
    private_output.mkdir(parents=True, exist_ok=True)
    private_output.chmod(0o700)
    seed_path = private_output / "seed_manifest.json"
    if seed_path.exists():
        seed = _read(seed_path)
        if seed["code_commit"] != code_commit:
            raise ValueError("existing v2 seed belongs to a different code revision")
    else:
        seed = {
            "schema_version": "janus-hidden-v2-seed-v1",
            "root_seed": secrets.randbits(63),
            "code_commit": code_commit,
            "process": "Python secrets.randbits(63) backed by the host OS CSPRNG",
        }
        _write(seed_path, seed, private=True)
    digest = hashlib.sha256(canonical_json(seed).encode()).hexdigest()
    frozen_inputs = _frozen_input_hashes(project_root)
    _write(
        commitment_path,
        {
            "schema_version": "janus-hidden-v2-seed-commitment-v1",
            "sha256": digest,
            "code_commit": code_commit,
            "private_seed_manifest": "artifacts/hidden-v2/seed_manifest.json",
            "reveal_policy": "reveal only after all four response manifests are committed",
            "development_gate_sha256": hashlib.sha256(canonical_json(gates).encode()).hexdigest(),
            "frozen_input_sha256": frozen_inputs,
        },
    )
    return digest


def _private_world(
    opaque_id: str,
    bundle_id: str,
    stratum: str,
    world: CandidateWorld,
    oracle_file: str,
    oracle_hash: str,
) -> dict[str, Any]:
    return {
        "opaque_world_id": opaque_id,
        "opaque_bundle_id": bundle_id,
        "stratum": stratum,
        "internal_world_id": world.world_id,
        "family": world.family.value,
        "scenario": world.scenario.model_dump(mode="json"),
        "decision_state": asdict(world.decision_state),
        "observation_sha256": hashlib.sha256(world.serialized_observation).hexdigest(),
        "oracle_file": oracle_file,
        "oracle_sha256": oracle_hash,
    }


def _candidate_hash(world: CandidateWorld) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "world_id": world.world_id,
                "scenario": world.scenario.model_dump(mode="json"),
                "decision_state": asdict(world.decision_state),
                "observation": json.loads(world.serialized_observation),
                "summary_features": world.summary_features,
            }
        ).encode()
    ).hexdigest()


def _replicate_schedule(bundle_count: int, rng: Any) -> dict[tuple[int, int], tuple[str, ...]]:
    fourth_member_bundles = {
        int(value) for value in rng.permutation(bundle_count)[: bundle_count // 2]
    }
    schedule: dict[tuple[int, int], tuple[str, ...]] = {}
    for bundle_index in range(bundle_count):
        extra_members = {0, 1, 2}
        if bundle_index in fourth_member_bundles:
            extra_members.add(3)
        for member_index in range(4):
            schedule[(bundle_index, member_index)] = (
                ("replicate-00", "replicate-01")
                if member_index in extra_members
                else ("replicate-00",)
            )
    return schedule


def generate_v2_hidden_release(
    project_root: Path,
    private_output: Path,
    public_manifest_path: Path,
    commitment_path: Path,
    *,
    code_commit: str,
    workers: int = 4,
) -> bool:
    gates = development_gate_report(project_root)
    if not gates["passed"]:
        raise ValueError("hidden-v2 generation is blocked until every development gate passes")
    seed_path = private_output / "seed_manifest.json"
    if not seed_path.exists() or not commitment_path.exists():
        raise ValueError("prepare and commit the fresh v2 seed before generation")
    seed = _read(seed_path)
    commitment = _read(commitment_path)
    if seed["code_commit"] != code_commit or commitment["code_commit"] != code_commit:
        raise ValueError("v2 seed commitment does not match the frozen code commit")
    expected = hashlib.sha256(canonical_json(seed).encode()).hexdigest()
    if expected != commitment["sha256"]:
        raise ValueError("v2 seed commitment verification failed")
    current_inputs = _frozen_input_hashes(project_root)
    if current_inputs != commitment.get("frozen_input_sha256"):
        raise ValueError("a frozen v2 input changed after the seed commitment")

    plan_path = project_root / "configs/experiments/hidden-v2.json"
    prompt_path = project_root / "configs/prompts/primary-v2.txt"
    plan = _read(plan_path)
    root_seed = int(seed["root_seed"])
    id_config = CandidateGeneratorConfig(
        candidates_per_family=int(plan["id_candidates_per_family"]), root_seed=root_seed
    )
    ood_config = CandidateGeneratorConfig(
        candidates_per_family=int(plan["ood_candidates_per_family"]), root_seed=root_seed
    )
    id_candidates = generate_candidates("hidden-v2-id-private", id_config)
    ood_candidates = generate_ood_candidates("hidden-v2-ood-private", ood_config)
    id_bundles = match_ambiguity_bundles(
        id_candidates,
        bundle_count=int(plan["id_bundles"]),
        split="hidden-v2-id-private",
    )
    ood_bundles = match_ambiguity_bundles(
        ood_candidates,
        bundle_count=int(plan["ood_bundles"]),
        split="hidden-v2-ood-private",
    )
    all_bundles: tuple[tuple[str, AmbiguityBundle], ...] = tuple(
        [("id", value) for value in id_bundles] + [("ood", value) for value in ood_bundles]
    )
    maximum_distance = max(bundle.maximum_pairwise_distance for _, bundle in all_bundles)
    if maximum_distance > float(plan["maximum_bundle_distance"]):
        raise ValueError("fresh v2 candidates failed the frozen ambiguity-distance gate")

    accepted_ids = {world.world_id for _, bundle in all_bundles for world in bundle.worlds}
    candidate_audit = {
        "schema_version": "janus-v2-hidden-candidate-audit-v1",
        "generation_configs": {
            "id": id_config.model_dump(mode="json"),
            "ood": ood_config.model_dump(mode="json"),
        },
        "acceptance_criteria": {
            "selection": "deterministic minimum-distance ambiguity-bundle matching",
            "maximum_bundle_distance": plan["maximum_bundle_distance"],
            "structural_ood_maximum_distance": plan["structural_ood_maximum_distance"],
            "structural_ood_minimum_pairs": plan["structural_ood_minimum_pairs"],
            "structural_ood_minimum_families": plan["structural_ood_minimum_families"],
        },
        "candidate_counts": {
            "id": len(id_candidates),
            "ood": len(ood_candidates),
            "accepted": len(accepted_ids),
            "rejected": len(id_candidates) + len(ood_candidates) - len(accepted_ids),
        },
        "accepted_candidate_sha256": {
            world.world_id: _candidate_hash(world)
            for world in (*id_candidates, *ood_candidates)
            if world.world_id in accepted_ids
        },
        "rejected_candidates": {
            world.world_id: {
                "sha256": _candidate_hash(world),
                "reason": "not selected by frozen deterministic bundle allocation",
            }
            for world in (*id_candidates, *ood_candidates)
            if world.world_id not in accepted_ids
        },
        "manual_post_model_selection": False,
    }
    candidate_audit_path = private_output / "candidate_audit.json"
    _write(candidate_audit_path, candidate_audit, private=True)

    worlds = tuple(world for _, bundle in all_bundles for world in bundle.worlds)
    oracle_payloads, oracle_hashes = load_or_evaluate_v2_oracles(
        worlds,
        output_directory=private_output,
        root_seed=root_seed,
        rollout_count=int(plan["oracle_rollouts"]),
        workers=workers,
        market=MarketConfig(initial_cash_multiple=100),
        intervention=InterventionConfig(),
        dynamics=DynamicsConfig(recovery_periods=20),
    )
    reversal = audit_structural_ood_reversals(
        tuple(world for bundle in id_bundles for world in bundle.worlds),
        tuple(world for bundle in ood_bundles for world in bundle.worlds),
        {key: primary_oracle(value) for key, value in oracle_payloads.items()},
        {key: primary_oracle(value) for key, value in oracle_payloads.items()},
        maximum_distance=float(plan["structural_ood_maximum_distance"]),
        minimum_pairs=int(plan["structural_ood_minimum_pairs"]),
    )
    if not reversal["passed"] or reversal["family_count"] < int(
        plan["structural_ood_minimum_families"]
    ):
        raise ValueError("fresh hidden-v2 candidates failed the structural OOD reversal gate")
    _write(private_output / "structural_ood_audit.json", reversal, private=True)
    primary_oracles = {key: primary_oracle(value) for key, value in oracle_payloads.items()}
    hidden_action_coverage = audit_action_coverage(
        tuple(bundle for _, bundle in all_bundles), primary_oracles
    )
    hidden_contrasts = audit_matched_contrasts(
        tuple(bundle for _, bundle in all_bundles), primary_oracles
    )
    if not hidden_action_coverage["passed"] or not hidden_contrasts["passed"]:
        raise ValueError("fresh hidden-v2 candidates failed action coverage or contrast gates")
    hidden_design_audit_path = private_output / "design_audit.json"
    _write(
        hidden_design_audit_path,
        {
            "schema_version": "janus-v2-hidden-design-audit-v1",
            "action_coverage": hidden_action_coverage,
            "matched_contrasts": hidden_contrasts,
        },
        private=True,
    )

    rng = NamedRNG(root_seed).generator("model_replicates", "hidden-v2-public-order")
    bundle_order = tuple(int(value) for value in rng.permutation(len(all_bundles)))
    schedule = _replicate_schedule(len(all_bundles), rng)
    public_bundles = []
    private_worlds = []
    world_count = 0
    trial_count = 0
    for public_bundle_index, source_index in enumerate(bundle_order):
        stratum, bundle = all_bundles[source_index]
        bundle_id = f"janus-v2-hidden-bundle-{public_bundle_index:03d}"
        member_order = tuple(int(value) for value in rng.permutation(4))
        public_worlds = []
        for public_member_index, source_member_index in enumerate(member_order):
            world = bundle.worlds[source_member_index]
            world_id = f"janus-v2-hidden-world-{world_count:03d}"
            world_count += 1
            replicates = schedule[(public_bundle_index, public_member_index)]
            trial_count += len(replicates)
            oracle_name = f"{world.world_id}-v2-k{int(plan['oracle_rollouts'])}.json"
            public_worlds.append(
                {
                    "world_id": world_id,
                    "observation": json.loads(world.serialized_observation),
                    "observation_sha256": hashlib.sha256(world.serialized_observation).hexdigest(),
                    "replicate_ids": list(replicates),
                }
            )
            private_worlds.append(
                _private_world(
                    world_id,
                    bundle_id,
                    stratum,
                    world,
                    f"oracles/{oracle_name}",
                    oracle_hashes[oracle_name],
                )
            )
        public_bundles.append({"bundle_id": bundle_id, "worlds": public_worlds})
    if trial_count != int(plan["planned_trials_per_model"]):
        raise AssertionError("v2 replicate schedule does not equal the frozen call count")

    private_manifest = {
        "schema_version": "janus-hidden-v2-private-v1",
        "code_commit": code_commit,
        "seed_commitment_sha256": expected,
        "worlds": private_worlds,
        "maximum_bundle_distance": maximum_distance,
        "structural_ood_audit_sha256": hashlib.sha256(
            (private_output / "structural_ood_audit.json").read_bytes()
        ).hexdigest(),
        "candidate_audit_file": "candidate_audit.json",
        "candidate_audit_sha256": hashlib.sha256(candidate_audit_path.read_bytes()).hexdigest(),
        "design_audit_file": "design_audit.json",
        "design_audit_sha256": hashlib.sha256(hidden_design_audit_path.read_bytes()).hexdigest(),
    }
    private_path = private_output / "private_manifest.json"
    _write(private_path, private_manifest, private=True)
    baseline_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((project_root / "manifests/v2/baselines").glob("*"))
        if path.is_file()
    }
    public_manifest = {
        "schema_version": "janus-hidden-v2-public-v1",
        "benchmark_version": BENCHMARK_VERSION,
        "code_commit": code_commit,
        "bundle_count": len(all_bundles),
        "world_count": world_count,
        "trials_per_model": trial_count,
        "seed_commitment_sha256": expected,
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "decision_schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
        "frozen_baseline_sha256": baseline_hashes,
        "bundles": public_bundles,
    }
    _write(public_manifest_path, public_manifest)
    _write(
        commitment_path,
        {
            **commitment,
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "public_manifest_sha256": hashlib.sha256(public_manifest_path.read_bytes()).hexdigest(),
            "private_manifest_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
            "world_count": world_count,
            "bundle_count": len(all_bundles),
            "trials_per_model": trial_count,
            "maximum_bundle_distance": maximum_distance,
        },
    )
    return True
