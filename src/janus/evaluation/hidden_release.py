"""Committed-seed hidden-set generation with opaque public identifiers."""

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
from janus.evaluation.harness import decision_schema_bytes
from janus.evaluation.phase0d_runner import _load_or_evaluate
from janus.interventions.actions import InterventionConfig
from janus.scenarios.generator import CandidateGeneratorConfig, CandidateWorld, generate_candidates
from janus.scenarios.hidden import generate_ood_candidates
from janus.scenarios.matching import AmbiguityBundle, match_ambiguity_bundles
from janus.simulation.runner import DynamicsConfig


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write_json(path: Path, payload: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n")
    if private:
        path.chmod(0o600)


def prepare_hidden_seed(
    private_output: Path,
    commitment_path: Path,
    *,
    code_commit: str,
) -> str:
    private_output.mkdir(parents=True, exist_ok=True)
    private_output.chmod(0o700)
    seed_path = private_output / "seed_manifest.json"
    if seed_path.exists():
        seed_manifest = _read_json(seed_path)
        if seed_manifest["code_commit"] != code_commit:
            raise ValueError("existing hidden seed was committed against a different code revision")
    else:
        seed_manifest = {
            "schema_version": "janus-hidden-seed-v1",
            "root_seed": secrets.randbits(63),
            "code_commit": code_commit,
            "process": "Python secrets.randbits(63) backed by the host OS CSPRNG",
        }
        _write_json(seed_path, seed_manifest, private=True)
    canonical = canonical_json(seed_manifest).encode()
    commitment = hashlib.sha256(canonical).hexdigest()
    _write_json(
        commitment_path,
        {
            "schema_version": "janus-hidden-seed-commitment-v1",
            "sha256": commitment,
            "code_commit": code_commit,
            "private_seed_manifest": "artifacts/hidden-v1/seed_manifest.json",
            "reveal_policy": "reveal only after model response manifests are committed",
        },
    )
    return commitment


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


def generate_hidden_release(
    private_output: Path,
    public_manifest_path: Path,
    commitment_path: Path,
    plan_path: Path,
    prompt_path: Path,
    *,
    code_commit: str,
    workers: int = 4,
) -> bool:
    seed_path = private_output / "seed_manifest.json"
    if not seed_path.exists() or not commitment_path.exists():
        raise ValueError("prepare and commit the hidden seed before generation")
    seed_manifest = _read_json(seed_path)
    commitment = _read_json(commitment_path)
    if seed_manifest["code_commit"] != code_commit or commitment["code_commit"] != code_commit:
        raise ValueError("hidden seed commitment does not match the frozen code commit")
    expected = hashlib.sha256(canonical_json(seed_manifest).encode()).hexdigest()
    if commitment["sha256"] != expected:
        raise ValueError("hidden seed commitment verification failed")
    plan = _read_json(plan_path)
    root_seed = int(seed_manifest["root_seed"])

    id_config = CandidateGeneratorConfig(
        candidates_per_family=int(plan["id_candidates_per_family"]),
        root_seed=root_seed,
    )
    ood_config = CandidateGeneratorConfig(
        candidates_per_family=int(plan["ood_candidates_per_family"]),
        root_seed=root_seed,
    )
    id_candidates = generate_candidates("hidden-id-private", id_config)
    ood_candidates = generate_ood_candidates("hidden-ood-private", ood_config)
    id_bundles = match_ambiguity_bundles(
        id_candidates,
        bundle_count=int(plan["id_bundles"]),
        split="hidden-id-private",
    )
    ood_bundles = match_ambiguity_bundles(
        ood_candidates,
        bundle_count=int(plan["ood_bundles"]),
        split="hidden-ood-private",
    )
    all_bundles: tuple[tuple[str, AmbiguityBundle], ...] = tuple(
        [("id", bundle) for bundle in id_bundles] + [("ood", bundle) for bundle in ood_bundles]
    )
    maximum_distance = max(bundle.maximum_pairwise_distance for _, bundle in all_bundles)
    if maximum_distance > float(plan["maximum_bundle_distance"]):
        raise ValueError(f"hidden matching failed frozen distance gate: {maximum_distance:.6f}")

    worlds = tuple(world for _, bundle in all_bundles for world in bundle.worlds)
    market = MarketConfig(initial_cash_multiple=100.0)
    intervention = InterventionConfig()
    dynamics = DynamicsConfig(recovery_periods=20)
    oracles, oracle_hashes = _load_or_evaluate(
        worlds,
        output_directory=private_output,
        root_seed=root_seed,
        rollout_count=int(plan["oracle_rollouts"]),
        workers=workers,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )
    if len(oracles) != 100:
        raise AssertionError("hidden release must contain exactly 100 oracle worlds")

    rng = NamedRNG(root_seed).generator("model_replicates", "hidden-public-order")
    bundle_order = tuple(int(value) for value in rng.permutation(len(all_bundles)))
    public_bundles = []
    private_worlds = []
    public_world_count = 0
    for public_bundle_index, source_index in enumerate(bundle_order):
        stratum, bundle = all_bundles[source_index]
        bundle_id = f"janus-hidden-bundle-{public_bundle_index:03d}"
        member_order = tuple(int(value) for value in rng.permutation(4))
        public_worlds = []
        for member_index in member_order:
            world = bundle.worlds[member_index]
            opaque_id = f"janus-hidden-world-{public_world_count:03d}"
            public_world_count += 1
            oracle_name = f"{world.world_id}-k{int(plan['oracle_rollouts'])}.json"
            public_worlds.append(
                {
                    "world_id": opaque_id,
                    "observation": json.loads(world.serialized_observation),
                    "observation_sha256": hashlib.sha256(world.serialized_observation).hexdigest(),
                }
            )
            private_worlds.append(
                _private_world(
                    opaque_id,
                    bundle_id,
                    stratum,
                    world,
                    f"oracles/{oracle_name}",
                    oracle_hashes[oracle_name],
                )
            )
        public_bundles.append({"bundle_id": bundle_id, "worlds": public_worlds})

    private_manifest = {
        "schema_version": "janus-hidden-private-v1",
        "code_commit": code_commit,
        "seed_commitment_sha256": expected,
        "worlds": private_worlds,
        "maximum_bundle_distance": maximum_distance,
    }
    private_path = private_output / "private_manifest.json"
    _write_json(private_path, private_manifest, private=True)
    for path in (private_output / "oracles").glob("*.json"):
        path.chmod(0o600)

    public_manifest = {
        "schema_version": "janus-hidden-public-v1",
        "benchmark_version": "janus-v1",
        "code_commit": code_commit,
        "bundle_count": 25,
        "world_count": 100,
        "seed_commitment_sha256": expected,
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "decision_schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
        "bundles": public_bundles,
    }
    _write_json(public_manifest_path, public_manifest)
    release_commitment = {
        **commitment,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "public_manifest_sha256": hashlib.sha256(public_manifest_path.read_bytes()).hexdigest(),
        "private_manifest_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "world_count": 100,
        "bundle_count": 25,
        "maximum_bundle_distance": maximum_distance,
    }
    _write_json(commitment_path, release_commitment)
    return True
