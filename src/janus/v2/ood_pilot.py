"""Use the completed, revealed v1 release as a declared v2 OOD design pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from janus.evaluation.phase0d_runner import _oracle_from_dict
from janus.oracle.counterfactual import OracleResult
from janus.scenarios.generator import (
    CandidateGeneratorConfig,
    CandidateWorld,
    generate_scenario_configs,
    simulate_candidate,
)
from janus.scenarios.hidden import generate_ood_scenario_configs
from janus.v2.design import audit_structural_ood_reversals


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def run_structural_ood_pilot(project_root: Path, output_path: Path) -> dict[str, Any]:
    """Audit cue/action reversals on v1, which is development data for v2."""

    private = _read(project_root / "artifacts/hidden-v1/private_manifest.json")
    root_seed = int(_read(project_root / "artifacts/hidden-v1/seed_manifest.json")["root_seed"])
    configs = {
        "id": CandidateGeneratorConfig(candidates_per_family=80, root_seed=root_seed),
        "ood": CandidateGeneratorConfig(candidates_per_family=160, root_seed=root_seed),
    }
    worlds: dict[str, CandidateWorld] = {}
    oracles: dict[str, OracleResult] = {}
    scenario_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
    for row in private["worlds"]:
        stratum = str(row["stratum"])
        family = str(row["family"])
        internal_id = str(row["internal_world_id"])
        cache_key = (stratum, family)
        if cache_key not in scenario_cache:
            from janus.scenarios.families import LatentFamily

            family_id = LatentFamily(family)
            config = configs[stratum]
            scenario_cache[cache_key] = (
                generate_scenario_configs("hidden-id-private", family_id, config)
                if stratum == "id"
                else generate_ood_scenario_configs("hidden-ood-private", family_id, config)
            )
        scenario = next(
            value for value in scenario_cache[cache_key] if value.scenario_id == internal_id
        )
        world = simulate_candidate(
            "hidden-id-private" if stratum == "id" else "hidden-ood-private",
            scenario,
            configs[stratum],
        )
        worlds[str(row["opaque_world_id"])] = CandidateWorld(
            world_id=str(row["opaque_world_id"]),
            split=world.split,
            family=world.family,
            scenario=world.scenario,
            decision_state=world.decision_state,
            observation=world.observation,
            serialized_observation=world.serialized_observation,
            summary_features=world.summary_features,
        )
        oracle_payload = _read(project_root / "artifacts/hidden-v1" / str(row["oracle_file"]))
        oracles[str(row["opaque_world_id"])] = _oracle_from_dict(oracle_payload["oracle"])

    id_worlds = tuple(
        worlds[str(row["opaque_world_id"])] for row in private["worlds"] if row["stratum"] == "id"
    )
    ood_worlds = tuple(
        worlds[str(row["opaque_world_id"])] for row in private["worlds"] if row["stratum"] == "ood"
    )
    id_ids = {world.world_id for world in id_worlds}
    ood_ids = {world.world_id for world in ood_worlds}
    report = audit_structural_ood_reversals(
        id_worlds,
        ood_worlds,
        {key: value for key, value in oracles.items() if key in id_ids},
        {key: value for key, value in oracles.items() if key in ood_ids},
        maximum_distance=1.25,
        minimum_pairs=4,
    )
    report.update(
        {
            "pilot_source": "completed and revealed hidden-v1 release",
            "pilot_only": True,
            "final_v2_requirement": (
                "apply the same preregistered reversal gate to fresh hidden-v2 candidates; "
                "abort generation if it fails"
            ),
            "paid_api_calls": 0,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
