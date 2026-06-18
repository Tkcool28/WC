#!/usr/bin/env python3
"""Build the production goal model artifact.

Usage:
    python3 scripts/build_goal_model_artifact.py
    python3 scripts/build_goal_model_artifact.py --output data/artifacts/goal_model.json
    python3 scripts/build_goal_model_artifact.py --shrinkage 5 --output data/artifacts/goal_model_sh5.json

Requirements:
    - Deterministic (no network calls, no random seed dependency)
    - Reads only tracked data files
    - Validates output artifact
    - Prints summary
    - Fails loudly on invalid inputs
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import sys
from pathlib import Path

from soccer_ev_model.goal_model import RegularizedTeamPoissonModel
from soccer_ev_model.goal_model_data import build_goal_matches, load_raw_matches
from soccer_ev_model.goal_model_production import build_artifact, save_artifact, load_artifact


def file_metadata(path: Path) -> dict:
    """Compute file hash and size for provenance."""
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build goal model artifact")
    parser.add_argument("--output", default="data/artifacts/goal_model.json",
                        help="Output artifact path")
    parser.add_argument("--shrinkage", type=float, default=5.0,
                        help="Shrinkage parameter (default: 5.0)")
    parser.add_argument("--data-path", default="data/processed/international_matches.json",
                        help="Path to processed matches JSON")
    parser.add_argument("--identity-path", default="data/team_identity.json",
                        help="Path to team identity JSON")
    args = parser.parse_args()

    print(f"Loading matches from {args.data_path}...")
    raw = load_raw_matches(args.data_path)
    matches, excluded = build_goal_matches(raw)
    excluded_count = sum(excluded.values())
    print(f"  {len(matches)} usable matches, {excluded_count} excluded")
    print(f"  Date range: {matches[0].match_date} to {matches[-1].match_date}")

    if len(matches) < 100:
        print("ERROR: fewer than 100 matches — refusing to build artifact", file=sys.stderr)
        return 1

    print(f"\nFitting RegularizedTeamPoissonModel (shrinkage={args.shrinkage})...")
    model = RegularizedTeamPoissonModel.fit(matches, shrinkage=args.shrinkage, iterations=50)
    print(f"  Converged: {model.converged} in {model.iterations_run} iterations")
    print(f"  Global rate: {model.global_rate:.4f}")
    print(f"  Home advantage: {model.home_advantage:.4f}")
    print(f"  Teams: {len(model.attacks)}")

    print("\nBuilding artifact...")
    source_files = {
        "matches": file_metadata(Path(args.data_path)),
        "identity": file_metadata(Path(args.identity_path)),
    }
    artifact = build_artifact(
        model=model,
        matches=matches,
        excluded_count=excluded_count,
        source_files=source_files,
    )

    output_path = Path(args.output)
    save_artifact(artifact, output_path)
    print(f"  Saved to {output_path}")

    # Validate: reload and check round-trip
    print("\nValidating artifact...")
    loaded = load_artifact(output_path)
    assert loaded.artifact_version == artifact.artifact_version
    assert loaded.shrinkage == artifact.shrinkage
    assert loaded.global_rate == artifact.global_rate
    assert loaded.home_advantage == artifact.home_advantage
    assert set(loaded.attacks.keys()) == set(artifact.attacks.keys())
    assert set(loaded.defenses.keys()) == set(artifact.defenses.keys())
    assert loaded.data_cutoff == artifact.data_cutoff
    assert loaded.training_row_count == artifact.training_row_count
    print("  Validation passed ✓")

    # Summary
    print("\n" + "=" * 60)
    print("ARTIFACT SUMMARY")
    print("=" * 60)
    print(f"  Version:         {artifact.artifact_version}")
    print(f"  Model:           {artifact.model_version}")
    print(f"  Data cutoff:     {artifact.data_cutoff}")
    print(f"  Training rows:   {artifact.training_row_count}")
    print(f"  Excluded rows:   {artifact.excluded_row_count}")
    print(f"  Shrinkage:       {artifact.shrinkage}")
    print(f"  Global rate:     {artifact.global_rate:.4f}")
    print(f"  Home advantage:  {artifact.home_advantage:.4f}")
    print(f"  Teams:           {len(artifact.attacks)}")
    print(f"  Converged:       {artifact.converged} ({artifact.iterations_run} iters)")
    print(f"  FIFA prior w:    {artifact.fifa_prior_weight}")
    print(f"  Squad prior w:   {artifact.squad_prior_weight}")
    print(f"  Source files:    {list(artifact.source_files.keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
