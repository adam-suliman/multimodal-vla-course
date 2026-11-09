#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
runner="${SO101_RUNNER:-./scripts/run.sh}"
run_id="acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
"$runner" doctor
"$runner" fetch --offline
"$runner" check-kinematics
"$runner" check-scene --output "runs/$run_id-scene"
"$runner" evaluate --policy expert --output "runs/$run_id-expert" --video
"$runner" audit --run "runs/$run_id-expert"
"$runner" collect --seed 10000 --episodes 10 --output "runs/$run_id-demos" --video
"$runner" export --run "runs/$run_id-demos" --dataset "runs/$run_id-dataset"
"$runner" check-dataset --dataset "runs/$run_id-dataset"
"$runner" policy-smoke
"$runner" evaluate --policy smolvla --output "runs/$run_id-smolvla" --video
"$runner" audit --run "runs/$run_id-smolvla" --vram-budget-mib 12288
printf 'Acceptance artifacts: runs/%s-*\n' "$run_id"
