#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

cd "$repo_root"

usage() {
  printf 'Usage: %s [--region AWS_REGION] [--phase validation|test] [--neighbors K]\n' "$0"
  printf 'Example: %s --region ap-southeast-2 --phase test --neighbors 50\n' "$0"
}

aws_region="ap-southeast-1"
training_phase="test"
neighbors="50"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --region)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --region\n' >&2
        exit 2
      fi

      aws_region="$2"
      shift 2
      ;;
    --phase)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --phase\n' >&2
        exit 2
      fi

      training_phase="$2"
      shift 2
      ;;
    --neighbors)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --neighbors\n' >&2
        exit 2
      fi

      neighbors="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$aws_region" =~ ^[a-z]{2}(-[a-z0-9]+)+-[0-9]+$ ]]; then
  printf 'Invalid AWS region: %s\n' "$aws_region" >&2
  exit 2
fi

if [[ "$training_phase" != "validation" && "$training_phase" != "test" ]]; then
  printf 'Invalid phase: %s\n' "$training_phase" >&2
  exit 2
fi

if [[ ! "$neighbors" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Neighbors must be a positive integer: %s\n' "$neighbors" >&2
  exit 2
fi

command -v aws >/dev/null
command -v sky >/dev/null

aws_account_id="$(aws sts get-caller-identity --query Account --output text)"
s3_bucket="${S3_BUCKET:-hust-it4613-item-knn-${aws_account_id}-${aws_region}}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! "$aws_account_id" =~ ^[0-9]{12}$ ]]; then
  printf 'Invalid AWS account ID: %s\n' "$aws_account_id" >&2
  exit 1
fi

printf 'AWS region: %s (SkyPilot selects the availability zone)\n' "$aws_region"
printf 'Training phase: %s\n' "$training_phase"
printf 'Neighbors: %s (includes self-similarity)\n' "$neighbors"

printf 'S3 bucket: s3://%s (SkyPilot creates it when absent)\n' "$s3_bucket"
printf 'Run ID: %s\n' "$run_id"

sky launch \
  --cluster "it4613-item-knn" \
  --env "S3_BUCKET=$s3_bucket" \
  --env "RUN_ID=$run_id" \
  --env "PHASE=$training_phase" \
  --env "NEIGHBORS=$neighbors" \
  --infra "aws/$aws_region" \
  --idle-minutes-to-autostop 10 \
  --down \
  "infra/skypilot/item-knn-aws.yaml"
