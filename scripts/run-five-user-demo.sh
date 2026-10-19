#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

usage() {
  printf 'Usage: %s [--output-dir PATH] [--llm] [--llm-model MODEL]\n' "$0"
  printf 'Build three known-user and two cold-start profiles, then recommend.\n'
}

output_dir="artifacts/demo-5"
llm_enabled="false"
llm_model="gpt-4.1-mini"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --output-dir\n' >&2
        exit 2
      fi

      output_dir="$2"
      shift 2
      ;;
    --llm)
      llm_enabled="true"
      shift
      ;;
    --llm-model)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --llm-model\n' >&2
        exit 2
      fi

      llm_model="$2"
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

cd "$repo_root"

profile_command=".venv/bin/recsys-profile"
recommend_command=".venv/bin/recsys-recommend"
splits_path="artifacts/movies-tv-2023/splits.parquet"
catalog_path="artifacts/movies-tv-2023/catalog_enriched.parquet"
bundle_path="runs/item-knn-movies-tv-2023/test/20261015T173128Z/bundle"
manifest_path="artifacts/movies-tv-2023/search-index-manifest-v1.json"

for path in "$profile_command" "$recommend_command" "$splits_path" "$catalog_path" "$bundle_path" "$manifest_path"; do
  if [[ ! -e "$path" ]]; then
    printf 'Required path does not exist: %s\n' "$path" >&2
    exit 1
  fi
done

user_ids=(
  "AGUFDVO4TYIFQHQUMDXFHSHH43BA"
  "AHWQI5RDKU3AHHGKAKNLB6BMQSFQ"
  "AHQBMR4BHOVTHOV4QI6W63SDA7ZA"
  "COLD_START_DEMO_001"
  "COLD_START_DEMO_002"
)
as_of_values=(
  "1677290584675"
  "1565971148115"
  "1464208244000"
  "1700000000000"
  "1700000000000"
)
user_kinds=("known" "known" "known" "cold-start" "cold-start")

mkdir -p "$output_dir/profiles" "$output_dir/recommendations"

printf 'Output directory: %s\n' "$output_dir"
printf 'LLM explanations: %s\n' "$llm_enabled"

for index in "${!user_ids[@]}"; do
  ordinal=$((index + 1))
  user_id="${user_ids[$index]}"
  as_of="${as_of_values[$index]}"
  user_kind="${user_kinds[$index]}"

  profile_path="$output_dir/profiles/${ordinal}-${user_kind}.json"
  recommendation_path="$output_dir/recommendations/${ordinal}-${user_kind}.json"

  printf '[%s/5] Building %s profile: %s\n' "$ordinal" "$user_kind" "$user_id"
  "$profile_command" build \
    --splits "$splits_path" \
    --catalog "$catalog_path" \
    --user-id "$user_id" \
    --as-of "$as_of" \
    --phase test \
    --artifact-version movies-tv-2023 \
    --output "$profile_path"

  recommend_args=(
    --profile "$profile_path"
    --item-knn-bundle "$bundle_path"
    --search-manifest "$manifest_path"
    --no-semantic
    --output "$recommendation_path"
  )
  if [[ "$llm_enabled" == "true" ]]; then
    recommend_args+=(--llm-explanation --llm-model "$llm_model")
  fi

  printf '[%s/5] Running recommender: %s\n' "$ordinal" "$user_id"
  "$recommend_command" "${recommend_args[@]}"
done

printf 'Completed. Results: %s/recommendations\n' "$output_dir"
