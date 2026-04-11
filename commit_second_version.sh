#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./commit_second_version.sh [options]

Commits the "second version" work in this repo and can optionally push to GitHub.

Options:
  --message MSG         Commit message to use.
  --push                Push the commit to origin after committing.
  --with-plots          Include generated plot folders in the commit.
  --with-results        Include generated results folders in the commit.
  --with-checkpoints    Include checkpoints folders in the commit.
  --all-generated       Include plots, results, and checkpoints.
  --dry-run             Show what would be staged/committed without changing git state.
  --help                Show this help message.

Examples:
  ./commit_second_version.sh
  ./commit_second_version.sh --with-plots --message "Add second-version figures and one-shot pipelines"
  ./commit_second_version.sh --all-generated --push
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

commit_message="Add second-version one-shot pipelines, comparisons, and misc 2D figures"
push_after_commit=false
include_plots=false
include_results=false
include_checkpoints=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --message)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --message" >&2; exit 1; }
      commit_message="$1"
      ;;
    --push)
      push_after_commit=true
      ;;
    --with-plots)
      include_plots=true
      ;;
    --with-results)
      include_results=true
      ;;
    --with-checkpoints)
      include_checkpoints=true
      ;;
    --all-generated)
      include_plots=true
      include_results=true
      include_checkpoints=true
      ;;
    --dry-run)
      dry_run=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

core_paths=(
  "PROMPTS_ONESHOT.md"
  "PROMPTS_ONESHOT_1D.md"
  "configs/oneshot_sr.yaml"
  "configs/oneshot_sr_1d.yaml"
  "results_comparison_summary.md"
  "scripts/plot_comparison.py"
  "scripts/plot_comparison_1d.py"
  "scripts/plot_misc_2d.py"
  "scripts/plot_misc_burgers.py"
  "scripts/plot_results_oneshot.py"
  "scripts/run_inference_oneshot.py"
  "scripts/run_inference_oneshot_1d.py"
  "scripts/train_oneshot.py"
  "scripts/train_oneshot_1d.py"
  "src/inference/pipeline_oneshot.py"
  "src/inference/pipeline_oneshot_1d.py"
  "src/models/unet_oneshot.py"
  "src/models/unet_oneshot_1d.py"
  "src/training/train_oneshot.py"
  "src/training/train_oneshot_1d.py"
)

plot_paths=(
  "plots_oneshot"
  "plots_oneshot_1d"
  "plots_2d_misc/fig3_resolution_pyramid.png"
  "plots_2d_misc/fig3_resolution_pyramid.pdf"
  "plots_2d_misc/fig5_sequential_inverse_problem.png"
  "plots_2d_misc/fig5_sequential_inverse_problem.pdf"
)

result_paths=(
  "results_oneshot"
  "results_oneshot_1d"
)

checkpoint_paths=(
  "checkpoints_oneshot"
  "checkpoints_oneshot_1d"
)

paths_to_stage=("${core_paths[@]}")

if [[ "$include_plots" == true ]]; then
  paths_to_stage+=("${plot_paths[@]}")
fi

if [[ "$include_results" == true ]]; then
  paths_to_stage+=("${result_paths[@]}")
fi

if [[ "$include_checkpoints" == true ]]; then
  paths_to_stage+=("${checkpoint_paths[@]}")
fi

existing_paths=()
for path in "${paths_to_stage[@]}"; do
  if [[ -e "$path" ]]; then
    existing_paths+=("$path")
  fi
done

if [[ ${#existing_paths[@]} -eq 0 ]]; then
  echo "Nothing to stage from the configured second-version paths." >&2
  exit 1
fi

echo "Repo: $repo_root"
echo "Branch: $(git branch --show-current)"
echo "Remote: $(git remote get-url origin)"
echo
echo "Paths to stage:"
printf '  %s\n' "${existing_paths[@]}"
echo
echo "Commit message:"
echo "  $commit_message"
echo

if [[ "$dry_run" == true ]]; then
  echo "Dry run enabled: no git add/commit/push was executed."
  exit 0
fi

git add -- "${existing_paths[@]}"

if git diff --cached --quiet; then
  echo "No staged changes detected after git add." >&2
  exit 1
fi

git commit -m "$commit_message"

if [[ "$push_after_commit" == true ]]; then
  current_branch="$(git branch --show-current)"
  git push origin "$current_branch"
fi

echo
echo "Done. Current status:"
git status --short
