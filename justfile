default:
    @just --list --unsorted

# Run every check. THE entrypoint after implementing a feature.
# Non-fail-fast: runs all four, then prints one summary. `just` dependencies
# would stop at the first failure and hide the rest.
verify:
    #!/usr/bin/env bash
    set -uo pipefail
    failed=()
    for check in lint format types test; do
        echo "=== just $check ==="
        just "$check" || failed+=("$check")
    done
    if [ ${#failed[@]} -eq 0 ]; then
        echo "=== all checks passed ==="
    else
        echo "=== FAILED: ${failed[*]} ==="
        exit 1
    fi

# Lint. The ruff-check hook runs with --fix; this one only reports.
lint:
    uv run ruff check

# Formatting. The ruff-format hook rewrites; this one only reports.
format:
    uv run ruff format --check

# Static types: the pyright hook, called by id so its command lives in one place.
types:
    uv run prek run pyright --all-files

# The fast suite (synthetic data, CPU, seconds): the pytest-fast hook, by id.
# `slow` is CPU smoke training and end-to-end runs.
test:
    uv run prek run pytest-fast --all-files

# Every hook in .pre-commit-config.yaml. ruff-check --fix and ruff-format
# REWRITE files here, unlike `just verify`.
hooks:
    uv run prek run --all-files
