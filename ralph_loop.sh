#!/usr/bin/env bash
# =============================================================================
# Ralph Loop — Iterative Task Runner
# =============================================================================
# Runs a command repeatedly until it succeeds or hits the max iteration limit.
# Usage:  ./ralph_loop.sh "command to run" [max_iterations]
# Example: ./ralph_loop.sh "cd frontend && npm run test:run" 5
# =============================================================================

set -euo pipefail

COMMAND="${1:-}"
MAX_ITER="${2:-10}"

if [ -z "$COMMAND" ]; then
    echo ""
    echo "  Ralph Loop - Iterative Task Runner"
    echo "  ===================================="
    echo ""
    echo "  Usage:  ./ralph_loop.sh \"command\" [max_iterations]"
    echo ""
    echo "  Examples:"
    echo "    ./ralph_loop.sh \"cd backend && python -m pytest tests/ -x\" 10"
    echo "    ./ralph_loop.sh \"cd frontend && npm run test:run\" 5"
    echo "    ./ralph_loop.sh \"cd frontend && npm run build\" 3"
    echo ""
    exit 1
fi

echo ""
echo "  ============================================="
echo "   Ralph Loop - Starting"
echo "  ============================================="
echo "   Command:    $COMMAND"
echo "   Max Iters:  $MAX_ITER"
echo "   Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ============================================="
echo ""

ITERATION=0

while [ "$ITERATION" -lt "$MAX_ITER" ]; do
    ITERATION=$((ITERATION + 1))

    echo ""
    echo "  [Iteration $ITERATION/$MAX_ITER] Running at $(date '+%H:%M:%S')..."
    echo "  -----------------------------------------"

    set +e
    bash -c "$COMMAND"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo ""
        echo "  ============================================="
        echo "   SUCCESS on iteration $ITERATION!"
        echo "   Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  ============================================="
        exit 0
    fi

    echo ""
    echo "  [Iteration $ITERATION] FAILED (exit code: $EXIT_CODE)"

    if [ "$ITERATION" -ge "$MAX_ITER" ]; then
        echo ""
        echo "  ============================================="
        echo "   STOPPED - Hit max iterations ($MAX_ITER)"
        echo "   The task did not succeed after $MAX_ITER attempts."
        echo "   Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  ============================================="
        exit 1
    fi

    echo "  Retrying in 2 seconds..."
    sleep 2
done
