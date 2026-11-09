#!/usr/bin/env bash
set -euo pipefail
case "${1:-doctor}" in
    export|check-dataset|video|policy-smoke|test)
        exec /opt/policy/bin/python -m so101_lab "$@" ;;
    *)
        exec /opt/sim/bin/python -m so101_lab "$@" ;;
esac
