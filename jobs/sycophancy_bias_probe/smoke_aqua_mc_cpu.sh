#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' "[warning] smoke_aqua_mc_cpu.sh is a compatibility alias; prefer smoke_aqua_mc_auto.sh because the default device request is --device auto." >&2
exec "$SCRIPT_DIR/smoke_aqua_mc_auto.sh" "$@"
