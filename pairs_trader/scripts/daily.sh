#!/usr/bin/env bash
# Daily broker loop. Schedule after 19:05 New York time on weekdays, e.g. (crontab, machine in UTC-5/-4 aware TZ):
#   CRON_TZ=America/New_York
#   7 19 * * 1-5  /path/to/pairs_trader/scripts/daily.sh >> /path/to/pairs_trader/runs/logs/daily.log 2>&1
# Paper keys for --broker alpaca-paper come from the environment (APCA_API_KEY_ID / APCA_API_SECRET_KEY);
# put them in ~/.config/pairs_trader.env (chmod 600), never in this repository.
# HEARTBEAT_URL (optional): a dead-man's-switch URL (e.g. a healthchecks-style ping) that alerts if no run happens.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f "$HOME/.config/pairs_trader.env" ] && set -a && . "$HOME/.config/pairs_trader.env" && set +a
BROKER="${BROKER:-sim}"
python -m pairs_trader.cli trade --data data/sp500 --state "runs/trade_sp500_${BROKER}" --broker "$BROKER" --refresh
