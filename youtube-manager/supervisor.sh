#!/usr/bin/env bash
set -uo pipefail
APP="$HOME/data/apps/youtube-manager"
cd "$APP" || exit 1
exec 9>"$APP/supervisor.lock"
flock -n 9 || exit 0
echo "$(date -Is) supervisor started" >> "$APP/supervisor.log"
while :; do
  python3 -u "$APP/telegram_bot.py" >> "$APP/jarvis-youtube.log" 2>&1
  status=$?
  echo "$(date -Is) bot exited status=$status; restarting in 10s" >> "$APP/supervisor.log"
  sleep 10
done
