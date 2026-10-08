#!/usr/bin/env bash
set -euo pipefail
APP="$HOME/data/apps/youtube-manager"
SRC="$HOME/data/apps/jarvis-youtube-source"
test -f "$APP/.env"
git -C "$SRC" pull --ff-only
command -v flock
command -v crontab
python3 -m py_compile "$SRC/youtube-manager/telegram_bot.py"
install -m 755 "$SRC/youtube-manager/telegram_bot.py" "$APP/telegram_bot.py"
install -m 755 "$SRC/youtube-manager/supervisor.sh" "$APP/supervisor.sh"
chmod 600 "$APP/.env"
HOOK="@reboot /bin/bash $APP/supervisor.sh"
CURRENT="$(crontab -l 2>/dev/null || true)"
if ! printf '%s\n' "$CURRENT" | grep -Fxq "$HOOK"; then { printf '%s\n' "$CURRENT"; printf '%s\n' "$HOOK"; } | crontab -; fi
screen -S jarvis-youtube -X quit 2>/dev/null || true
sleep 2
nohup bash "$APP/supervisor.sh" >/dev/null 2>&1 </dev/null &
echo 'Supervisor launched; check supervisor.log and jarvis-youtube.log'
