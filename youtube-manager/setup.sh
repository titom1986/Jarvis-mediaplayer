#!/usr/bin/env bash
set -Eeuo pipefail
APP="$HOME/data/apps/youtube-manager"
SOURCE="$HOME/data/apps/jarvis-youtube-source"
LOG="$APP/jarvis-youtube.log"
echo "== Jarvis YouTube setup =="
test -f "$APP/.env" || { echo "ERROR: missing existing .env"; exit 1; }
test -d "$SOURCE/.git" || { echo "ERROR: missing Git clone $SOURCE"; exit 1; }
git -C "$SOURCE" pull --ff-only
install -m 755 "$SOURCE/youtube-manager/telegram_bot.py" "$APP/telegram_bot.py"
python3 -m py_compile "$APP/telegram_bot.py"
command -v docker >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
command -v screen >/dev/null
docker inspect ytdl-sub --format '{{.State.Status}}' | grep -qx running
test -d "$HOME/data/youtube"
mkdir -p "$HOME/data/youtube-audio"
chmod 600 "$APP/.env"
if ! grep -q '^TELEGRAM_ALLOWED_USERS=' "$APP/.env"; then printf '\nTELEGRAM_ALLOWED_USERS=1092870317\n' >> "$APP/.env"; fi
echo "Dependencies: OK"
echo "Stopping existing Jarvis YouTube instances (if any)..."
screen -S jarvis-youtube -X quit 2>/dev/null || true
# Stop only an existing python process running this exact script; preserve other Python services.
for pid in $(pgrep -f 'python3 telegram_bot.py' || true); do
  case "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" in "$APP") kill "$pid" 2>/dev/null || true;; esac
done
sleep 2
if pgrep -f 'python3 telegram_bot.py' >/dev/null; then
 echo "WARNING: another telegram_bot.py process is running; stop it before starting this service."
 exit 1
fi
: > "$LOG"
screen -dmS jarvis-youtube bash -c 'cd "$1" && exec python3 -u telegram_bot.py >> "$2" 2>&1' _ "$APP" "$LOG"
sleep 3
if ! screen -list | grep -q '[.]jarvis-youtube'; then
 echo "ERROR: bot exited. Recent log:"; tail -n 15 "$LOG"; exit 1
fi
echo "Telegram bot: RUNNING (screen jarvis-youtube)"
if grep -q '^PLEX_URL=' "$APP/.env" && grep -q '^PLEX_TOKEN=' "$APP/.env" && grep -q '^PLEX_VIDEO_SECTION=' "$APP/.env" && grep -q '^PLEX_MUSIC_SECTION=' "$APP/.env"; then
 echo "Plex scans: configured (not yet validated)"
else
 echo "Plex scans: NOT CONFIGURED; media downloads work but automatic library refresh is disabled."
fi
echo "Log: $LOG"
echo "Send /status to your Telegram bot; then test a YouTube link."
