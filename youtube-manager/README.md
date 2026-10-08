# Jarvis YouTube (Telegram)

Private long-polling Telegram bot. Python 3 standard library, host Docker, ffmpeg, ffprobe and existing ytdl-sub container.

Run from /home/titom/data/apps/youtube-manager. Create .env with TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS=1092870317. Never commit .env. Optionally set PLEX_URL, PLEX_TOKEN, PLEX_VIDEO_SECTION, PLEX_MUSIC_SECTION to trigger library scans. Set YOUTUBE_VIDEO_DIR and YOUTUBE_AUDIO_DIR only if your paths differ from ~/data/youtube and ~/data/youtube-audio.

Run: python3 telegram_bot.py

Only private chats from allowlisted accounts are processed. Send YouTube URL and choose Video Plex or Video + Plexamp. One worker, 20 queued jobs. The audio export copies Opus without transcoding. Plex scans only when correctly configured. The source video is not deleted. Only one polling process per token. First test should be an already-downloaded YouTube video.

IMPORTANT: Existing ytdl-sub config and Plex credentials are not changed by this repository. The bot must be launched on the Seedhost host, not inside ytdl-sub.
