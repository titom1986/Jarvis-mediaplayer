#!/usr/bin/env python3
"""Private YouTube -> ytdl-sub -> Plex/Plexamp Telegram gateway. Python stdlib."""
import json, os, re, shutil, subprocess, threading, queue, time, urllib.request, urllib.parse
from pathlib import Path

BASE=Path(__file__).resolve().parent
for line in (BASE/'.env').read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
TOKEN=os.environ.get('TELEGRAM_BOT_TOKEN','')
ALLOWED={int(x) for x in os.environ.get('TELEGRAM_ALLOWED_USERS','').split(',') if x.strip().isdigit()}
if not TOKEN or not ALLOWED: raise SystemExit('Missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USERS in .env')
API='https://api.telegram.org/bot'+TOKEN+'/'
JOBS=queue.Queue(maxsize=20)
PENDING={}
BUSY={'id':None}
QUEUED=set()
QUEUE_LOCK=threading.Lock()
URL=re.compile(r'https?://[^\s<>]+')
VID=re.compile(r'^[A-Za-z0-9_-]{11}$')
VIDEO=Path(os.environ.get('YOUTUBE_VIDEO_DIR',str(Path.home()/'data/youtube')))
AUDIO=Path(os.environ.get('YOUTUBE_AUDIO_DIR',str(Path.home()/'data/youtube-audio')))

def request(method,params):
    req=urllib.request.Request(API+method,data=json.dumps(params).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=65 if method=='getUpdates' else 20) as res:
        data=json.load(res)
    if not data.get('ok'): raise RuntimeError('Telegram request failed')
    return data['result']

def send(chat,message,buttons=None):
    d={'chat_id':chat,'text':message[:3800]}
    if buttons:d['reply_markup']={'inline_keyboard':buttons}
    return request('sendMessage',d)

def get_id(link):
    u=urllib.parse.urlsplit(link)
    host=(u.hostname or '').lower()
    if host in ('youtu.be','www.youtu.be'):
        v=u.path.strip('/').split('/')[0]
    elif host in ('youtube.com','www.youtube.com','m.youtube.com','music.youtube.com'):
        if u.path=='/watch':v=urllib.parse.parse_qs(u.query).get('v',[''])[0]
        elif u.path.startswith(('/shorts/','/live/','/embed/')):v=u.path.split('/')[2]
        else:raise ValueError('Unsupported YouTube link')
    else:raise ValueError('Not YouTube')
    if not VID.fullmatch(v):raise ValueError('Invalid video ID')
    return v

def scan(section):
    host=os.environ.get('PLEX_URL','').rstrip('/')
    token=os.environ.get('PLEX_TOKEN','')
    sid=os.environ.get(section,'')
    if not (host and token and sid):return 'Plex scan non configuré'
    q=urllib.parse.urlencode({'X-Plex-Token':token})
    req=urllib.request.Request(f'{host}/library/sections/{sid}/refresh?{q}',method='GET')
    with urllib.request.urlopen(req,timeout=25):pass
    return 'Scan Plex demandé'

def locate(vid):
    matches=[]
    if not VIDEO.is_dir():raise RuntimeError('Dossier vidéo introuvable : '+str(VIDEO))
    for info in VIDEO.rglob('*.info.json'):
        try:
            d=json.loads(info.read_text())
            if str(d.get('id',''))==vid:
                mkvs=list(info.parent.glob('*.mkv'))
                if len(mkvs)==1:matches.append((mkvs[0],d))
        except (OSError,ValueError):continue
    if len(matches)!=1:raise RuntimeError(f'Fichier vidéo introuvable ou ambigu : {len(matches)} correspondances')
    return matches[0]

def existing_video(vid):
    try:return locate(vid)
    except RuntimeError:return None

def music_export(vid):
    mkv,info=locate(vid)
    p=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=codec_name','-of','default=noprint_wrappers=1:nokey=1',str(mkv)],capture_output=True,text=True,check=True)
    if p.stdout.strip()!='opus':raise RuntimeError('Audio non-Opus : extraction sans réencodage impossible')
    title=str(info.get('title') or mkv.stem).replace('/','-').replace('\\','-')
    artist=str(info.get('channel') or info.get('uploader') or 'YouTube').replace('/','-')
    if ' | ' in title: track,album=title.split(' | ',1)
    else:track=album=title
    clean=lambda s:re.sub(r'[<>:"/\\|?*\x00-\x1f]','-',s).strip(' .')[:120] or 'Unknown'
    folder=AUDIO/clean(artist)/clean(album)
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/(clean(track)+'.opus')
    # Install local album artwork before Plex can discover the audio file.
    # Do not overwrite existing artwork when several tracks share a folder.
    cover=folder/'cover.jpg'
    if not cover.exists():
        poster=next((p for p in (mkv.parent/'poster.jpg',mkv.parent/'cover.jpg',mkv.parent/'poster.png',mkv.parent/'cover.png') if p.is_file()),None)
        if poster:
            # Plex recognizes cover.jpg; convert PNG rather than copying PNG bytes under a JPEG extension.
            staged=folder/'.cover.tmp.jpg'
            try:
                if poster.suffix.lower()=='.png':
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(poster),'-frames:v','1',str(staged)],check=True,timeout=60)
                else:shutil.copyfile(poster,staged)
                staged.replace(cover)
            finally:staged.unlink(missing_ok=True)
        else:
            print('No local artwork for music:',vid,flush=True)
    if out.exists() and out.stat().st_size>0:return 'Audio déjà présent : '+str(out)
    if out.exists():out.unlink()
    tmp=out.with_suffix('.tmp.opus')
    try:
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(mkv),'-map','0:a:0','-c:a','copy','-metadata','title='+track,'-metadata','artist='+artist,'-metadata','album='+album,'-y',str(tmp)],check=True,timeout=1800)
        tmp.rename(out)
    finally:
        tmp.unlink(missing_ok=True)
    return 'Audio créé : '+str(out)

def worker():
    while True:
        chat,vid,mode=JOBS.get()
        BUSY['id']=vid
        try:
            found=existing_video(vid)
            if found:
                send(chat,'Vidéo déjà présente ; réutilisation du fichier local : '+vid)
                mkv,_=found
            else:
                send(chat,'Téléchargement en cours : '+vid)
                subprocess.run(['docker','exec','ytdl-sub','ytdl-sub','--config','/config/config.yaml','dl','--yt','--u','https://youtu.be/'+vid],check=True,timeout=14400,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                mkv,_=locate(vid)
            parts=[]
            if mode in ('v','m'):
                parts.append('Vidéo prête : '+mkv.name)
                if os.environ.get('PLEX_URL') and os.environ.get('PLEX_TOKEN') and os.environ.get('PLEX_VIDEO_SECTION'):
                    parts.append(scan('PLEX_VIDEO_SECTION'))
            if mode in ('m','a'):
                parts.append(music_export(vid))
                if os.environ.get('PLEX_URL') and os.environ.get('PLEX_TOKEN') and os.environ.get('PLEX_MUSIC_SECTION'):
                    parts.append(scan('PLEX_MUSIC_SECTION'))
            if mode=='a':
                parts.append('Mode audio : la vidéo source reste stockée sur le serveur (réutilisable).')
            msg='\n'.join(parts)
            send(chat,msg)
        except subprocess.CalledProcessError as e:
            details=(e.stderr or b'')
            if isinstance(details,bytes):details=details.decode('utf-8','replace')
            print('Job failed:',vid,'mode:',mode,'exit:',e.returncode,'stderr:',details[-2000:],flush=True)
            send(chat,'Échec (code '+str(e.returncode)+') : '+(details.strip()[-450:] or 'voir jarvis-youtube.log'))
        except Exception as e:
            print('Job error:',vid,'mode:',mode,type(e).__name__,str(e)[:1000],flush=True)
            send(chat,'Erreur : '+str(e)[:350])
        finally:
            with QUEUE_LOCK:QUEUED.discard((vid,mode))
            BUSY['id']=None
            JOBS.task_done()

def handle(update):
    msg=update.get('message')
    if msg:
        user=msg.get('from',{}).get('id')
        if user not in ALLOWED or msg.get('chat',{}).get('type')!='private':return
        chat=msg['chat']['id'];txt=msg.get('text','')
        if txt.startswith(('/start','/help')):
            send(chat,'Envoie un lien YouTube. Choisis Vidéo Plex, Vidéo + Plexamp ou Audio Plexamp. /status pour la file.');return
        if txt.startswith('/status'):
            send(chat,'En cours : '+str(BUSY['id'] or 'aucun')+' ; en attente : '+str(JOBS.qsize()));return
        for url in URL.findall(txt):
            try:vid=get_id(url.rstrip('.,;!?)'))
            except ValueError:continue
            PENDING[(user,vid)]=time.monotonic()
            send(chat,'Vidéo '+vid+' : choisir la destination.',[[{'text':'Vidéo Plex','callback_data':'v:'+vid},{'text':'Vidéo + Plexamp','callback_data':'m:'+vid}],[{'text':'Audio Plexamp','callback_data':'a:'+vid}]])
            return
        send(chat,'Envoie un lien YouTube valide.')
    cb=update.get('callback_query')
    if cb:
        user=cb.get('from',{}).get('id')
        if user not in ALLOWED or cb.get('message',{}).get('chat',{}).get('type')!='private':return
        data=cb.get('data','');match=re.fullmatch(r'([vma]):([A-Za-z0-9_-]{11})',data)
        if not match:return
        mode,vid=match.groups()
        if time.monotonic()-PENDING.pop((user,vid),-1e10)>3600:return
        request('answerCallbackQuery',{'callback_query_id':cb['id']})
        chat=cb['message']['chat']['id']
        with QUEUE_LOCK:
            if (vid,mode) in QUEUED:
                send(chat,'Demande déjà en cours ou en attente : '+vid+' (mode '+mode+')')
                return
            try:
                JOBS.put_nowait((chat,vid,mode))
                QUEUED.add((vid,mode))
            except queue.Full:
                send(chat,'File pleine, réessaie plus tard.')
                return
        send(chat,'Ajouté à la file : '+vid)

threading.Thread(target=worker,daemon=True).start()
offset=None
print('Jarvis YouTube running; allowlisted accounts:',len(ALLOWED),flush=True)
while True:
    try:
        updates=request('getUpdates',{'timeout':45,'offset':offset,'allowed_updates':['message','callback_query']})
        for update in updates:
            offset=update['update_id']+1
            try:handle(update)
            except Exception as e:print('Update handling error:',type(e).__name__,flush=True)
    except Exception as e:
        print('Polling error:',type(e).__name__,flush=True)
        time.sleep(5)
