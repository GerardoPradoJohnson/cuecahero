import {FlyStage} from './stage.js';
'use strict';
const $ = id => document.getElementById(id);
const colors = ['#c292ff','#75d0b1','#f0bd70','#e88e9c'];
let live=null, shown=null, geometry=null, frameImage=null, frameSerial=0, history=[], lastHistory=-1, lastEvent='', online=false;
let recording=null, playback=false, replayIndex=0, replayStart=0, replayOffset=0, sound=false, audio=null, lastPulse=-1;
let dopamineRewardPulse=0.0;
const gameCtx=$('game').getContext('2d'), brainCtx=$('brain').getContext('2d');
let stage=null;try{stage=new FlyStage($('fly-stage'));}catch(e){$('fly-stage').closest('.game-wrap').classList.add('stage-failed');console.warn('3D unavailable; displaying original game.',e);}
for(const view of ['room','tv'])$('view-'+view).onclick=()=>{stage?.setView(view);for(const v of ['room','tv']){$('view-'+v).classList.toggle('selected',v===view);$('view-'+v).setAttribute('aria-pressed',String(v===view));}};
const format=n=>Math.round(n).toLocaleString('es-CL');
const clock=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
function error(message){$('error').textContent=message;$('error').hidden=!message;}
async function control(command){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command)});if(!r.ok)throw Error((await r.json()).error);error('');}catch(e){error(e.message);}}
function setFrame(encoded){const serial=++frameSerial;const image=new Image();image.onload=()=>{if(serial===frameSerial){frameImage=image;if(shown)stage?.update(shown,image);}};image.src=`data:image/png;base64,${encoded}`;}
function display(s){
  if(s.mode!=='running'||s.driver!=='manual'){if(keyboardLanes.size||pointerLanes.size)releaseKeys();}
  shown=s;setFrame(s.frame);$('stage-feed').textContent=recording?'PARTIDA EVALUADA / REPLAY':'LIVE / FLY ROOM';
  $('score').textContent=String(s.game.score).padStart(4,'0');$('combo').innerHTML=`${s.game.combo}<span>×</span>`;
  $('accuracy').textContent=s.game.accuracy===null?'—':`${Math.round(s.game.accuracy)}%`;
  $('time').textContent=`${clock(s.game.time)} / ${clock(s.game.duration)}`;$('progress').style.width=`${100*s.game.time/s.game.duration}%`;
  const labels={paused:'EN PAUSA',running:'EN CURSO',loading:'CARGANDO CEREBRO',finished:'SESIÓN COMPLETA',error:'REVISAR ERROR'};
  $('game-status').textContent=recording?'REPRODUCCIÓN':labels[s.mode]||s.mode;
  $('start').textContent=s.mode==='running'?'Ⅱ Pausar':s.mode==='loading'?'Cargando…':s.mode==='finished'?'↻ Nueva sesión':s.game.time>0?'▶ Continuar':'▶ Iniciar sesión';
  $('start').disabled=!online||s.mode==='loading'||!!recording;
  for(const [id,driver] of [['manual','manual'],['neural','neural']]){const selected=s.driver===driver;$(id).classList.toggle('selected',selected);$(id).setAttribute('aria-pressed',String(selected));$(id).disabled=s.mode==='loading'||!!recording;}
  $('speed').value=String(s.speed);$('speed').disabled=!!recording;$('reset').disabled=!!recording||s.mode==='loading';
  $('backend').textContent=s.driver==='manual'?'SIN ESTIMULACIÓN':s.neural.backend==='native-cpu'?'CPU · C++':s.neural.backend==='cpu'?'CPU · NUMBA':'CARGANDO';
  if(recording?.header?.kind==='frozen_live_neural_evaluation')$('backend').textContent='EVALUACIÓN GRABADA';
  $('active').textContent=format(s.telemetry.active_neurons);$('neural-time').innerHTML=`${format(s.neural.time_ms)} <small>ms</small>`;
  $('compute').innerHTML=`${s.telemetry.step_wall_ms.toFixed(1)} <small>ms</small>`;$('spikes').textContent=format(s.telemetry.total_spikes);
  $('brain-note').textContent=s.driver==='manual'?'En modo manual, el cerebro está en reposo. Activa MaleCNS para cerrar el ciclo sensorial.':'Píxeles → encoder visual → conectoma → 4 salidas. Baseline fijo, sin aprendizaje ni acceso a notas futuras.';
  $('instruction').textContent=s.driver==='manual'?'Pulsa D · F · J · K al cruzar la línea.':'MaleCNS controla los cuatro carriles.';
  if(s.mode==='running'&&songTime(s)<0)$('instruction').textContent=`Prepárate · la música empieza en ${Math.ceil(-songTime(s))}…`;
  $('brain-mode-notice').hidden=s.driver!=='manual';
  $('activate-brain').disabled=!!recording;
  document.querySelectorAll('[data-lane]').forEach(b=>{
    const i=Number(b.dataset.lane),lane=s.game.lanes?.[i],event=lane?.last_event;
    b.disabled=s.driver!=='manual'||s.mode!=='running'||!!recording;
    b.classList.toggle('observing',s.driver==='neural'||!!recording);
    const fresh=event&&s.game.time-event.time>=0&&s.game.time-event.time<.4;
    b.classList.toggle('pressed',!!s.game.action[i]||(fresh&&event.kind!=='miss'));
    b.classList.toggle('hit',!!fresh&&['good','perfect'].includes(event.kind));
    b.classList.toggle('miss',!!fresh&&['miss','empty'].includes(event.kind));
    b.querySelector('.lane-result').textContent=event?({perfect:'PERFECTO',good:'BIEN',miss:'SE ESCAPÓ',empty:'FUERA DE TIEMPO'}[event.kind]):'LISTO';
    b.querySelector('.lane-count').textContent=lane?`✓ ${lane.hits} · × ${lane.misses} · Ø ${lane.wrong}`:'REPLAY ANTERIOR';
    b.title='✓ Aciertos · × Notas perdidas · Ø Pulsaciones vacías';
  });
  for(let i=0;i<4;i++)$('rate-'+i).style.width=`${Math.min(100,s.neural.readout_rates[i]*(s.neural.readout_kind==='calibrated'?100:.25))}%`;
  document.querySelectorAll('.readout-grid small').forEach((label,i)=>{label.textContent=(s.neural.readout_labels||['DNa02 · L','DNpe017 · L','DNpe017 · R','DNa02 · R'])[i];});
  if(s.driver==='neural'&&s.neural.readout_kind==='calibrated')$('brain-note').textContent=recording?'Partida del checkpoint congelado, evaluada con el cerebro completo y reproducida a velocidad normal.':'Lector externo de spikes; simulacion neuronal en vivo.';
  const l=s.learning;
  if(l&&l.enabled){
    $('plasticity-badge').innerHTML='<span class="dot" style="background:#75d0b1"></span> Plasticidad activa';
    $('plasticity-badge').className='badge';
    $('plasticity-box').hidden=false;
    $('plasticity-status-text').textContent=`${format(l.plastic_edges)} sinapsis plásticas (${format(l.changed_edges)} modificadas · ΔW medio: ${((l.mean_fraction||1)*100-100).toFixed(2)}%)`;
    if(s.driver==='neural')$('brain-note').textContent='MaleCNS adaptativo: coincidencia pre/post modula 4.276 sinapsis con RewardSignal.';
  }else{
    $('plasticity-badge').innerHTML='<span class="dot"></span> Pesos fijos';
    $('plasticity-badge').className='frozen';
    $('plasticity-box').hidden=true;
  }
  const calibrated=s.neural.readout_kind==='calibrated';
  $('sensor-method').textContent=calibrated?'El encoder ajusta contraste y proyecta la imagen sobre una región de receptores oficiales, con agrupación espacial. Es una adaptación de ingeniería, no una medición de la visión de la mosca.':'Un encoder visual convierte esos píxeles en corrientes R1–R6 y un sesgo tónico de lámina, siguiendo la aproximación declarada de DOOMFLY.';
  $('decoder-method').textContent=(recording?.header?.kind==='frozen_live_neural_evaluation'||s.live_training?.method?.startsWith('Supervised'))?'Lector externo entrenado durante 200 epocas supervisadas con spikes registrados. En evaluacion solo recibe actividad neuronal; no recibe etiquetas, calendario de notas ni recompensas.':calibrated?'Un lector externo calibrado convierte actividad reciente de poblaciones de lámina en cuatro pulsaciones. Durante el juego solo recibe spikes; los pesos del conectoma permanecen fijos.':'Cuatro grupos de lectura generan pulsaciones mediante un mapeo fijo, sin acceso a notas futuras ni recompensas.';
  $('replay').disabled=!s.session||!!recording;$('export').disabled=!s.session;
  if(s.revision!==lastHistory){history.push(s.telemetry.spikes);if(history.length>120)history.shift();lastHistory=s.revision;}
  const event=s.game.events.filter(e=>e.kind!=='hold_complete').at(-1), key=event?`${s.game.time}:${event.kind}:${event.lane}`:'';
  if(event&&key!==lastEvent){
    lastEvent=key;
    const labels={perfect:'PERFECTO',good:'BIEN',miss:'SE ESCAPÓ',empty:'FUERA DE TIEMPO'};
    $('judgement').textContent=labels[event.kind];
    $('judgement').style.color=event.kind==='miss'||event.kind==='empty'?'#aaa1b1':colors[event.lane];
    $('judgement').style.opacity='1';
    setTimeout(()=>{$('judgement').style.opacity='0';},360);

  }
  error(s.error||'');
  if(s.game.time===0){lastPulse=-1;history=[];}
  const lt = s.live_training;
  if (lt && $('live-gen-badge')) {
    const gen = lt.generation || 0;
    const hist = lt.history || [];
    const supervisedReadout = String(lt.method || '').startsWith('Supervised');
    const trainingTitle = $('live-training-box')?.querySelector('.live-training-title .tiny-label');
    if (trainingTitle) trainingTitle.textContent = supervisedReadout
      ? 'LECTOR EXTERNO · 200 ÉPOCAS SUPERVISADAS'
      : 'EVOLUCIÓN GENERACIONAL · RL EN VIVO';
    $('live-gen-badge').innerHTML = `Generación ${gen} <small id="live-gen-status">${gen === 0 ? 'Partiendo de cero' : `Episodio #${gen + 1} (en vivo)`}</small>`;
    if(!lt.enabled)$('live-gen-badge').textContent=`Checkpoint ${gen} · pesos congelados`;
    let curAcc = s.game.accuracy;
    if (curAcc === null && hist.length > 0) curAcc = hist[hist.length - 1].accuracy;
    const accVal = curAcc !== null ? curAcc : 0.0;
    const failVal = Math.max(0, 100.0 - accVal);
    $('live-acc-badge').textContent = `${Math.round(accVal)}% Precisión`;
    $('live-acc-badge').style.color = accVal >= 50 ? '#75d0b1' : '#f0bd70';
    if ($('live-fail-badge')) {
      $('live-fail-badge').textContent = `${Math.round(failVal)}% Fallos`;
    }
    drawLiveTrainingChart(hist, gen, s.game);
  }
  if (s.game?.song) {
    if ($('track-title')) $('track-title').innerHTML = `${s.game.song.title}<small id="track-subtitle">${s.game.song.subtitle}</small>`;
    if ($('track-bpm')) $('track-bpm').textContent = `${s.game.bpm} BPM`;
    if ($('song-select') && $('song-select').value !== s.game.song.id) {
      $('song-select').value = s.game.song.id;
    }
  }
  syncSongAudio(s);
}
let songAudio = null, currentSongUrl = '';
let checkpointRows=[];
function createSongAudio(url){
  const element=$('song-audio');
  element.pause();element.src=url;element.preload='auto';
  return element;
}
function songTime(s){return (s?.game?.time||0)-(s?.game?.song?.audio_offset||0);}
function syncSongAudio(s) {
  if (!sound) {
    if (songAudio && !songAudio.paused) songAudio.pause();
    return;
  }
  const songUrl = s.game?.song?.audio_url || '/audio/primer_panuelo.wav';
  if (!songAudio || currentSongUrl !== songUrl) {
    if (songAudio) songAudio.pause();
    songAudio = createSongAudio(songUrl);
    songAudio.preload = 'auto';
    currentSongUrl = songUrl;
  }
  const targetTime=songTime(s);
  if(targetTime<0){songAudio.pause();if(songAudio.currentTime!==0)songAudio.currentTime=0;return;}
  songAudio.playbackRate = s.speed || 1.0;
  if (s.mode === 'running' && (!recording || playback)) {
    if ((!recording||songAudio.paused) && Math.abs(songAudio.currentTime - targetTime) > 0.15) {
      songAudio.currentTime = Math.max(0, targetTime);
    }
    if (songAudio.paused) {
      songAudio.play().catch(e => console.warn('Audio play prevented:', e));
    }
  } else {
    if (!songAudio.paused) songAudio.pause();
    if (s.game.time === 0) songAudio.currentTime = 0;
  }
}
async function loadSongsCatalog() {
  try {
    const r = await fetch('/api/songs');
    if (!r.ok) return;
    const songs = await r.json();
    const select = $('song-select');
    if (!select) return;
    select.innerHTML = '';
    songs.forEach(song => {
      const opt = document.createElement('option');
      opt.value = song.id;
      opt.textContent = `${song.title} (${song.bpm} BPM · ${song.notes_count} notas)`;
      select.appendChild(opt);
    });
    if (shown?.game?.song?.id) {
      select.value = shown.game.song.id;
    }
  } catch (e) {
    console.warn('Could not load songs catalog:', e);
  }
}
$('song-select')?.addEventListener('change', e => {
  control({type: 'select_song', song_id: e.target.value});
});
async function poll(){try{const r=await fetch('/api/state');if(!r.ok)throw Error('Servidor no disponible');const s=await r.json();const reconnected=!online;online=true;live=s;$('connection').textContent='Laboratorio local conectado';$('connection-dot').style.background='#75d0b1';if(!recording&&(!shown||reconnected||s.revision!==shown.revision))display(s);}catch(e){online=false;$('connection').textContent='Sin conexión al servidor';$('connection-dot').style.background='#e88e9c';$('start').disabled=true;}setTimeout(poll,65);}
$('start').onclick=()=>{
  if(live?.driver==='neural'&&live?.mode!=='running'&&$('checkpoint-select')?.value&&!recording){watchCheckpoint();return;}
  if (sound && live?.mode !== 'running') {
    const songUrl = live?.game?.song?.audio_url || currentSongUrl || '/audio/la_consentida.wav';
    if (!songAudio || currentSongUrl !== songUrl) {
      if (songAudio) songAudio.pause();
      songAudio = createSongAudio(songUrl);
      songAudio.preload = 'auto';
      currentSongUrl = songUrl;
    }
    songAudio.currentTime = Math.max(0, songTime(live));
    if(songTime(live)<0){
      songAudio.muted=true;
      songAudio.play().then(()=>{songAudio.pause();songAudio.currentTime=0;songAudio.muted=false;}).catch(()=>{songAudio.muted=false;});
    }else{songAudio.play().catch(e=>console.warn('Audio play on start gesture:',e));}
  }
  control({type:live?.mode==='running'?'pause':'start'});
};
$('reset').onclick=()=>{history=[];if(songAudio){songAudio.pause();songAudio.currentTime=0;}control({type:'reset'});};
$('reset-training-btn')?.addEventListener('click',()=>control({type:'reset_training'}));
$('save-checkpoint-btn')?.addEventListener('click', async () => {
  try {
    await control({ type: 'save_checkpoint' });
    const btn = $('save-checkpoint-btn');
    if (btn) {
      const orig = btn.innerHTML;
      btn.innerHTML = '✓ ¡Guardado!';
      setTimeout(() => { btn.innerHTML = orig; }, 2200);
    }
    loadTrainingRuns();
  } catch (e) {
    console.error('Error al guardar checkpoint:', e);
  }
});
$('erase-memory-btn')?.addEventListener('click',()=>control({type:'erase_memory'}));
$('activate-brain').onclick=()=>control({type:'driver',value:'neural'});
$('manual').onclick=()=>control({type:'driver',value:'manual'});$('neural').onclick=()=>control({type:'driver',value:'neural'});
$('speed').onchange=e=>control({type:'speed',value:Number(e.target.value)});
const keyboardLanes=new Set(),pointerLanes=new Map();
let inputQueue=Promise.resolve();
function sendHeldKeys(){
  const lanes=[...new Set([...keyboardLanes,...pointerLanes.values()])];
  inputQueue=inputQueue.then(()=>control({type:'held_keys',lanes}));
}
function releaseKeys(){keyboardLanes.clear();pointerLanes.clear();sendHeldKeys();}
function canPlay(){return live?.mode==='running'&&live.driver==='manual'&&!recording;}
document.querySelectorAll('[data-lane]').forEach(button=>{
  button.addEventListener('pointerdown',e=>{
    if(!canPlay())return;
    e.preventDefault();button.setPointerCapture(e.pointerId);
    pointerLanes.set(e.pointerId,Number(button.dataset.lane));sendHeldKeys();
  });
  for(const type of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(type,e=>{
    if(pointerLanes.delete(e.pointerId))sendHeldKeys();
  });
  button.addEventListener('click',e=>{if(e.detail===0&&canPlay())control({type:'keys',lanes:[Number(button.dataset.lane)]});});
});
document.addEventListener('keydown',e=>{
  if($('method-dialog').open||['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
  const lane=['KeyD','KeyF','KeyJ','KeyK'].indexOf(e.code);
  if(lane>=0&&!e.repeat&&canPlay()){e.preventDefault();keyboardLanes.add(lane);sendHeldKeys();}
  if(e.code==='Space'&&!e.repeat&&!recording&&document.activeElement.tagName!=='BUTTON'){e.preventDefault();$('start').click();}
});
document.addEventListener('keyup',e=>{
  const lane=['KeyD','KeyF','KeyJ','KeyK'].indexOf(e.code);
  if(keyboardLanes.delete(lane)){e.preventDefault();sendHeldKeys();}
});
window.addEventListener('blur',releaseKeys);
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeys();});
$('methods').onclick=()=>$('method-dialog').showModal();$('method-close').onclick=()=>$('method-dialog').close();
$('sound').onclick=async()=>{
  sound=!sound;
  $('sound').textContent=sound?'♪ Sonido activo':'♪ Activar sonido';
  $('sound').setAttribute('aria-pressed',String(sound));
  if (sound) {
    const songUrl = shown?.game?.song?.audio_url || currentSongUrl || '/audio/la_consentida.wav';
    if (!songAudio || currentSongUrl !== songUrl) {
      if (songAudio) songAudio.pause();
      songAudio = createSongAudio(songUrl);
      songAudio.preload = 'auto';
      currentSongUrl = songUrl;
    }
    if (shown?.mode === 'running'&&songTime(shown)>=0) {
      songAudio.currentTime = Math.max(0,songTime(shown));
      songAudio.play().catch(e => console.warn('Audio play:', e));
    } else {
      songAudio.muted=true;
      songAudio.play().then(()=>{
        songAudio.pause();songAudio.currentTime=Math.max(0,songTime(shown));songAudio.muted=false;
      }).catch(()=>{songAudio.muted=false;});
    }
  } else {
    if (songAudio && !songAudio.paused) songAudio.pause();
  }
  if (shown) syncSongAudio(shown);
};
async function getReplay(){const id=live?.session;if(!id)throw Error('Todavía no hay una sesión grabada.');const response=await fetch('/api/replay?id='+encodeURIComponent(id));if(!response.ok)throw Error((await response.json()).error);return response.json();}
$('replay').onclick=async()=>{try{await control({type:'pause'});const data=await getReplay();if(!data.frames.length)throw Error('La sesión todavía no tiene pasos registrados.');recording=data;replayIndex=0;replayOffset=0;replayStart=performance.now();playback=true;history=[];$('replay-bar').hidden=false;$('scrub').max=data.frames.length-1;$('replay-play').textContent='Pausar';display(data.frames[0]);}catch(e){error(e.message);}};
$('replay-close').onclick=()=>{recording=null;playback=false;$('replay-bar').hidden=true;history=[];if(live)display(live);};
$('replay-play').onclick=()=>{if(replayIndex===recording.frames.length-1){replayIndex=0;if(songAudio)songAudio.currentTime=0;display(recording.frames[0]);$('scrub').value=0;}playback=!playback;replayOffset=recording.frames[replayIndex].game.time-recording.frames[0].game.time;replayStart=performance.now();$('replay-play').textContent=playback?'Pausar':'Continuar';if(shown)syncSongAudio(shown);};
$('scrub').oninput=e=>{replayIndex=Number(e.target.value);if(songAudio)songAudio.currentTime=Math.max(0,songTime(recording.frames[replayIndex]));replayOffset=recording.frames[replayIndex].game.time-recording.frames[0].game.time;replayStart=performance.now();history=[];display(recording.frames[replayIndex]);};
$('export').onclick=async()=>{try{const data=recording||await getReplay();const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`cueca-hero-${data.header.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){error(e.message);}};
let yaw=.3,pitch=-.28,drag=null,lastBrainFrame=null;
$('brain').addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];$('brain').setPointerCapture(e.pointerId);});
$('brain').addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];});
$('brain').addEventListener('pointerup',()=>drag=null);$('brain').addEventListener('pointercancel',()=>drag=null);
$('brain').addEventListener('keydown',e=>{if(e.key.startsWith('Arrow')){e.preventDefault();yaw+=e.key==='ArrowLeft'?-.1:e.key==='ArrowRight'?.1:0;pitch+=e.key==='ArrowUp'?-.1:e.key==='ArrowDown'?.1:0;}});
function drawBrain(now){
  const dt=lastBrainFrame===null?0:Math.min((now-lastBrainFrame)/1000,.1);
  lastBrainFrame=now;
  if(!drag&&!recording&&shown?.mode==='running'&&shown?.driver==='neural'&&shown?.live_training?.enabled&&shown?.live_training?.is_training){
    yaw=(yaw+dt*.09)%(Math.PI*2);
  }
  const canvas=$('brain'),ratio=Math.min(devicePixelRatio,2),w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){
    canvas.width=Math.round(w*ratio);
    canvas.height=Math.round(h*ratio);
  }
  brainCtx.setTransform(ratio,0,0,ratio,0,0);
  brainCtx.clearRect(0,0,w,h);
  if(!geometry)return;

  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w*.45,h*.57);
  const counts=shown?.neural.sample_counts||[];
  const roles=geometry.roles||[];
  const nPoints=geometry.points.length;

  const dopamineActive=roles.some((role,i)=>role==='dopamine'&&counts[i]>0);
  dopamineRewardPulse=Math.max(dopamineActive?1:0,dopamineRewardPulse-dt*2.5);
  $('legend-dopamine')?.classList.toggle('reward-glow',dopamineRewardPulse>.15);

  // 1. Project all 3D soma coordinates
  const proj=new Array(nPoints);
  const points=new Array(nPoints);
  for(let i=0;i<nPoints;i++){
    const pt=geometry.points[i];
    const x1=pt[0]*cy+pt[2]*sy, z1=-pt[0]*sy+pt[2]*cy;
    const y1=pt[1]*cp-z1*sp, z2=pt[1]*sp+z1*cp;
    const perspective=3/(3+z2);
    const item={
      x:w/2+x1*scale*perspective,
      y:h*.48+y1*scale*perspective,
      z:z2,
      i,
      role:roles[i]||'intrinsic',
      p:perspective
    };
    proj[i]=item;
    points[i]=item;
  }
  points.sort((a,b)=>b.z-a.z);

  // 2. Draw Biological Synaptic Network (Edges & Action Potential Pulses)
  const edges=geometry.edges||[];
  if(edges.length>0){
    brainCtx.save();

    // Baseline inactive synapses (batched single path for extreme performance)
    brainCtx.beginPath();
    brainCtx.strokeStyle='rgba(110, 88, 142, 0.16)';
    brainCtx.lineWidth=0.65;
    for(let e=0;e<edges.length;e++){
      const edge=edges[e];
      const p1=proj[edge[0]], p2=proj[edge[1]];
      if(!p1||!p2)continue;
      if(counts[edge[0]]>0||counts[edge[1]]>0)continue;
      brainCtx.moveTo(p1.x,p1.y);
      brainCtx.lineTo(p2.x,p2.y);
    }
    brainCtx.stroke();

    // Active synapsing circuits with travelling action potential pulses
    for(let e=0;e<edges.length;e++){
      const edge=edges[e];
      const pre=edge[0], post=edge[1];
      const cPre=counts[pre]>0, cPost=counts[post]>0;
      if(!cPre&&!cPost)continue;

      const p1=proj[pre], p2=proj[post];
      if(!p1||!p2)continue;

      brainCtx.beginPath();
      brainCtx.strokeStyle='rgba(197, 164, 238, 0.55)';
      brainCtx.lineWidth=1.15*Math.min(p1.p,p2.p);
      brainCtx.moveTo(p1.x,p1.y);
      brainCtx.lineTo(p2.x,p2.y);
      brainCtx.stroke();

      // Traveling action potential spike particle
      const tPulse=((now*0.0035)+(pre*0.13))%1.0;
      const px=p1.x+(p2.x-p1.x)*tPulse;
      const py=p1.y+(p2.y-p1.y)*tPulse;
      brainCtx.beginPath();
      brainCtx.fillStyle='#c5a4ee';
      brainCtx.arc(px,py,1.3*p1.p,0,Math.PI*2);
      brainCtx.fill();
    }
    brainCtx.restore();
  }

  // One hue for the entire network; only dopamine activity gets a halo.
  for(const p of points){
    const active=counts[p.i]>0;
    brainCtx.save();
    brainCtx.fillStyle='#c5a4ee';
    brainCtx.globalAlpha=active?.9:Math.max(.22,.60-p.z*.18);
    if(p.role==='dopamine'&&active){
      brainCtx.shadowColor='#c5a4ee';
      brainCtx.shadowBlur=5*p.p;
    }
    brainCtx.beginPath();
    brainCtx.arc(p.x,p.y,Math.max(.4,(active?1.7:1.1)*p.p),0,Math.PI*2);
    brainCtx.fill();
    brainCtx.restore();
  }
}

function drawSpark(){const c=$('spark'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#c5a4ee';ctx.lineWidth=1.4;ctx.beginPath();const max=Math.max(1,...history);for(let i=0;i<history.length;i++){const x=i/(Math.max(2,history.length)-1)*c.width,y=c.height-3-history[i]/max*(c.height-7);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}
function render(now){if(recording&&playback){const fallback=recording.frames[0].game.time+replayOffset+(now-replayStart)/1000;const target=sound&&songAudio&&!songAudio.paused&&!songAudio.ended?songAudio.currentTime+(shown?.game?.song?.audio_offset||0):fallback;let next=replayIndex;while(next<recording.frames.length-1&&recording.frames[next+1].game.time<=target)next++;if(next!==replayIndex){replayIndex=next;display(recording.frames[next]);$('scrub').value=next;}if(next===recording.frames.length-1){playback=false;$('replay-play').textContent='Repetir';}}if(frameImage)gameCtx.drawImage(frameImage,0,0,320,400);stage?.render(now);drawBrain(now);drawSpark();requestAnimationFrame(render);}
fetch('/api/brain').then(r=>{if(!r.ok)throw Error('No se pudo cargar la anatomía');return r.json();}).then(g=>{geometry=g;const rc=g.role_counts||{};$('sample-label').textContent=`${format(g.sample_size)} SOMAS · ${format(g.edge_count||g.edges?.length||0)} CONEXIONES (Audio: ${rc.auditory||0} · Dopamina: ${rc.dopamine||0} · Motor: ${rc.motor||0} · Visual: ${rc.stimulus||0})`}).catch(e=>error(e.message));
loadSongsCatalog();
poll();requestAnimationFrame(render);

// --- Training & Checkpoint Comparison View ---
let trainingRuns = [], currentRunCurve = null, compareRunCurve = null;

function setTab(tab) {
  const isSim = tab === 'sim';
  const navSim = $('nav-sim');
  const navTrain = $('nav-training');
  if (navSim) {
    navSim.classList.toggle('selected', isSim);
    navSim.setAttribute('aria-pressed', String(isSim));
  }
  if (navTrain) {
    navTrain.classList.toggle('selected', !isSim);
    navTrain.setAttribute('aria-pressed', String(!isSim));
  }
  if ($('sim-view')) $('sim-view').hidden = !isSim;
  if ($('training-view')) $('training-view').hidden = isSim;
  if (!isSim && !trainingRuns.length) {
    loadTrainingRuns();
  }
}

$('nav-sim')?.addEventListener('click', () => setTab('sim'));
$('nav-training')?.addEventListener('click', () => setTab('training'));

async function loadTrainingRuns() {
  try {
    const res = await fetch('/api/training/runs');
    if (!res.ok) throw Error('No se pudo cargar el listado de entrenamientos');
    trainingRuns = await res.json();
    const runSelect = $('run-select');
    const compareSelect = $('compare-select');
    if (!runSelect || !compareSelect) return;
    runSelect.innerHTML = '';
    compareSelect.innerHTML = '';

    for (const r of trainingRuns) {
      const opt1 = document.createElement('option');
      opt1.value = r.id;
      opt1.textContent = `${r.name} (${r.checkpoints_count} ckpts)`;
      if (r.id === 'rl/rl-v3') opt1.selected = true;
      runSelect.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = r.id;
      opt2.textContent = `${r.name} (${r.checkpoints_count} ckpts)`;
      if (r.id === 'rl/rl-v2') opt2.selected = true;
      compareSelect.appendChild(opt2);
    }
    await updateSelectedRun();
  } catch (e) {
    error('Error cargando entrenamientos: ' + e.message);
  }
}

async function updateSelectedRun() {
  const runSelect = $('run-select');
  if (!runSelect) return;
  const runId = runSelect.value;
  const isCompare = $('compare-toggle')?.checked;
  const compareId = isCompare ? $('compare-select')?.value : null;
  if ($('compare-select')) $('compare-select').disabled = !isCompare;
  if ($('compare-legend')) $('compare-legend').hidden = !isCompare;

  try {
    const r1 = await fetch('/api/training/curve?run=' + encodeURIComponent(runId));
    if (!r1.ok) throw Error('Error al cargar curva de la corrida');
    currentRunCurve = await r1.json();

    if (compareId) {
      const r2 = await fetch('/api/training/curve?run=' + encodeURIComponent(compareId));
      if (r2.ok) compareRunCurve = await r2.json();
    } else {
      compareRunCurve = null;
    }

    renderTrainingDashboard(currentRunCurve, compareRunCurve);
  } catch (e) {
    error(e.message);
  }
}

$('run-select')?.addEventListener('change', updateSelectedRun);
$('compare-select')?.addEventListener('change', updateSelectedRun);
$('compare-toggle')?.addEventListener('change', () => {
  if ($('compare-select')) $('compare-select').disabled = !$('compare-toggle').checked;
  updateSelectedRun();
});

function renderTrainingDashboard(data, compData) {
  const sum = data.summary || {};
  const curve = data.curve || [];
  const supervised = data.kind === 'supervised_neural_readout';

  if ($('chart-run-badge')) $('chart-run-badge').textContent = data.run_dir || 'ENTRENAMIENTO';
  if ($('kpi-acc')) $('kpi-acc').textContent = sum.initial_accuracy != null ? `${sum.initial_accuracy.toFixed(1)}% → ${sum.final_accuracy.toFixed(1)}%` : '—';
  if ($('kpi-acc-gain')) $('kpi-acc-gain').textContent = sum.accuracy_gain != null ? `+${sum.accuracy_gain.toFixed(1)}%` : (supervised ? 'evaluación final' : '0%');
  if ($('kpi-wrong')) $('kpi-wrong').textContent = sum.initial_wrong != null ? `${sum.initial_wrong.toFixed(1)} → ${sum.final_wrong.toFixed(1)}` : '—';
  if ($('kpi-wrong-diff')) $('kpi-wrong-diff').textContent = sum.wrong_reduction != null ? `-${sum.wrong_reduction.toFixed(1)} fallos` : '0';

  if (curve.length > 0 && $('kpi-score')) {
    if (supervised) {
      $('kpi-score').textContent = sum.final_score == null ? 'Pendiente' : format(Math.round(sum.final_score));
      if ($('kpi-score-gain')) $('kpi-score-gain').textContent = `${curve.length} épocas`;
    } else {
      const initScore = Math.round(curve[0].mean_score || 0);
      const finalScore = Math.round(curve[curve.length - 1].mean_score || 0);
      $('kpi-score').textContent = `${format(initScore)} → ${format(finalScore)}`;
      if ($('kpi-score-gain')) $('kpi-score-gain').textContent = `+${format(finalScore - initScore)}`;
    }
  }

  if ($('kpi-gens')) $('kpi-gens').textContent = String(curve.length);
  if ($('kpi-seeds')) $('kpi-seeds').textContent = data.eval_seeds ? `${data.eval_seeds.length} semillas (${data.eval_seeds.join(', ')})` : 'Semilla fija';
  if ($('checkpoints-meta')) $('checkpoints-meta').textContent = `${curve.length} puntos evaluados · ${data.run_dir || ''}`;

  // Render Checkpoints Table
  const tbody = $('checkpoints-tbody');
  if (tbody) {
    tbody.innerHTML = '';
    const measuredAccuracies = curve.map(c => c.mean_accuracy).filter(Number.isFinite);
    const maxAcc = measuredAccuracies.length ? Math.max(...measuredAccuracies) : null;

    curve.forEach((pt, idx) => {
      const tr = document.createElement('tr');
      const isBest = Number.isFinite(pt.mean_accuracy) && pt.mean_accuracy === maxAcc;
      const accBadge = isBest ? '<span class="badge" style="background:#75d0b130;color:#75d0b1;margin-left:6px">★ MEJOR</span>' : '';
      const sizeStr = pt.file_size_kb ? `${pt.file_size_kb} KB` : (pt.checkpoint === 'baseline' ? 'memoria' : '—');
      const canLoad = pt.checkpoint && pt.checkpoint !== 'baseline';
      const loadBtn = canLoad ? `<button class="quiet small btn-load-ckpt" data-path="${data.run_dir}/${pt.checkpoint}" style="padding:2px 8px;font-size:11px;color:#75d0b1;border-color:#75d0b140" title="Cargar este checkpoint en la simulación 3D">Cargar ▶</button>` : '—';
      tr.innerHTML = `
        <td><strong>${pt.generation ?? idx}</strong></td>
        <td><code>${pt.checkpoint || '—'}</code>${accBadge}</td>
        <td><strong>${Number.isFinite(pt.mean_accuracy) ? `${pt.mean_accuracy.toFixed(1)}%` : '—'}</strong></td>
        <td>${Number.isFinite(pt.mean_hits) ? pt.mean_hits.toFixed(1) : '—'}</td>
        <td>${Number.isFinite(pt.mean_wrong) ? pt.mean_wrong.toFixed(1) : '—'}</td>
        <td>${Number.isFinite(pt.mean_score) ? format(Math.round(pt.mean_score)) : '—'}</td>
        <td style="color:var(--muted)">${sizeStr}</td>
        <td>${loadBtn}</td>
      `;
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll('.btn-load-ckpt').forEach(b => {
      b.onclick = async () => {
        const relPath = b.dataset.path;
        try {
          const selector = $('checkpoint-select');
          if (selector && checkpointRows.some(row => row.path === relPath)) {
            selector.value = relPath;
            await watchCheckpoint();
          } else {
            await control({ type: 'load_checkpoint', path: relPath });
          }
          setTab('sim');
          const genBadge = $('live-gen-status');
          if (genBadge) genBadge.textContent = `Cargado: ${relPath.split('/').pop()}`;
        } catch (e) {
          error(e.message);
        }
      };
    });
  }

  drawTrainingChart(curve, supervised ? null : compData?.curve, supervised);
}

function drawTrainingChart(curve, compareCurve, supervised = false) {
  const canvas = $('training-chart');
  if (!canvas || !curve || !curve.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 920;
  const h = 280;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 45, padR = 30, padT = 20, padB = 30;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  // Horizontal grid (0%, 25%, 50%, 75%, 100%)
  ctx.strokeStyle = '#252330';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#827c8d';
  ctx.font = '10px ui-monospace,monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let pct = 0; pct <= 100; pct += 25) {
    const y = padT + chartH - (pct / 100) * chartH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();
    ctx.fillText(`${pct}%`, padL - 8, y);
  }

  const n = curve.length;
  const xFor = i => padL + (i / Math.max(1, n - 1)) * chartW;
  const yForPct = val => padT + chartH - (Math.max(0, Math.min(100, val)) / 100) * chartH;

  if (supervised) {
    const losses = curve.map(pt => pt.loss).filter(Number.isFinite);
    const maxLoss = Math.max(...losses, 1e-9), minLoss = Math.min(...losses, 0);
    const yForLoss = value => padT + ((value - minLoss) / Math.max(1e-9, maxLoss - minLoss)) * chartH;
    ctx.strokeStyle = '#75d0b1'; ctx.lineWidth = 2.6; ctx.beginPath();
    curve.forEach((pt, i) => {
      if (!Number.isFinite(pt.loss)) return;
      const y = yForLoss(pt.loss);
      if (i === 0) ctx.moveTo(xFor(i), y); else ctx.lineTo(xFor(i), y);
    });
    ctx.stroke();
    ctx.fillStyle = '#75d0b1'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`Pérdida supervisada: ${losses.at(-1)?.toFixed(4) ?? '—'}`, padL + 8, padT + 8);
    ctx.textAlign = 'center';
    const lossStep = Math.max(1, Math.floor(n / 8));
    for (let i = 0; i < n; i += lossStep) ctx.fillText(`Ép ${curve[i].generation ?? i}`, xFor(i), padT + chartH + 8);
    return;
  }

  // X-axis labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {
    ctx.fillText(`Gen ${curve[i].generation ?? i}`, xFor(i), padT + chartH + 8);
  }

  // Draw Comparison Curve if present (amber dashed line)
  if (compareCurve && compareCurve.length) {
    ctx.save();
    ctx.strokeStyle = '#f0bd70';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const cn = compareCurve.length;
    for (let i = 0; i < cn; i++) {
      const cx = padL + (i / Math.max(1, cn - 1)) * chartW;
      const cy = yForPct(compareCurve[i].mean_accuracy || 0);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Curve 3: Reducción de Fallos (% error) (coral)
  // Muestra el descenso del porcentaje de notas no acertadas conforme evoluciona la mosca
  ctx.strokeStyle = '#e88e9c';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  curve.forEach((pt, i) => {
    const errPct = Math.max(0, Math.min(100, 100 - (pt.mean_accuracy || 0)));
    const y = yForPct(errPct);
    if (i === 0) ctx.moveTo(xFor(i), y);
    else ctx.lineTo(xFor(i), y);
  });
  ctx.stroke();

  // Curve 2: Puntuación (% meta de 4.000 pts) (lavender)
  // Progreso intuitivo de puntos acumulados respecto a la meta de la canción
  ctx.strokeStyle = '#c292ff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  curve.forEach((pt, i) => {
    const scorePct = Math.max(0, Math.min(100, ((pt.mean_score || 0) / 4000.0) * 100));
    const y = yForPct(scorePct);
    if (i === 0) ctx.moveTo(xFor(i), y);
    else ctx.lineTo(xFor(i), y);
  });
  ctx.stroke();

  // Curve 1: Accuracy (emerald with gradient area fill)
  const grad = ctx.createLinearGradient(0, padT, 0, padT + chartH);
  grad.addColorStop(0, '#75d0b135');
  grad.addColorStop(1, '#75d0b100');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(xFor(0), padT + chartH);
  curve.forEach((pt, i) => {
    ctx.lineTo(xFor(i), yForPct(pt.mean_accuracy || 0));
  });
  ctx.lineTo(xFor(n - 1), padT + chartH);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#75d0b1';
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  curve.forEach((pt, i) => {
    const y = yForPct(pt.mean_accuracy || 0);
    if (i === 0) ctx.moveTo(xFor(i), y);
    else ctx.lineTo(xFor(i), y);
  });
  ctx.stroke();

  // Dots on accuracy points
  curve.forEach((pt, i) => {
    const x = xFor(i);
    const y = yForPct(pt.mean_accuracy || 0);
    ctx.fillStyle = '#1b1b21';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#75d0b1';
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  // Callout tag on final accuracy point
  if (n > 0) {
    const lastX = xFor(n - 1);
    const lastY = yForPct(curve[n - 1].mean_accuracy || 0);
    const text = `${(curve[n - 1].mean_accuracy || 0).toFixed(1)}%`;
    ctx.font = 'bold 11px ui-monospace,monospace';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = '#21332a';
    ctx.strokeStyle = '#75d0b1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(lastX - tw - 16, lastY - 14, tw + 12, 18, 4);
    else ctx.rect(lastX - tw - 16, lastY - 14, tw + 12, 18);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#c4ffe1';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, lastX - 10 - tw / 2, lastY - 5);
  }
}

function drawLiveTrainingChart(history, currentGen, game) {
  const canvas = $('live-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 460;
  const h = canvas.clientHeight || 125;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padLeft = 32, padRight = 16, padTop = 14, padBottom = 20;
  const plotW = Math.max(10, w - padLeft - padRight);
  const plotH = Math.max(10, h - padTop - padBottom);

  ctx.lineWidth = 1;
  ctx.font = '9px ui-monospace,monospace';
  ctx.fillStyle = '#6a6278';
  ctx.textAlign = 'right';

  for (let pct = 0; pct <= 100; pct += 25) {
    const y = padTop + plotH - (pct / 100.0) * plotH;
    ctx.strokeStyle = pct === 0 ? '#2d2938' : '#181722';
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + plotW, y);
    ctx.stroke();
    ctx.fillText(`${pct}%`, padLeft - 5, y + 3);
  }

  const allPoints = [...history];
  if (game && (game.hits > 0 || game.misses > 0 || game.time > 0.5)) {
    const judged = game.hits + game.misses;
    allPoints.push({
      generation: currentGen,
      accuracy: judged > 0 ? (100.0 * game.hits / judged) : 0,
      score: game.score,
      wrong: game.wrong,
      isLive: true
    });
  }

  if (allPoints.length === 0) {
    ctx.fillStyle = '#9b93a8';
    ctx.textAlign = 'center';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('Generación 0 en curso · Explorando y aprendiendo...', padLeft + plotW / 2, padTop + plotH / 2 + 3);
    return;
  }

  const maxGen = Math.max(currentGen, allPoints.length - 1, 3);
  const xForGen = g => padLeft + (g / maxGen) * plotW;
  const yForPct = p => padTop + plotH - (Math.max(0, Math.min(100, p)) / 100.0) * plotH;

  ctx.textAlign = 'center';
  ctx.fillStyle = '#7a7288';
  for (let g = 0; g <= maxGen; g += (maxGen > 12 ? 4 : maxGen > 6 ? 2 : 1)) {
    const x = xForGen(g);
    ctx.fillText(`G${g}`, x, h - 5);
  }

  const notesCount = (game?.song?.notes_count || 48);
  const targetScore = Math.max(1000, notesCount * 75); // Meta de puntuación para la canción (~3.600 pts)

  function drawSeries(color, scaleFn) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    allPoints.forEach((pt, i) => {
      const g = pt.generation !== undefined ? pt.generation : i;
      const x = xForGen(g);
      const y = yForPct(scaleFn(pt, i));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    allPoints.forEach((pt, i) => {
      const g = pt.generation !== undefined ? pt.generation : i;
      const x = xForGen(g);
      const y = yForPct(scaleFn(pt, i));
      ctx.fillStyle = color;
      ctx.beginPath();
      if (pt.isLive) {
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // 1. Reducción de Fallos en rosa coral: % error (100 - Precisión)
  // Comienza alto (80-95%) y desciende claramente hacia el 0% conforme aprende la mosca
  drawSeries('#e88e9c', pt => Math.max(0, Math.min(100, 100 - (pt.accuracy || 0))));

  // 2. Puntuación en violeta (% de la meta de la canción)
  // Progreso intuitivo de puntos acumulados respecto a la meta de la pista
  drawSeries('#c292ff', pt => Math.max(0, Math.min(100, ((pt.score || 0) / targetScore) * 100)));

  // 3. Precisión en verde menta (% de aciertos)
  drawSeries('#75d0b1', pt => Math.max(0, Math.min(100, pt.accuracy || 0)));

  if (allPoints.length > 0) {
    const last = allPoints[allPoints.length - 1];
    const failPct = Math.round(Math.max(0, 100 - (last.accuracy || 0)));
    const accPct = Math.round(last.accuracy || 0);
    const scoreVal = last.score || 0;
    if ($('live-notes-info')) {
      $('live-notes-info').textContent = `Precisión: ${accPct}% · Fallos: ${failPct}% · Puntos: ${format(scoreVal)}`;
    }
  }
}

// Optional page-scoped inspection tool. Same observable state as the panel.
if(document.modelContext?.registerTool){
  const lifecycle=new AbortController();
  addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  try{Promise.resolve(document.modelContext.registerTool({
    name:'inspect_cueca_session',title:'Inspect Cueca Hero session',
    description:'Read the visible local experiment, clocks, score and neural activity. Does not start or alter a run.',
    inputSchema:{type:'object',properties:{},additionalProperties:false},
    annotations:{readOnlyHint:true,untrustedContentHint:false},
    execute(input){if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).length)throw Error('Expected an empty object');if(!shown)throw Error('Session has not loaded');return {mode:shown.mode,driver:shown.driver,game:shown.game,telemetry:shown.telemetry,backend:shown.neural.backend,replay:!!recording};}
  },{signal:lifecycle.signal})).catch(()=>{});}catch{}
}

async function loadCheckpointCatalogue(){
  try{
    const response=await fetch('/api/checkpoints');if(!response.ok)throw Error('No se pudo leer el catalogo');
    const data=await response.json();checkpointRows=data.checkpoints;
    const select=$('checkpoint-select'),previous=select.value;
    select.replaceChildren();
    for(const row of checkpointRows){const option=document.createElement('option');option.value=row.path;option.textContent=`${row.name} · ${row.metrics.hits}/${row.metrics.total_notes} aciertos`;select.appendChild(option);}
    if(checkpointRows.some(r=>r.path===previous))select.value=previous;
    if(!checkpointRows.length){const option=document.createElement('option');option.value='';option.textContent='Entrenamiento en curso';select.appendChild(option);}
    $('watch-checkpoint').disabled=!select.value;
    const job=data.jobs.at(-1);
    const stages={initializing:'Preparando el cerebro',collect_silent:'Registrando actividad neuronal',collect_teacher:'Recogiendo ejemplos de golpes y sostenidas',training:'Entrenando el lector',evaluating:'Evaluando el checkpoint con MaleCNS completo',complete:'Checkpoint listo'};
    $('checkpoint-status').textContent=job?`${stages[job.stage]||job.stage}${job.completed!==undefined?` · ${job.completed}/200 epocas`:''}${job.seconds!==undefined?` · ${job.seconds.toFixed(0)} s de la cancion`:''}`:'No hay checkpoints evaluados todavia.';
  }catch(e){$('checkpoint-status').textContent=e.message;}
  setTimeout(loadCheckpointCatalogue,15000);
}
async function watchCheckpoint(){
  const path=$('checkpoint-select').value;if(!path)return;
  const button=$('watch-checkpoint');button.disabled=true;button.textContent='Cargando partida...';
  try{
    await control({type:'pause'});
    await control({type:'load_checkpoint',path});
    const response=await fetch('/api/checkpoint-playback?path='+encodeURIComponent(path));
    if(!response.ok)throw Error((await response.json()).error);
    const data=await response.json();if(!data.frames.length)throw Error('La partida esta vacia');
    if(songAudio){songAudio.pause();songAudio.currentTime=0;}
    recording=data;replayIndex=0;replayOffset=0;replayStart=performance.now();playback=true;history=[];
    $('replay-bar').hidden=false;$('scrub').max=data.frames.length-1;$('scrub').value=0;$('replay-play').textContent='Pausar';
    display(data.frames[0]);
    $('checkpoint-status').textContent='Mostrando la partida evaluada. Activa el sonido para escuchar la grabacion.';
  }catch(e){error(e.message);}finally{button.disabled=false;button.textContent='Ver jugar a velocidad normal';}
}
$('watch-checkpoint').onclick=watchCheckpoint;
loadCheckpointCatalogue();
