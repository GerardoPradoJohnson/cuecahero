"""Build a playable, audio-derived chart. Offline only; never feeds the brain.
Requires requirements-audio.txt. Keeps the supplied recording unchanged.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / 'frontend/audio/consentida.mp3'
OUT = ROOT / 'outputs/audio-review'
OFFSET = 2.4

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(AUDIO.read_bytes()).hexdigest()
    cache = OUT / 'consentida-analysis.npz'
    identity = OUT / 'consentida-analysis.sha256'
    if not cache.exists() or not identity.exists() or identity.read_text() != digest:
        y, sr = librosa.load(AUDIO, sr=22050)
        hop = 256
        H, P = librosa.decompose.hpss(abs(librosa.stft(y, n_fft=2048, hop_length=hop)))
        onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(P, ref=np.max), sr=sr, hop_length=hop)
        _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, start_bpm=110, tightness=60)
        frames = librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=hop)
        pitch, magnitude = librosa.piptrack(S=H, sr=sr, fmin=160, fmax=900, threshold=.35)
        idx = magnitude.argmax(axis=0)
        f0 = pitch[idx, np.arange(pitch.shape[1])]
        np.savez_compressed(cache, sr=sr, hop=hop, onset=onset, beats=beats, onsets=frames,
                            f0=f0, rms=librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0])
        identity.write_text(digest)
    z = np.load(cache)
    dt = float(z['hop'] / z['sr'])
    onset, rms, frames = z['onset'], z['rms'], z['onsets']
    f0 = z['f0']
    midi = median_filter(np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1)), 0), size=9)
    duration = sf.info(AUDIO).duration
    active = np.flatnonzero(rms > max(.008, float(np.max(rms)) * .035))
    ending = min(duration, float(active[-1] * dt + .1))
    # Keep strong attacks, thinning locally by strength rather than a fixed grid.
    candidates = [int(f) for f in frames if rms[f] > .012 and f*dt < ending-.15]
    threshold = float(np.quantile(onset[candidates], .24))
    chosen = []
    for f in sorted((f for f in candidates if onset[f] >= threshold), key=lambda f: -onset[f]):
        if all(abs(f-g)*dt >= .215 for g in chosen):
            chosen.append(f)
    chosen.sort()
    pitches=[]
    for f in chosen:
        values=midi[f+2:f+12];values=values[values>0]
        pitches.append(float(np.median(values)) if len(values) else 69.)
    boundaries=np.quantile(pitches,[.25,.5,.75])
    notes=[]
    for f,pitch in zip(chosen,pitches):
        notes.append(dict(id=len(notes),lane=int(np.searchsorted(boundaries,pitch)),
                          at=round(f*dt+OFFSET,4),sustain=0.,judgement=None,
                          audio_at=round(f*dt,4),pitch_midi=round(pitch,2)))
    # Holds require a stable harmonic pitch in the recording, not just a long gap.
    rounded=np.round(midi)
    starts=np.r_[0,np.flatnonzero(np.diff(rounded)!=0)+1]
    ends=np.r_[starts[1:],len(midi)]
    holds=[]
    for a,b in zip(starts,ends):
        if (b-a)*dt < .36 or np.median(midi[a:b]) < 45 or np.mean(rms[a:b]) < .02:
            continue
        attacks=[int(f) for f in frames if -.12 <= (f-a)*dt <= .10 and rms[f]>.02]
        if not attacks:continue
        attack=min(attacks,key=lambda f:abs(f-a))
        at=round(attack*dt,4)
        pitch=float(np.median(midi[a:b]))
        near=[n for n in notes if abs(n['audio_at']-at)<.215]
        # Prioritize this measured sustained tone over nearby accompaniment taps.
        if any(n.get('sustain',0) for n in near):continue
        note=dict(id=-1,lane=int(np.searchsorted(boundaries,pitch)),at=round(at+OFFSET,4),
                  sustain=0.,judgement=None,audio_at=at,pitch_midi=round(pitch,2))
        following=next((n['audio_at'] for n in notes if n not in near and n['lane']==note['lane'] and n['audio_at']>note['audio_at']),ending)
        end=min(b*dt-.025,following-.12,note['audio_at']+1.5,ending)
        length=end-note['audio_at']
        if length>=.32:
            notes=[n for n in notes if n not in near]
            notes.append(note);notes.sort(key=lambda n:n['at'])
            note['sustain']=round(length,4)
            holds.append(dict(note=note['id'],audio_start=note['audio_at'],audio_end=round(end,4),stable_pitch_midi=round(float(np.median(midi[a:b])),2)))
    notes.sort(key=lambda n:n['at'])
    for i,n in enumerate(notes):n['id']=i
    holds=[dict(note=n['id'],audio_start=n['audio_at'],audio_end=round(n['audio_at']+n['sustain'],4),stable_pitch_midi=n['pitch_midi']) for n in notes if n['sustain']>0]
    beats=z['beats']*dt
    bpm=int(round(60/np.median(np.diff(beats))))
    song=dict(id='la_consentida',title='La Consentida',subtitle='Grabaci\u00f3n aportada \u00b7 mapa adaptado al audio',
              artist='Jaime Atria (composici\u00f3n) \u00b7 int\u00e9rprete no identificado',bpm=bpm,
              bars=int(np.ceil(len(beats)/2)),meter='6/8',duration=round(duration+OFFSET,6),
              audio_url='/audio/consentida.mp3',audio_offset=OFFSET,
              beat_times=[round(float(t)+OFFSET,4) for t in beats],notes=notes,
              chart_source=dict(audio_sha256=digest,method='Spectral attacks; dominant harmonic register mapped low to high D/F/J/K; stable-pitch holds; locally spaced for playability.',
                                limitations='Automatic playable arrangement of the polyphonic recording, not a verified vocal score.',version=1))
    target=ROOT/'config/songs/la_consentida.json'
    backup=OUT/'la_consentida-before-mp3.json'
    if target.exists() and not backup.exists():backup.write_bytes(target.read_bytes())
    target.write_text(json.dumps(song,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    report=dict(audio_sha256=digest,audio_duration=duration,game_duration=song['duration'],audio_offset=OFFSET,
                bpm_estimate=bpm,notes=len(notes),holds=holds,first_note_audio=notes[0]['audio_at'],last_note_audio=notes[-1]['audio_at'],
                min_attack_spacing=min(np.diff([n['audio_at'] for n in notes])),lane_counts=[sum(n['lane']==i for n in notes) for i in range(4)],
                all_heads_on_detected_attacks=True,limitations=song['chart_source']['limitations'])
    (OUT/'consentida-mp3-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
