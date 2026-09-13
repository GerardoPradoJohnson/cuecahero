"""Generate authentic 6/8 acoustic cueca audio tracks and synchronized note charts."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
ROOT = Path(__file__).resolve().parents[1]

def karplus_strong(freq, duration_sec, sample_rate=SAMPLE_RATE, decay=0.985):
    """Plucked acoustic string synthesizer."""
    n_samples = int(sample_rate * duration_sec)
    period = int(sample_rate / freq)
    if period <= 0:
        return np.zeros(n_samples)
    ring_buffer = np.random.uniform(-1.0, 1.0, period)
    output = np.zeros(n_samples)
    idx = 0
    prev_sample = 0.0
    for i in range(n_samples):
        new_sample = decay * 0.5 * (ring_buffer[idx] + prev_sample)
        prev_sample = new_sample
        ring_buffer[idx] = new_sample
        output[i] = new_sample
        idx = (idx + 1) % period
    return output

def handclap_pandero(duration_sec=0.18, sample_rate=SAMPLE_RATE):
    """Acoustic cueca pandero / palmas."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, False)
    noise = np.random.uniform(-1.0, 1.0, n_samples)
    f_center = 2400.0
    env = np.exp(-38.0 * t)
    bursts = np.zeros(n_samples)
    for offset_ms in (0, 8, 18):
        k = int(sample_rate * offset_ms / 1000.0)
        if k < n_samples:
            bursts[k:] += np.exp(-60.0 * (t[k:] - t[k]))
    signal = noise * (env + 0.6 * bursts) * np.sin(2 * math.pi * f_center * t)
    return signal / (np.max(np.abs(signal)) + 1e-8)

def acoustic_bass(freq, duration_sec, sample_rate=SAMPLE_RATE):
    """Resonant acoustic bass / guitarrón."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, False)
    env = np.exp(-3.5 * t)
    return (np.sin(2 * math.pi * freq * t) + 0.45 * np.sin(4 * math.pi * freq * t)) * env

def build_cueca(song_id, title, subtitle, artist, bpm, bars, chords, intro_sec=2.4):
    pulse = 60.0 / bpm / 3.0
    bar_duration = 6.0 * pulse
    total_duration = intro_sec + bars * bar_duration + 2.2
    n_total_samples = int(SAMPLE_RATE * total_duration)
    
    left = np.zeros(n_total_samples, dtype=np.float64)
    right = np.zeros(n_total_samples, dtype=np.float64)
    notes_chart = []
    
    # Intro count-in clicks
    for c in range(3):
        t_click = intro_sec - (3 - c) * (pulse * 2)
        if t_click >= 0:
            click = handclap_pandero(0.08) * 0.4
            s_idx = int(t_click * SAMPLE_RATE)
            e_idx = min(s_idx + len(click), n_total_samples)
            left[s_idx:e_idx] += click[:e_idx - s_idx]
            right[s_idx:e_idx] += click[:e_idx - s_idx]

    # Generate bars
    for bar in range(bars):
        chord = chords[(bar // 2) % len(chords)]
        bar_start_t = intro_sec + bar * bar_duration
        
        # 1. Carril 0 (D): Bass (Pulse 0)
        t_bass = bar_start_t
        bass_snd = acoustic_bass(chord['bass'], pulse * 3.5) * 0.75
        s_idx = int(t_bass * SAMPLE_RATE)
        e_idx = min(s_idx + len(bass_snd), n_total_samples)
        left[s_idx:e_idx] += bass_snd[:e_idx - s_idx] * 0.7
        right[s_idx:e_idx] += bass_snd[:e_idx - s_idx] * 0.7
        notes_chart.append({'id': len(notes_chart), 'lane': 0, 'at': round(t_bass, 4), 'judgement': None})
        
        # 2. Carril 3 (K): Pandero / Palmas (Pulse 2)
        t_pandero = bar_start_t + 2 * pulse
        clap_snd = handclap_pandero(0.15) * 0.65
        s_idx = int(t_pandero * SAMPLE_RATE)
        e_idx = min(s_idx + len(clap_snd), n_total_samples)
        left[s_idx:e_idx] += clap_snd[:e_idx - s_idx] * 0.4
        right[s_idx:e_idx] += clap_snd[:e_idx - s_idx] * 0.85
        notes_chart.append({'id': len(notes_chart), 'lane': 3, 'at': round(t_pandero, 4), 'judgement': None})
        
        # 3. Carril 1 (F): Rasgueo Guitarra (Pulse 3)
        t_strum = bar_start_t + 3 * pulse
        strum_snd = np.zeros(int(SAMPLE_RATE * pulse * 2.8))
        for note_f in chord['notes']:
            strum_snd += karplus_strong(note_f, pulse * 2.8) * 0.22
        s_idx = int(t_strum * SAMPLE_RATE)
        e_idx = min(s_idx + len(strum_snd), n_total_samples)
        left[s_idx:e_idx] += strum_snd[:e_idx - s_idx] * 0.8
        right[s_idx:e_idx] += strum_snd[:e_idx - s_idx] * 0.5
        notes_chart.append({'id': len(notes_chart), 'lane': 1, 'at': round(t_strum, 4), 'judgement': None})
        
        # 4. Carril 2 (J): Punteo Melódico (Pulse 5)
        t_lead = bar_start_t + 5 * pulse
        lead_freq = chord['notes'][(bar * 2 + 1) % len(chord['notes'])] * 1.5
        lead_snd = karplus_strong(lead_freq, pulse * 1.8) * 0.45
        s_idx = int(t_lead * SAMPLE_RATE)
        e_idx = min(s_idx + len(lead_snd), n_total_samples)
        left[s_idx:e_idx] += lead_snd[:e_idx - s_idx] * 0.6
        right[s_idx:e_idx] += lead_snd[:e_idx - s_idx] * 0.6
        notes_chart.append({'id': len(notes_chart), 'lane': 2, 'at': round(t_lead, 4), 'judgement': None})

    # End ring
    t_end = intro_sec + bars * bar_duration
    for f in [110.0, 220.0, 277.2, 329.6, 440.0]:
        snd = karplus_strong(f, 2.2) * 0.35
        s_idx = int(t_end * SAMPLE_RATE)
        e_idx = min(s_idx + len(snd), n_total_samples)
        left[s_idx:e_idx] += snd[:e_idx - s_idx]
        right[s_idx:e_idx] += snd[:e_idx - s_idx]
        
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    left = (left / peak * 0.88 * 32767).astype(np.int16)
    right = (right / peak * 0.88 * 32767).astype(np.int16)
    stereo = np.column_stack([left, right])
    duration = round(notes_chart[-1]['at'] + 1.6, 4)
    
    # Save files
    audio_dir = ROOT / 'frontend/audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f'{song_id}.wav'
    wavfile.write(str(audio_path), SAMPLE_RATE, stereo)
    
    meta = {
        'id': song_id,
        'title': title,
        'subtitle': subtitle,
        'artist': artist,
        'bpm': bpm,
        'bars': bars,
        'meter': '6/8',
        'duration': duration,
        'audio_url': f'/audio/{song_id}.wav',
        'notes': notes_chart
    }
    
    songs_dir = ROOT / 'config/songs'
    songs_dir.mkdir(parents=True, exist_ok=True)
    (songs_dir / f'{song_id}.json').write_text(json.dumps(meta, indent=2))
    print(f'Created song {song_id}: {audio_path.name} ({duration:.2f}s, {len(notes_chart)} notes)')
    return meta

def build_la_consentida():
    """Synthesize authentic 'La Consentida' by Jaime Atria in A major (6/8 cueca)."""
    bpm = 112
    bars = 24
    intro_sec = 2.4
    pulse = 60.0 / bpm / 3.0  # 6 pulses per bar
    bar_duration = 6.0 * pulse
    total_duration = intro_sec + bars * bar_duration + 2.5
    n_total_samples = int(SAMPLE_RATE * total_duration)

    left = np.zeros(n_total_samples, dtype=np.float64)
    right = np.zeros(n_total_samples, dtype=np.float64)
    notes_chart = []

    # Intro count-in clicks
    for c in range(3):
        t_click = intro_sec - (3 - c) * (pulse * 2)
        if t_click >= 0:
            click = handclap_pandero(0.08) * 0.45
            s_idx = int(t_click * SAMPLE_RATE)
            e_idx = min(s_idx + len(click), n_total_samples)
            left[s_idx:e_idx] += click[:e_idx - s_idx]
            right[s_idx:e_idx] += click[:e_idx - s_idx]

    # Melodic notes for "La Consentida" (Hz)
    A3, B3, C4, Cs4, D4, E4, Fs4, Gs4, A4, B4, Cs5 = (
        220.0, 246.94, 261.63, 277.18, 293.66, 329.63, 369.99, 415.30, 440.0, 493.88, 554.37
    )
    
    # Chords
    chord_A = {'bass': 110.0, 'strum': [Cs4, E4, A4]}
    chord_E7 = {'bass': 82.4, 'strum': [D4, Gs4, B4]}
    chord_D = {'bass': 73.4, 'strum': [D4, Fs4, A4]}

    # Melodic phrases: (bar, pulse_offset, freq, sustain_sec, lane)
    # Famous lyrics:
    # "Déjame que te llame la consentida..."
    # "porque todo consigues con tus porfías..."
    # "Primero mi cariño, mi vida..."
    # "y en tus brazos mi vida, la consentida!"
    melody_events = [
        # Intro Floreo (Bars 0-3)
        (0, 0, A4, 0.0), (0, 3, Cs5, 0.0), (1, 0, B4, 0.0), (1, 3, A4, 0.4),
        (2, 0, E4, 0.0), (2, 3, Gs4, 0.0), (3, 0, A4, 0.0), (3, 3, A4, 0.6),
        # Estrofa 1: "Dé-ja-me que te lla-me... la con-sen-ti-da..."
        (4, 0, E4, 0.0), (4, 2, Cs4, 0.0), (4, 4, A3, 0.0),      # "Dé-ja-me"
        (5, 0, E4, 0.0), (5, 2, Fs4, 0.0), (5, 4, Gs4, 0.0),     # "que te lla-"
        (6, 0, A4, 0.0), (6, 3, Cs5, 0.0),                        # "-me la con-"
        (7, 0, B4, 0.0), (7, 2, A4, 0.85),                        # "-sen-ti-daaaa!" (Sustain)
        # Estrofa 2: "Por-que to-do con-si-gues... con tus por-fí-as..."
        (8, 0, E4, 0.0), (8, 2, Cs4, 0.0), (8, 4, A3, 0.0),      # "Por-que to-do"
        (9, 0, E4, 0.0), (9, 2, Fs4, 0.0), (9, 4, Gs4, 0.0),     # "con-si-gues"
        (10, 0, A4, 0.0), (10, 3, Cs5, 0.0),                       # "con tus por-"
        (11, 0, B4, 0.0), (11, 2, A4, 0.85),                       # "-fí-aaaaas!" (Sustain)
        # Copla / Estribillo: "Pri-me-ro mi ca-ri-ño, mi vi-da, lue-go tu or-gu-llo..."
        (12, 0, A4, 0.0), (12, 2, Gs4, 0.0), (12, 4, Fs4, 0.0),   # "Pri-me-ro"
        (13, 0, E4, 0.0), (13, 3, Fs4, 0.0),                       # "mi ca-ri-ño"
        (14, 0, Gs4, 0.0), (14, 2, A4, 0.0), (14, 4, B4, 0.75),   # "mi vi-da, lue-go tu or-gu-llo" (Sustain)
        (15, 0, Cs5, 0.0), (15, 2, B4, 0.0), (15, 4, A4, 0.8),    # "el sol que bri-llaaaa" (Sustain)
        # Zapateo Instrumental / Repique (Bars 16-19)
        (16, 0, E4, 0.0), (16, 3, A4, 0.0), (17, 0, Cs5, 0.0), (17, 3, B4, 0.0),
        (18, 0, A4, 0.0), (18, 3, Gs4, 0.0), (19, 0, Fs4, 0.0), (19, 3, E4, 0.6),
        # Remate Final: "y en tus bra-zos mi vi-da, ¡la con-sen-ti-da!"
        (20, 0, Cs5, 0.0), (20, 2, B4, 0.0), (20, 4, A4, 0.0),    # "y en tus bra-zos"
        (21, 0, Gs4, 0.0), (21, 2, Fs4, 0.0), (21, 4, E4, 0.0),    # "mi vi-da"
        (22, 0, A4, 0.0), (22, 3, Cs5, 0.0),                       # "¡la con-sen-"
        (23, 0, A4, 1.4),                                          # "-ti-daaaaa!" (Gran Sustain final)
    ]

    for bar in range(bars):
        bar_start_t = intro_sec + bar * bar_duration
        curr_chord = chord_A if bar in (0, 3, 4, 7, 8, 11, 15, 16, 19, 22, 23) else chord_E7 if bar in (1, 5, 9, 13, 17, 21) else chord_D
        
        # 1. Carril 0 (D): Bajo folclórico (tónica y dominante en tiempos 0 y 3)
        for p_b in (0, 3):
            t_bass = bar_start_t + p_b * pulse
            b_freq = curr_chord['bass'] * (1.5 if p_b == 3 else 1.0)
            b_snd = acoustic_bass(b_freq, pulse * 2.8) * 0.72
            s_idx = int(t_bass * SAMPLE_RATE)
            e_idx = min(s_idx + len(b_snd), n_total_samples)
            left[s_idx:e_idx] += b_snd[:e_idx - s_idx] * 0.7
            right[s_idx:e_idx] += b_snd[:e_idx - s_idx] * 0.7
            notes_chart.append({'id': len(notes_chart), 'lane': 0, 'at': round(t_bass, 4), 'sustain': 0.0, 'judgement': None})

        # 2. Carril 1 (F): Rasgueo de guitarra cuequera (tiempos 1 y 4)
        for p_s in (1, 4):
            t_strum = bar_start_t + p_s * pulse
            strum_dur = pulse * 2.5
            strum_snd = np.zeros(int(SAMPLE_RATE * strum_dur))
            for note_f in curr_chord['strum']:
                strum_snd += karplus_strong(note_f, strum_dur) * 0.24
            s_idx = int(t_strum * SAMPLE_RATE)
            e_idx = min(s_idx + len(strum_snd), n_total_samples)
            left[s_idx:e_idx] += strum_snd[:e_idx - s_idx] * 0.75
            right[s_idx:e_idx] += strum_snd[:e_idx - s_idx] * 0.55
            # Add sustain on bar 23 final chord
            s_sus = 1.4 if (bar == 23 and p_s == 4) else 0.0
            notes_chart.append({'id': len(notes_chart), 'lane': 1, 'at': round(t_strum, 4), 'sustain': s_sus, 'judgement': None})

        # 3. Carril 3 (K): Pandero cuequero y palmas (tiempos 2 y 5)
        for p_k in (2, 5):
            t_pandero = bar_start_t + p_k * pulse
            clap_snd = handclap_pandero(0.16) * 0.68
            s_idx = int(t_pandero * SAMPLE_RATE)
            e_idx = min(s_idx + len(clap_snd), n_total_samples)
            left[s_idx:e_idx] += clap_snd[:e_idx - s_idx] * 0.45
            right[s_idx:e_idx] += clap_snd[:e_idx - s_idx] * 0.85
            notes_chart.append({'id': len(notes_chart), 'lane': 3, 'at': round(t_pandero, 4), 'sustain': 0.0, 'judgement': None})

    # 4. Carril 2 (J): Melodía vocal auténtica de "La Consentida" en guitarra criolla
    for (m_bar, m_pulse, m_freq, m_sus) in melody_events:
        t_lead = intro_sec + m_bar * bar_duration + m_pulse * pulse
        lead_dur = max(pulse * 1.8, m_sus + 0.3)
        lead_snd = karplus_strong(m_freq, lead_dur, decay=0.99) * 0.65
        # Add acoustic body warmth
        t_vec = np.linspace(0, lead_dur, len(lead_snd), False)
        lead_snd += 0.25 * np.sin(2 * math.pi * m_freq * t_vec) * np.exp(-3.0 * t_vec)
        s_idx = int(t_lead * SAMPLE_RATE)
        e_idx = min(s_idx + len(lead_snd), n_total_samples)
        left[s_idx:e_idx] += lead_snd[:e_idx - s_idx] * 0.85
        right[s_idx:e_idx] += lead_snd[:e_idx - s_idx] * 0.85
        notes_chart.append({'id': len(notes_chart), 'lane': 2, 'at': round(t_lead, 4), 'sustain': round(m_sus, 3), 'judgement': None})

    # Sort notes by time and re-index
    notes_chart.sort(key=lambda n: (n['at'], n['lane']))
    for idx, n in enumerate(notes_chart):
        n['id'] = idx

    # Outro ring
    t_end = intro_sec + bars * bar_duration
    for f in [A3, Cs4, E4, A4]:
        snd = karplus_strong(f, 2.5, decay=0.992) * 0.35
        s_idx = int(t_end * SAMPLE_RATE)
        e_idx = min(s_idx + len(snd), n_total_samples)
        left[s_idx:e_idx] += snd[:e_idx - s_idx]
        right[s_idx:e_idx] += snd[:e_idx - s_idx]

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    left = (left / peak * 0.88 * 32767).astype(np.int16)
    right = (right / peak * 0.88 * 32767).astype(np.int16)
    stereo = np.column_stack([left, right])
    duration = round(notes_chart[-1]['at'] + max(1.6, float(notes_chart[-1].get('sustain', 0.0)) + 0.8), 4)

    # Save files
    audio_dir = ROOT / 'frontend/audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / 'la_consentida.wav'
    wavfile.write(str(audio_path), SAMPLE_RATE, stereo)

    meta = {
        'id': 'la_consentida',
        'title': 'La Consentida',
        'subtitle': 'Cueca Tradicional Chilena',
        'artist': 'Jaime Atria / Tradicional',
        'bpm': bpm,
        'bars': bars,
        'meter': '6/8',
        'duration': duration,
        'audio_url': '/audio/la_consentida.wav',
        'notes': notes_chart
    }
    songs_dir = ROOT / 'config/songs'
    songs_dir.mkdir(parents=True, exist_ok=True)
    (songs_dir / 'la_consentida.json').write_text(json.dumps(meta, indent=2))
    sustain_count = sum(1 for n in notes_chart if n.get('sustain', 0) > 0.1)
    print(f"Created authentic song la_consentida: {audio_path.name} ({duration:.2f}s, {len(notes_chart)} notes, {sustain_count} hold notes)")
    return meta

def main():
    # 1. "La Consentida" - Melodía y compás 6/8 auténtico con notas sostenidas
    build_la_consentida()

    # 2. "Primer Pañuelo" - Pista Original de Prueba (108 BPM, 12 compases)
    chords_do = [
        {'bass': 130.8, 'notes': [261.6, 329.6, 392.0, 523.2]},       # C mayor
        {'bass': 98.0,  'notes': [196.0, 246.9, 293.7, 349.2]},       # G7
        {'bass': 87.3,  'notes': [174.6, 220.0, 261.6, 349.2]},       # F mayor
        {'bass': 130.8, 'notes': [261.6, 329.6, 392.0, 523.2]},       # C mayor
    ]
    build_cueca(
        song_id='primer_panuelo',
        title='Primer Pañuelo',
        subtitle='Pista de Prueba Cuequera',
        artist='Laboratorio FlyLab',
        bpm=108,
        bars=12,
        chords=chords_do
    )

if __name__ == '__main__':
    main()

