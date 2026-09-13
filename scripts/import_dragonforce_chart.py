"""Adapt a public Clone Hero chart to the user-provided five-minute audio edit.

The source chart is not bundled. Pass its notes.chart path with --chart.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ROOT

RESOLUTION = 192
COUNT_IN = 2.0
LANES = (0, 1, 2, 2, 3)

# Audio correlation plateaus measured against the 7:21 chart audio. Each tuple is
# target-audio start/end and source-audio offset. The target edit removes two intro
# phrases, most of the central solo, and two later passages without time-stretching.
ALIGNMENT = (
    (0.0, 9.0, 0.664),
    (9.0, 16.0, 10.264),
    (16.0, 200.0, 19.864),
    (200.0, 268.0, 98.642),
    (268.0, 285.0, 117.842),
    (285.0, 300.142, 137.042),
)


def section(text, name):
    match = re.search(rf'^\[{re.escape(name)}\]\s*\{{(.*?)^\}}', text, re.M | re.S)
    if not match:
        raise ValueError(f'Missing [{name}] section')
    return match.group(1)


def tempo_map(text):
    changes = [(int(tick), int(raw) / 1000.0) for tick, raw in
               re.findall(r'^\s*(\d+)\s*=\s*B\s+(\d+)', section(text, 'SyncTrack'), re.M)]
    if not changes or changes[0][0] != 0:
        raise ValueError('Chart tempo map must begin at tick 0')
    return changes


def tick_seconds(tick, changes):
    elapsed = 0.0
    for i, (start, bpm) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else tick
        if tick <= start:
            break
        span = min(tick, end) - start
        if span > 0:
            elapsed += span / RESOLUTION * 60.0 / bpm
        if tick < end:
            break
    return elapsed


def source_to_target(source_time):
    for target_start, target_end, offset in ALIGNMENT:
        target = source_time - offset
        if target_start <= target < target_end:
            return target, target_end
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart', required=True, type=Path)
    args = parser.parse_args()
    text = args.chart.read_text(encoding='utf-8-sig')
    changes = tempo_map(text)
    raw_notes = [(int(t), int(lane), int(length)) for t, lane, length in
                 re.findall(r'^\s*(\d+)\s*=\s*N\s+([0-4])\s+(\d+)',
                            section(text, 'ExpertSingle'), re.M)]

    adapted = {}
    for tick, source_lane, length in raw_notes:
        source_at = tick_seconds(tick, changes)
        mapped = source_to_target(source_at)
        if not mapped:
            continue
        target_at, segment_end = mapped
        lane = LANES[source_lane]
        sustain = 0.0
        if length:
            source_end = tick_seconds(tick + length, changes)
            mapped_end = source_to_target(source_end)
            if mapped_end:
                sustain = max(0.0, mapped_end[0] - target_at)
            else:
                sustain = max(0.0, segment_end - target_at)
        key = (round(target_at + COUNT_IN, 6), lane)
        adapted[key] = max(adapted.get(key, 0.0), sustain)

    notes = []
    last_lane_time = [-999.0] * 4
    for (at, lane), sustain in sorted(adapted.items()):
        # The renderer and keyboard controls run at 30 Hz and need a release frame
        # between repeated presses on one lane.
        if at - last_lane_time[lane] < 0.075:
            continue
        last_lane_time[lane] = at
        note = {'id': len(notes), 'lane': lane, 'at': at}
        if sustain >= 0.12:
            note['sustain'] = round(sustain, 6)
        notes.append(note)

    for lane in range(4):
        lane_notes = [note for note in notes if note['lane'] == lane]
        for note, following in zip(lane_notes, lane_notes[1:]):
            if 'sustain' in note:
                available = following['at'] - note['at'] - 0.06
                if available < 0.12:
                    note.pop('sustain')
                else:
                    note['sustain'] = round(min(note['sustain'], available), 6)

    audio = ROOT / 'frontend/audio/dragonforce.mp3'
    duration = sf.info(audio).duration
    beats = []
    max_tick = max(t for t, _, _ in raw_notes)
    for tick in range(0, max_tick + 1, RESOLUTION):
        mapped = source_to_target(tick_seconds(tick, changes))
        if mapped:
            beats.append(round(mapped[0] + COUNT_IN, 6))

    output = {
        'id': 'through_the_fire_and_flames',
        'title': 'Through the Fire and Flames',
        'subtitle': 'DragonForce · chart adaptado a cuatro teclas',
        'artist': 'DragonForce',
        'album': 'Inhuman Rampage',
        'year': 2006,
        'bpm': 200,
        'meter': '4/4',
        'audio_url': '/audio/dragonforce.mp3',
        'audio_offset': COUNT_IN,
        'duration': round(duration + COUNT_IN, 6),
        'beat_times': beats,
        'notes': notes,
        'chart_source': {
            'format': 'Clone Hero notes.chart · ExpertSingle',
            'charter': 'MrOriginalityCH',
            'repository': 'https://github.com/ChristianMendozaa/CH_MGVNUN',
            'source_duration_seconds': 441.237,
            'target_duration_seconds': round(duration, 6),
            'adaptation': 'Piecewise audio correlation, five frets folded into four lanes.',
            'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
        },
    }
    target = ROOT / 'config/songs/through_the_fire_and_flames.json'
    target.write_text(json.dumps(output, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(f'{target}: {len(notes)} notes, {sum("sustain" in note for note in notes)} sustains')


if __name__ == '__main__':
    main()
