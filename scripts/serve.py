"""Loopback-only local application; no cloud service or GPU needed."""
import argparse
import json
import mimetypes
import queue
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ROOT
from experiments.runtime import Experiment
from visualization.brain import load_geometry
from replay import read_session
from training.history import list_training_runs, get_run_curve

def make_handler(experiment, geometry, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[1]) not in ('200', '202'):
                super().log_message(format, *args)

        def reply(self, status, data, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; media-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'self'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            return self.headers.get('Host') in hosts and self.headers.get('Origin', f'http://127.0.0.1:{port}') in {f'http://{h}' for h in hosts}

        def do_GET(self):
            if not self.trusted():
                return self.reply(403, {'error':'Local origin required'})
            parsed = urlparse(self.path)
            if parsed.path == '/api/state':
                return self.reply(200, experiment.snapshot)
            if parsed.path == '/api/brain':
                return self.reply(200, geometry)
            if parsed.path == '/api/replay':
                session = parse_qs(parsed.query).get('id', [experiment.last_session])[0]
                if not session or not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}', session):
                    return self.reply(404, {'error':'No recorded session yet'})
                try:
                    return self.reply(200, read_session(ROOT/'outputs/sessions'/(session+'.jsonl')))
                except (FileNotFoundError, ValueError):
                    return self.reply(404, {'error':'Session unavailable'})
            if parsed.path == '/api/training/runs':
                return self.reply(200, list_training_runs())
            if parsed.path == '/api/training/curve':
                run_id = parse_qs(parsed.query).get('run', ['rl/rl-v3'])[0]
                try:
                    return self.reply(200, get_run_curve(run_id))
                except (FileNotFoundError, ValueError) as err:
                    return self.reply(404, {'error': str(err)})
            if parsed.path == '/api/songs':
                songs = []
                songs_dir = ROOT / 'config/songs'
                if songs_dir.exists():
                    for f in sorted(songs_dir.glob('*.json')):
                        try:
                            sdata = json.loads(f.read_text())
                            songs.append({
                                'id': sdata.get('id', f.stem),
                                'title': sdata.get('title', f.stem),
                                'subtitle': sdata.get('subtitle', ''),
                                'artist': sdata.get('artist', ''),
                                'bpm': sdata.get('bpm', 108),
                                'bars': sdata.get('bars', 12),
                                'meter': sdata.get('meter', '6/8'),
                                'duration': sdata.get('duration', 0),
                                'audio_url': sdata.get('audio_url', f'/audio/{f.stem}.wav'),
                                'notes_count': len(sdata.get('notes', []))
                            })
                        except Exception:
                            pass
                return self.reply(200, songs)
            if parsed.path.startswith('/audio/'):
                filename = Path(parsed.path).name
                audio_file = ROOT / 'frontend/audio' / filename
                if audio_file.exists() and audio_file.suffix in ('.wav', '.mp3', '.ogg'):
                    ctype = 'audio/wav' if audio_file.suffix == '.wav' else 'audio/mpeg'
                    data = audio_file.read_bytes()
                    file_size = len(data)
                    range_header = self.headers.get('Range')
                    if range_header and range_header.startswith('bytes='):
                        try:
                            parts = range_header[6:].split('-')
                            start = int(parts[0]) if parts[0] else 0
                            end = int(parts[1]) if parts[1] else file_size - 1
                            end = min(end, file_size - 1)
                            chunk = data[start:end + 1]
                            self.send_response(206)
                            self.send_header('Content-Type', ctype)
                            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                            self.send_header('Content-Length', str(len(chunk)))
                            self.send_header('Accept-Ranges', 'bytes')
                            self.send_header('Cache-Control', 'public, max-age=3600')
                            self.send_header('X-Content-Type-Options', 'nosniff')
                            self.end_headers()
                            self.wfile.write(chunk)
                            return
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    self.send_response(200)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                return self.reply(404, {'error': 'Audio file not found'})
            routes = {'/':'index.html', '/app.js':'app.js', '/style.css':'style.css', '/favicon.svg':'favicon.svg', '/stage.js':'stage.js', '/vendor/three.module.js':'vendor/three.module.js'}
            if parsed.path not in routes:
                return self.reply(404, {'error':'Not found'})
            path = ROOT/'frontend'/routes[parsed.path]
            self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

        def do_POST(self):
            if not self.trusted():
                return self.reply(403, {'error':'Local origin required'})
            if self.path != '/api/control':
                return self.reply(404, {'error':'Not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request length')
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError('Command must be an object')
                kind = command.get('type')
                if kind not in ('start','pause','reset','erase_memory','driver','keys','speed','reset_training','select_song','save_checkpoint','load_checkpoint'):
                    raise ValueError('Unknown control')
                if kind == 'driver' and command.get('value') not in ('manual','neural'):
                    raise ValueError('Unknown driver')
                if kind == 'speed' and command.get('value') not in (.5, 1, 2, 4):
                    raise ValueError('Unsupported speed')
                if kind == 'keys' and (not isinstance(command.get('lanes'), list) or len(command['lanes']) > 4 or any(type(v) is not int or v not in range(4) for v in command['lanes'])):
                    raise ValueError('Invalid lanes')
                experiment.enqueue(command)
                self.reply(202, {'accepted':True})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.reply(400, {'error':str(error)})
            except queue.Full:
                self.reply(429, {'error':'Controls are busy'})
    return Handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/cueca_hero.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    geometry = load_geometry(config['visualization']['sample_size'])
    experiment = Experiment(config, geometry)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(experiment, geometry, args.port))
    print(f'Cueca Hero: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        experiment.close()
        server.server_close()
if __name__ == '__main__':
    main()
