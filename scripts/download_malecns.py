"""Download exact upstream files atomically, or reuse verified local copies."""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from data_common import ROOT, LOCK, TARGET, valid, verify

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-dir', type=Path, action='append', default=[], help='Search an existing dataset directory recursively; reuse via symlink.')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    roots = [ROOT / 'external/doomfly/connectome_data', *args.search_dir]
    plan = []
    for name, entry in LOCK.items():
        target = TARGET / name
        source = None
        if target.exists() or target.is_symlink():
            if not valid(target, entry):
                raise RuntimeError(f'Existing file is invalid; preserve and inspect it: {target}')
            source = target
        else:
            aliases = {name, entry['url'].rsplit('/', 1)[-1]}
            for root in roots:
                if root.exists():
                    for alias in aliases:
                        source = next((p.resolve() for p in root.rglob(alias) if valid(p, entry)), None)
                        if source:
                            break
                if source:
                    break
        print(f'{name}: {entry["bytes"] / 1e6:.2f} MB; ' + (f'reuse {source}' if source else entry['url']), flush=True)
        plan.append((name, entry, source))
    if args.plan:
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    needed = sum(e['bytes'] for _, e, s in plan if s is None)
    if shutil.disk_usage(TARGET).free < needed + 256 * 1024**2:
        raise RuntimeError('Insufficient disk space for download.')
    for name, entry, source in plan:
        target = TARGET / name
        if source:
            if source != target:
                # Windows requires Developer Mode for symlinks. Never silently copy GBs.
                target.symlink_to(source)
            continue
        partial = target.with_suffix('.download')
        if not valid(partial, entry):
            with urllib.request.urlopen(entry['url'], timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if not valid(partial, entry):
            raise RuntimeError(f'Checksum mismatch: {partial}')
        partial.replace(target)
        print(f'Verified {name}', flush=True)
    verify()
    (TARGET / 'source.lock.json').write_text(json.dumps(LOCK, indent=2) + '\n')
if __name__ == '__main__':
    main()
