"""Invoke the official importer/preparer with relocated output roots."""
import argparse
import sys
from data_common import ROOT, DATA, verify

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    verify()
    sys.path.insert(0, str(ROOT / 'external/doomfly'))
    from doom import connectome, prepare
    connectome.ROOT = DATA  # REGISTRY stays pinned to the upstream checkout.
    prepare.ROOT = DATA
    report = connectome.import_graph('malecns_v1')
    print(f'Imported {report["retained_neuron_candidates"]} neurons', flush=True)
    prepare.prepare('malecns_v1')
    if args.audit:
        from doom import audit_data
        (DATA / 'outputs/doom/audit').mkdir(parents=True, exist_ok=True)
        audit_data.ROOT = DATA
        audit_data.main()
if __name__ == '__main__':
    main()
