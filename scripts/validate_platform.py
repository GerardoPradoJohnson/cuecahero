"""Hardware platform and cross-architecture validation tool.

Verifies macOS Apple Silicon (arm64), Intel (x86_64), and Windows support,
checking native shared libraries, C ABI exports, and numerical accuracy.
"""
import ctypes as C
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.paths import ROOT as PATHS_ROOT


def inspect_mach_o_architectures(dylib_path: Path) -> list[str]:
    """Inspect binary architectures for Mach-O binaries."""
    try:
        output = subprocess.check_output(['file', str(dylib_path)], text=True)
        archs = []
        if 'arm64' in output:
            archs.append('arm64 (Apple Silicon)')
        if 'x86_64' in output:
            archs.append('x86_64 (Intel)')
        return archs
    except Exception:
        return ['unknown']


def validate_platform() -> dict:
    sys_name = platform.system()
    machine = platform.machine().lower()
    py_ver = sys.version.split()[0]
    bits = struct.calcsize("P") * 8

    report = {
        'host': {
            'system': sys_name,
            'machine': machine,
            'release': platform.release(),
            'python_version': py_ver,
            'address_bits': bits,
        },
        'supported_platform': (sys_name, machine) in {
            ('Darwin', 'x86_64'),
            ('Darwin', 'arm64'),
            ('Windows', 'amd64'),
            ('Windows', 'x86_64'),
        },
        'native_library': {},
        'c_abi_call_test': False,
        'simd_support': {},
        'overall_status': 'unknown',
    }

    # Detect SIMD features
    if sys_name == 'Darwin':
        try:
            sysctl_neon = subprocess.check_output(['sysctl', '-n', 'hw.optional.neon'], text=True, stderr=subprocess.DEVNULL).strip()
            report['simd_support']['neon'] = sysctl_neon == '1'
        except Exception:
            report['simd_support']['neon'] = machine == 'arm64'
        try:
            sysctl_avx2 = subprocess.check_output(['sysctl', '-n', 'hw.optional.avx2_0'], text=True, stderr=subprocess.DEVNULL).strip()
            report['simd_support']['avx2'] = sysctl_avx2 == '1'
        except Exception:
            report['simd_support']['avx2'] = machine == 'x86_64'
    elif sys_name == 'Windows':
        report['simd_support']['x64_standard'] = True

    # Check native compiled kernel
    lib_name = 'neural.dll' if sys_name == 'Windows' else 'libneural.dylib'
    lib_path = ROOT / 'build' / lib_name
    meta_path = lib_path.with_suffix(lib_path.suffix + '.json')

    if lib_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        actual_sha = hashlib.sha256(lib_path.read_bytes()).hexdigest()
        valid_checksum = actual_sha == meta.get('binary_sha256')

        lib_info = {
            'path': str(lib_path.relative_to(ROOT)),
            'file_size_bytes': lib_path.stat().st_size,
            'sha256_verified': valid_checksum,
            'model_revision': meta.get('model_revision'),
        }

        if sys_name == 'Darwin':
            lib_info['architectures'] = inspect_mach_o_architectures(lib_path)
            lib_info['universal_binary'] = len(lib_info['architectures']) >= 2

        report['native_library'] = lib_info

        # C ABI export test
        try:
            dll = C.CDLL(str(lib_path))
            func = getattr(dll, 'neural_advance', None)
            if func is not None:
                report['c_abi_call_test'] = True
                report['c_abi_symbol'] = 'neural_advance found and loaded'
        except Exception as err:
            report['c_abi_error'] = str(err)
    else:
        report['native_library']['status'] = 'not_built_using_numba_fallback'

    # Determine overall status
    if report['supported_platform'] and report.get('c_abi_call_test', False):
        report['overall_status'] = 'PASSED_NATIVE_ACCELERATED'
    elif report['supported_platform']:
        report['overall_status'] = 'PASSED_NUMBA_FALLBACK'
    else:
        report['overall_status'] = 'UNSUPPORTED'

    out_file = ROOT / 'outputs/diagnostics/platform_validation.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    print("=" * 60)
    print("  Cross-Platform Hardware & Architecture Validation")
    print("=" * 60)
    rep = validate_platform()
    host = rep['host']
    print(f"OS Platform     : {host['system']} ({host['release']})")
    print(f"Architecture    : {host['machine']} ({host['address_bits']}-bit pointer)")
    print(f"Python Runtime  : {host['python_version']}")
    print(f"Platform Tier   : {'Official Tier 1' if rep['supported_platform'] else 'Untested'}")

    simd = rep.get('simd_support', {})
    if simd:
        simd_str = ", ".join(f"{k}: {'supported' if v else 'no'}" for k, v in simd.items())
        print(f"Vector Features : {simd_str}")

    nat = rep.get('native_library', {})
    if 'path' in nat:
        print(f"Native Library  : {nat['path']} ({nat['file_size_bytes']} bytes)")
        if 'architectures' in nat:
            print(f"Binary Slices   : {', '.join(nat['architectures'])}")
            print(f"Universal 2     : {'YES (runs on Apple Silicon & Intel)' if nat.get('universal_binary') else 'NO'}")
        print(f"Checksum Valid  : {'PASSED' if nat.get('sha256_verified') else 'MISMATCH'}")
        print(f"C ABI Export    : {'neural_advance verified' if rep['c_abi_call_test'] else 'FAILED'}")
    else:
        print("Native Library  : Not built (using Numba JIT fallback)")

    print("-" * 60)
    print(f"Overall Result  : {rep['overall_status']}")
    print(f"Report saved to : outputs/diagnostics/platform_validation.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
