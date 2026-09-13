"""Unit tests for platform validation and cross-architecture tool."""
from scripts.validate_platform import validate_platform


def test_platform_validation():
    report = validate_platform()
    assert report['host']['system'] in ('Darwin', 'Windows', 'Linux')
    assert report['host']['address_bits'] == 64
    assert report['supported_platform'] is True
    assert report['overall_status'] in ('PASSED_NATIVE_ACCELERATED', 'PASSED_NUMBA_FALLBACK')
    if report['host']['system'] == 'Darwin':
        assert 'architectures' in report.get('native_library', {})
        archs = report['native_library']['architectures']
        assert any('arm64' in a for a in archs)
        assert any('x86_64' in a for a in archs)
