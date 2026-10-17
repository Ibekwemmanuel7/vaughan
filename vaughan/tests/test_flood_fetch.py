"""Fetcher must not persist a source response with a mismatched count."""
import io
import json
import sys
import pytest
from vaughan.flood import fetch_ian


def test_count_mismatch_writes_nothing(tmp_path, monkeypatch):
    responses = iter([{'type': 'FeatureCollection', 'features': []}, {'count': 1}])
    monkeypatch.setattr(fetch_ian, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(next(responses)).encode()))
    out = tmp_path / 'snapshot.geojson'
    monkeypatch.setattr(sys, 'argv', ['fetch_ian', '--out', str(out)])
    with pytest.raises(ValueError, match='Count mismatch'):
        fetch_ian.main()
    assert not out.exists()
