"""Fetch the bounded public reference snapshot; raw data remains untracked."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from .inspection import QUERY, SOURCE, BBOX, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/ian/hwms.geojson'))
    args = p.parse_args()
    with urlopen(QUERY, timeout=60) as response:
        raw = response.read()
    d = json.loads(raw)
    marks = evidence(d)
    # Count query detects silent transfer limits or a changing upstream snapshot.
    with urlopen(QUERY.replace('f=geojson', 'f=json') + '&returnCountOnly=true', timeout=60) as response:
        count = json.load(response)['count']
    if count != len(marks):
        raise ValueError('Count mismatch; refuse partial/changing snapshot')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    manifest = dict(source=SOURCE, query=QUERY, retrieved_at=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest(), count=len(marks), bbox=BBOX,
                    event_id=325, upstream_available_at=None,
                    use='Retrospective evidence baseline only; hosted USGS-labelled copy, not independently audited against STN')
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
