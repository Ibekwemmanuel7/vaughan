"""Retrospective Ian HWM proximity baseline, not an inundation classifier.

python -m vaughan.flood.inspection --evidence data/ian/hwms.geojson \
    --assets examples/flood/demo_assets.csv --out artifacts/ian-screen.json
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

BBOX = (-82.3, 26.4, -81.7, 26.8)
SOURCE = 'https://services.arcgis.com/3wFbqsFPLeKqOlIK/ArcGIS/rest/services/High_Water_Marks_Ian/FeatureServer/0'
QUERY = SOURCE + '/query?where=1%3D1&geometry=-82.3%2C26.4%2C-81.7%2C26.8&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&outSR=4326&f=geojson'


def point(lon, lat):
    lon, lat = float(lon), float(lat)
    if not math.isfinite(lon) or not math.isfinite(lat) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError('Invalid WGS84 coordinates')
    return lon, lat


def inside(p):
    return BBOX[0] <= p[0] <= BBOX[2] and BBOX[1] <= p[1] <= BBOX[3]


def distance(a, b):
    """Spherical distance in metres; adequate for the exploratory 100 m rule."""
    x, y = map(math.radians, a)
    u, v = map(math.radians, b)
    h = math.sin((v-y)/2)**2 + math.cos(y)*math.cos(v)*math.sin((u-x)/2)**2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1, h)))


def evidence(document):
    if document.get('type') != 'FeatureCollection' or document.get('exceededTransferLimit'):
        raise ValueError('Expected complete GeoJSON FeatureCollection')
    # ArcGIS outSR=4326 GeoJSON is required; reject an explicit alternate CRS.
    if 'crs' in document and document['crs']['properties']['name'] not in ('EPSG:4326', 'urn:ogc:def:crs:OGC:1.3:CRS84'):
        raise ValueError('Evidence must use WGS84 longitude/latitude')
    marks, seen = [], set()
    for f in document['features']:
        p = f['properties']
        if p['event_id'] != 325:
            raise ValueError('Expected Ian event 325 only')
        if f['geometry']['type'] != 'Point':
            raise ValueError('Expected HWM points')
        xy = point(*f['geometry']['coordinates'])
        if not inside(xy):
            raise ValueError('Evidence outside frozen pilot bounds')
        key = p['hwm_id']
        if key in seen:
            raise ValueError('Duplicate HWM identifier')
        seen.add(key)
        # Preserve quality codes without guessing their meaning or converting to probabilities.
        marks.append((key, xy, p.get('hwm_quality_id'), p.get('flag_date')))
    return marks


def screen(assets, document, radius_m=100):
    if not math.isfinite(radius_m) or not 0 < radius_m <= 500:
        raise ValueError('Exploratory radius must be in (0, 500] metres')
    marks = evidence(document)
    rows, ids = [], set()
    for a in assets:
        key = a['asset_id'].strip()
        if not key or key in ids:
            raise ValueError('Asset IDs must be nonempty and unique')
        ids.add(key)
        xy = point(a['longitude'], a['latitude'])
        nearest = min(((distance(xy, m[1]), m) for m in marks), default=None)
        state = ('outside_pilot' if not inside(xy) else
                 'insufficient_evidence' if nearest is None else
                 'review_near_event_evidence' if nearest[0] <= radius_m else
                 'unknown_no_nearby_evidence')
        rows.append(dict(asset_id=key, status=state,
                         nearest_hwm_id=nearest[1][0] if nearest and inside(xy) else None,
                         distance_m=round(nearest[0], 2) if nearest and inside(xy) else None,
                         source_quality_code=nearest[1][2] if nearest and inside(xy) else None,
                         flag_date=nearest[1][3] if nearest and inside(xy) else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--radius-m', type=float, default=100)
    args = parser.parse_args()
    raw = args.evidence.read_bytes()
    with args.assets.open(newline='', encoding='utf-8') as stream:
        rows = screen(list(csv.DictReader(stream)), json.loads(raw), args.radius_m)
    result = dict(schema_version=1, producer='Vaughan', method='ian-hwm-proximity-v1',
                  method_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  mode='retrospective_only', event_id=325, bbox=BBOX, crs='EPSG:4326',
                  radius_m=args.radius_m, processed_at=datetime.now(timezone.utc).isoformat(),
                  source=SOURCE, source_sha256=hashlib.sha256(raw).hexdigest(),
                  assets_sha256=hashlib.sha256(args.assets.read_bytes()).hexdigest(),
                  upstream_available_at=None, observation_interval=None,
                  event_interval=['2022-09-28', '2022-09-30'],
                  uncertainty='Point proximity only; no extent, depth, damage or safe-negative inference. Quality codes retained, not interpreted.',
                  review_status='unreviewed', assets=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
