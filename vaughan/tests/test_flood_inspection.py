"""Safety and data-contract tests for the experimental evidence baseline."""
import copy
import math
import pytest
from vaughan.flood.inspection import screen, distance


def doc():
    return {'type': 'FeatureCollection', 'features': [
        {'geometry': {'type': 'Point', 'coordinates': [-82, 26.5]},
         'properties': {'event_id': 325, 'hwm_id': 1, 'hwm_quality_id': None}}]}


def asset(lon=-82, lat=26.5):
    return {'asset_id': 'a', 'longitude': lon, 'latitude': lat}


def test_near_far_outside_and_empty_are_distinct():
    assert screen([asset()], doc())[0]['status'] == 'review_near_event_evidence'
    assert screen([asset(-81.71)], doc())[0]['status'] == 'unknown_no_nearby_evidence'
    assert screen([asset(-80)], doc())[0]['status'] == 'outside_pilot'
    assert screen([asset()], {'type': 'FeatureCollection', 'features': []})[0]['status'] == 'insufficient_evidence'


def test_radius_and_distance_units():
    assert distance((-82, 26.5), (-82, 26.501)) == pytest.approx(111.2, abs=.1)
    assert screen([asset(lat=26.501)], doc(), 100)[0]['status'] == 'unknown_no_nearby_evidence'
    assert screen([asset(lat=26.501)], doc(), 125)[0]['status'] == 'review_near_event_evidence'
    for r in (0, -1, 501, math.nan):
        with pytest.raises(ValueError): screen([asset()], doc(), r)


def test_wrong_event_duplicate_and_truncated_fail_closed():
    d = doc()
    d['features'][0]['properties']['event_id'] = 1
    with pytest.raises(ValueError): screen([asset()], d)
    d = doc()
    d['features'].append(copy.deepcopy(d['features'][0]))
    with pytest.raises(ValueError): screen([asset()], d)
    d = doc()
    d['exceededTransferLimit'] = True
    with pytest.raises(ValueError): screen([asset()], d)


def test_bad_coordinates_ids_and_crs_rejected():
    for a in (asset(math.nan), asset(lat=91), dict(asset(), asset_id='')):
        with pytest.raises(ValueError): screen([a], doc())
    with pytest.raises(ValueError): screen([asset(), asset()], doc())
    d = doc()
    d['crs'] = {'properties': {'name': 'EPSG:3857'}}
    with pytest.raises(ValueError): screen([asset()], d)
