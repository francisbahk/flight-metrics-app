"""
Tests for frontend/utils.py pure utility functions.
These have no Streamlit dependency and can run with plain pytest.
"""
import io
import pytest
from unittest.mock import patch

from frontend.utils import (
    apply_filters,
    build_manual_parsed,
    detect_codeshares,
    format_price,
    generate_flight_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_flight(**overrides):
    base = {
        "id": "F1",
        "airline": "UA",
        "flight_number": "UA100",
        "departure_time": "2026-03-01T08:00:00",
        "arrival_time": "2026-03-01T11:00:00",
        "price": 300.0,
        "duration_min": 180,
        "stops": 0,
        "origin": "JFK",
        "destination": "LAX",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# format_price
# ---------------------------------------------------------------------------

def test_format_price_normal():
    assert format_price(250) == "$250"


def test_format_price_zero_returns_na():
    assert format_price(0) == "N/A"


def test_format_price_none_returns_na():
    assert format_price(None) == "N/A"


def test_format_price_float_rounds():
    assert format_price(199.99) == "$200"


# ---------------------------------------------------------------------------
# detect_codeshares
# ---------------------------------------------------------------------------

def test_detect_codeshares_no_matches():
    f1 = _make_flight(id="F1", airline="UA", departure_time="2026-03-01T08:00:00", arrival_time="2026-03-01T11:00:00")
    f2 = _make_flight(id="F2", airline="DL", departure_time="2026-03-01T09:00:00", arrival_time="2026-03-01T12:00:00")
    result = detect_codeshares([f1, f2])
    assert result == {0: False, 1: False}


def test_detect_codeshares_identical_times_same_route():
    f1 = _make_flight(id="F1", airline="UA")
    f2 = _make_flight(id="F2", airline="LH")  # same times/route → codeshare
    result = detect_codeshares([f1, f2])
    assert result[0] is True
    assert result[1] is True


def test_detect_codeshares_empty():
    assert detect_codeshares([]) == {}


def test_detect_codeshares_single_flight():
    assert detect_codeshares([_make_flight()]) == {0: False}


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------

def test_apply_filters_no_filters():
    flights = [_make_flight(), _make_flight(id="F2", airline="DL")]
    assert apply_filters(flights) == flights


def test_apply_filters_by_airline():
    f1 = _make_flight(id="F1", airline="UA")
    f2 = _make_flight(id="F2", airline="DL")
    result = apply_filters([f1, f2], airlines=["UA"])
    assert result == [f1]


def test_apply_filters_by_connections():
    f_direct = _make_flight(id="F1", stops=0)
    f_one_stop = _make_flight(id="F2", stops=1)
    result = apply_filters([f_direct, f_one_stop], connections=[0])
    assert result == [f_direct]


def test_apply_filters_by_price_range():
    cheap = _make_flight(id="F1", price=100.0)
    expensive = _make_flight(id="F2", price=800.0)
    result = apply_filters([cheap, expensive], price_range=(50.0, 500.0))
    assert result == [cheap]


def test_apply_filters_by_duration_range():
    short = _make_flight(id="F1", duration_min=90)
    long_ = _make_flight(id="F2", duration_min=360)
    result = apply_filters([short, long_], duration_range=(0, 200))
    assert result == [short]


def test_apply_filters_by_origin():
    f_jfk = _make_flight(id="F1", origin="JFK")
    f_ewr = _make_flight(id="F2", origin="EWR")
    result = apply_filters([f_jfk, f_ewr], origins=["JFK"])
    assert result == [f_jfk]


def test_apply_filters_by_destination():
    f_lax = _make_flight(id="F1", destination="LAX")
    f_sfo = _make_flight(id="F2", destination="SFO")
    result = apply_filters([f_lax, f_sfo], destinations=["SFO"])
    assert result == [f_sfo]


def test_apply_filters_departure_time_range():
    # 08:00 → hour 8.0; filter keeps hours 6–10
    flight = _make_flight(departure_time="2026-03-01T08:00:00")
    excluded = _make_flight(id="F2", departure_time="2026-03-01T14:00:00")
    result = apply_filters([flight, excluded], departure_range=(6.0, 10.0))
    assert result == [flight]


def test_apply_filters_arrival_time_range():
    flight = _make_flight(arrival_time="2026-03-01T11:00:00")
    excluded = _make_flight(id="F2", arrival_time="2026-03-01T23:00:00")
    result = apply_filters([flight, excluded], arrival_range=(8.0, 14.0))
    assert result == [flight]


def test_apply_filters_empty_list_returns_empty():
    assert apply_filters([], airlines=["UA"], connections=[0]) == []


# ---------------------------------------------------------------------------
# build_manual_parsed
# ---------------------------------------------------------------------------

def test_build_manual_parsed_one_way():
    from datetime import date
    result = build_manual_parsed(["JFK"], ["LAX"], date(2026, 3, 1))
    assert result["parsed_successfully"] is True
    assert result["origins"] == ["JFK"]
    assert result["destinations"] == ["LAX"]
    assert result["departure_dates"] == ["2026-03-01"]
    assert result["return_dates"] == []
    assert result["return_date"] is None


def test_build_manual_parsed_with_return():
    from datetime import date
    result = build_manual_parsed(["JFK"], ["LAX"], date(2026, 3, 1), return_date=date(2026, 3, 8))
    assert result["return_dates"] == ["2026-03-08"]
    assert result["return_date"] == "2026-03-08"


def test_build_manual_parsed_strips_whitespace():
    from datetime import date
    result = build_manual_parsed(["  jfk  ", " EWR"], ["lax"], date(2026, 3, 1))
    assert result["origins"] == ["JFK", "EWR"]
    assert result["destinations"] == ["LAX"]


def test_build_manual_parsed_ignores_blank_codes():
    from datetime import date
    result = build_manual_parsed(["JFK", ""], ["LAX"], date(2026, 3, 1))
    assert result["origins"] == ["JFK"]


# ---------------------------------------------------------------------------
# generate_flight_csv
# ---------------------------------------------------------------------------

def _sample_flights():
    flights = [_make_flight(id=f"F{i}", price=100.0 * i or 100.0) for i in range(1, 4)]
    flights[0]["price"] = 100.0
    flights[1]["price"] = 200.0
    flights[2]["price"] = 300.0
    return flights


def test_generate_flight_csv_columns():
    flights = _sample_flights()
    csv = generate_flight_csv(flights, [flights[0], flights[1]], k=5)
    lines = csv.strip().split("\n")
    header = lines[0].split("\t")
    expected_cols = [
        "unique_id", "is_best", "rank", "name", "origin", "destination",
        "departure_time", "arrival_time", "duration", "stops", "price",
        "dis_from_origin", "dis_from_dest", "departure_dt", "departure_seconds",
        "arrival_dt", "arrival_seconds", "duration_min",
    ]
    assert header == expected_cols


def test_generate_flight_csv_row_count():
    flights = _sample_flights()
    csv = generate_flight_csv(flights, [flights[0]], k=5)
    lines = [l for l in csv.strip().split("\n") if l]
    assert len(lines) == 4  # header + 3 flights


def test_generate_flight_csv_is_best_flag():
    flights = _sample_flights()
    csv = generate_flight_csv(flights, [flights[0]], k=5)
    lines = csv.strip().split("\n")
    first_data = lines[1].split("\t")
    second_data = lines[2].split("\t")
    # is_best column is index 1
    assert first_data[1] == "1"
    assert second_data[1] == "0"


def test_generate_flight_csv_unranked_value():
    """Unranked flights get rank = ((k+1) + N) / 2"""
    flights = _sample_flights()  # N=3
    k = 5
    expected_unranked = ((k + 1) + len(flights)) / 2  # (6+3)/2 = 4.5
    csv = generate_flight_csv(flights, [flights[0]], k=k)
    lines = csv.strip().split("\n")
    # flights[1] is not selected → unranked
    second_row = lines[2].split("\t")
    rank_col = float(second_row[2])
    assert rank_col == expected_unranked


def test_generate_flight_csv_tab_in_airline_name_cleaned():
    """Tab characters in text fields must be replaced with spaces."""
    flight = _make_flight(id="F1", airline="UA\tTest")
    # Patch get_airline_name to return the tab-containing airline
    with patch("frontend.utils.get_airline_name", return_value="United\tAirlines"):
        csv = generate_flight_csv([flight], [flight], k=5)
    # No raw tab in the name column
    lines = csv.strip().split("\n")
    data_row = lines[1].split("\t")
    name_col = data_row[3]  # 'name' is column index 3
    assert "\t" not in name_col
