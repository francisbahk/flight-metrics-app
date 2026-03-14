"""
Pure utility functions shared across frontend pages.
These have no Streamlit rendering side-effects (except get_airline_name
which reads st.session_state for cached airline names).
"""
import io
from datetime import datetime

import pandas as pd
import streamlit as st


def format_price(price):
    """Format price for display, showing 'N/A' for 0 or None."""
    if price is None or price == 0:
        return "N/A"
    return f"${price:.0f}"


def get_airline_name(code):
    """Convert airline IATA code to full name, using session-state cache."""
    if 'airline_names' in st.session_state:
        if code in st.session_state.airline_names:
            return st.session_state.airline_names[code]
    return code  # Return code as fallback


def build_manual_parsed(origins, destinations, departure_date, return_date=None):
    """
    Build the parsed dict structure that parse_flight_prompt_with_llm returns,
    from manual form inputs.
    """
    origin_codes = [c.strip().upper() for c in origins if c.strip()]
    dest_codes = [c.strip().upper() for c in destinations if c.strip()]
    dep_dates = [departure_date.strftime("%Y-%m-%d")]
    ret_dates = [return_date.strftime("%Y-%m-%d")] if return_date else []

    return {
        "parsed_successfully": True,
        "origins": origin_codes,
        "destinations": dest_codes,
        "departure_dates": dep_dates,
        "return_dates": ret_dates,
        "return_date": ret_dates[0] if ret_dates else None,
        "preferences": {
            "prefer_direct": False,
            "prefer_cheap": False,
            "prefer_fast": False,
            "avoid_early_departures": False,
            "min_connection_time": 0,
            "max_layover_time": 10000,
            "preferred_airlines": [],
            "avoid_airports": [],
            "fly_america_act": False,
        },
        "constraints": {
            "latest_arrival": None,
            "earliest_departure": None,
        },
        "original_prompt": f"Manual search: {', '.join(origin_codes)} to {', '.join(dest_codes)} on {dep_dates[0]}",
    }


def detect_codeshares(flights):
    """
    Detect which flights are codeshares based on identical departure_time,
    arrival_time, origin, and destination.

    Returns dict mapping flight index -> is_codeshare boolean.
    """
    codeshare_map = {}
    for i, flight in enumerate(flights):
        is_codeshare = False
        for j, other in enumerate(flights):
            if i == j:
                continue
            if (flight['departure_time'] == other['departure_time'] and
                    flight['arrival_time'] == other['arrival_time'] and
                    flight['origin'] == other['origin'] and
                    flight['destination'] == other['destination']):
                is_codeshare = True
                break
        codeshare_map[i] = is_codeshare
    return codeshare_map


def apply_filters(flights, airlines=None, connections=None, price_range=None,
                  duration_range=None, departure_range=None, arrival_range=None,
                  origins=None, destinations=None):
    """Filter flights based on user-selected criteria."""
    filtered = flights

    if airlines and len(airlines) > 0:
        filtered = [f for f in filtered if f['airline'] in airlines]

    if connections is not None and len(connections) > 0:
        filtered = [f for f in filtered if f['stops'] in connections]

    if origins and len(origins) > 0:
        filtered = [f for f in filtered if f['origin'] in origins]

    if destinations and len(destinations) > 0:
        filtered = [f for f in filtered if f['destination'] in destinations]

    if price_range:
        min_price, max_price = price_range
        filtered = [f for f in filtered if min_price <= f['price'] <= max_price]

    if duration_range:
        min_dur, max_dur = duration_range
        filtered = [f for f in filtered if min_dur <= f['duration_min'] <= max_dur]

    if departure_range:
        min_hour, max_hour = departure_range
        filtered_by_dept = []
        for f in filtered:
            dept_dt = datetime.fromisoformat(f['departure_time'].replace('Z', '+00:00'))
            hour = dept_dt.hour + dept_dt.minute / 60.0
            if min_hour <= hour <= max_hour:
                filtered_by_dept.append(f)
        filtered = filtered_by_dept

    if arrival_range:
        min_hour, max_hour = arrival_range
        filtered_by_arr = []
        for f in filtered:
            arr_dt = datetime.fromisoformat(f['arrival_time'].replace('Z', '+00:00'))
            hour = arr_dt.hour + arr_dt.minute / 60.0
            if min_hour <= hour <= max_hour:
                filtered_by_arr.append(f)
        filtered = filtered_by_arr

    return filtered


def generate_flight_csv(all_flights, selected_flights, k=5):
    """
    Generate tab-separated CSV with flight data and rankings.

    Args:
        all_flights: List of all flight dicts shown to user.
        selected_flights: List of flight dicts user selected (in ranked order).
        k: Number of top flights selected (default 5).

    Returns:
        CSV string (tab-separated).
    """
    reference_date = datetime.now()
    N = len(all_flights)
    unranked_value = ((k + 1) + N) / 2

    def get_flight_key(f):
        return f"{f['id']}_{f.get('departure_time', '')}_{f.get('price', 0)}"

    selected_keys = {get_flight_key(f): (idx + 1) for idx, f in enumerate(selected_flights)}

    def clean_text(text):
        if isinstance(text, str):
            return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        return text

    csv_rows = []
    for idx, flight in enumerate(all_flights):
        flight_key = get_flight_key(flight)
        is_best = flight_key in selected_keys
        rank = selected_keys.get(flight_key, unranked_value)
        unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"
        name = get_airline_name(flight['airline'])

        try:
            dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
            arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
            dept_time_str = dept_dt.strftime("%I:%M %p on %a, %b %d")
            arr_time_str = arr_dt.strftime("%I:%M %p on %a, %b %d")
            dept_seconds = (dept_dt.date() - reference_date.date()).days * 86400 + \
                           (dept_dt.hour * 3600 + dept_dt.minute * 60 + dept_dt.second)
            arr_seconds = (arr_dt.date() - reference_date.date()).days * 86400 + \
                          (arr_dt.hour * 3600 + arr_dt.minute * 60 + arr_dt.second)
            dept_dt_formatted = f"1900-{dept_dt.strftime('%m-%d %H:%M:%S')}"
            arr_dt_formatted = f"1900-{arr_dt.strftime('%m-%d %H:%M:%S')}"
        except Exception:
            dept_time_str = flight['departure_time']
            arr_time_str = flight['arrival_time']
            dept_seconds = 0
            arr_seconds = 0
            dept_dt_formatted = ""
            arr_dt_formatted = ""

        row = {
            'unique_id': clean_text(unique_id),
            'is_best': 1 if is_best else 0,
            'rank': rank,
            'name': clean_text(name),
            'origin': clean_text(flight['origin']),
            'destination': clean_text(flight['destination']),
            'departure_time': clean_text(dept_time_str),
            'arrival_time': clean_text(arr_time_str),
            'duration': clean_text(f"{flight['duration_min']//60} hr {flight['duration_min']%60} min"),
            'stops': float(flight['stops']),
            'price': float(flight['price']),
            'dis_from_origin': 0.0,
            'dis_from_dest': 0.0,
            'departure_dt': clean_text(dept_dt_formatted),
            'departure_seconds': dept_seconds,
            'arrival_dt': clean_text(arr_dt_formatted),
            'arrival_seconds': arr_seconds,
            'duration_min': flight['duration_min'],
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    columns_order = [
        'unique_id', 'is_best', 'rank', 'name', 'origin', 'destination',
        'departure_time', 'arrival_time', 'duration', 'stops', 'price',
        'dis_from_origin', 'dis_from_dest', 'departure_dt', 'departure_seconds',
        'arrival_dt', 'arrival_seconds', 'duration_min',
    ]
    df = df[columns_order]
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, sep='\t')
    return csv_buffer.getvalue()
