"""
Flight-selection / ranking section.

Renders the single-panel flight list + drag-to-rank column.
Called when st.session_state.all_flights is populated but
outbound_submitted is False (i.e. user is still picking flights).
"""
import streamlit as st
from datetime import datetime
from streamlit_sortables import sort_items

from frontend.utils import (
    apply_filters,
    format_price,
    get_airline_name,
    generate_flight_csv,
)
from frontend.styles import NEON_METRIC_CSS


# CSS for the neon filter-heading animation (injected once per render)
_FILTER_HEADING_CSS = """
<style>
    @keyframes neonGlow10s {
        0%   { box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444; border-color: #ff4444; }
        15%  { box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444; border-color: #ff6666; }
        30%  { box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444; border-color: #ff4444; }
        45%  { box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444; border-color: #ff6666; }
        60%  { box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444; border-color: #ff4444; }
        75%  { box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444; border-color: #ff6666; }
        90%  { box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444; border-color: #ff4444; }
        100% { box-shadow: none; border-color: transparent; }
    }
    @keyframes shrinkPadding {
        0%   { padding: 2px 6px; margin: 0 2px; }
        100% { padding: 0; margin: 0; }
    }
    .filter-heading-neon {
        display: inline-block;
        animation: neonGlow10s 10s ease-in-out forwards;
        padding: 4px 12px;
        border-radius: 6px;
        border: 1.5px solid #ff4444;
    }
    .metric-neon {
        display: inline-block;
        animation: neonGlow10s 10s ease-in-out forwards,
                   shrinkPadding 2s ease-in-out 10s forwards;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #ff4444;
        margin: 0 2px;
    }
</style>
<script>
    setTimeout(() => {
        const elem = document.querySelector('.filter-heading-neon');
        if (elem) {
            elem.style.animation = 'none';
            setTimeout(() => {
                elem.style.animation = 'neonGlow10s 10s ease-in-out forwards';
            }, 10);
        }
    }, 100);
</script>
"""

_PROGRESS_BAR_CSS = """
<style>
    @keyframes progressPulse {{
        0%, 83.33% {{ opacity: 0.4; }}
        83.34%      {{ opacity: 0.4; }}
        87.5%       {{ opacity: 1; }}
        91.67%      {{ opacity: 1; }}
        95.84%      {{ opacity: 0.4; }}
        100%        {{ opacity: 0.4; }}
    }}
    .persistent-progress-container {{
        position: fixed;
        top: 60px;
        left: 280px;
        right: 20px;
        z-index: 999;
        background-color: rgba(255, 255, 255, 1);
        padding: 8px 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        animation: progressPulse 60s ease-in-out infinite;
    }}
    .persistent-progress-bar {{
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }}
    .persistent-progress-fill {{
        height: 100%;
        background-color: #4CAF50;
        width: {pct}%;
        transition: width 0.3s ease;
    }}
    .persistent-progress-text {{
        margin-top: 4px;
        font-size: 12px;
        color: #666;
        text-align: center;
        font-weight: 500;
    }}
</style>
<div class="persistent-progress-container">
    <div class="persistent-progress-bar">
        <div class="persistent-progress-fill"></div>
    </div>
    <div class="persistent-progress-text">{text}</div>
</div>
"""


def render_ranking_section():
    """
    Render sidebar filters + single-panel flight list + ranking column.
    Must only be called when st.session_state.all_flights is non-empty
    and outbound_submitted is False.
    """
    total_flights = len(st.session_state.all_flights)
    rank_limit = min(20, total_flights)

    progress_percent = 1.0 if st.session_state.outbound_submitted else 0.0
    progress_text = "Submitted" if st.session_state.outbound_submitted else "0 / 1 sections submitted"

    st.markdown(
        _PROGRESS_BAR_CSS.format(pct=progress_percent * 100, text=progress_text),
        unsafe_allow_html=True
    )

    st.markdown(f"### Found {len(st.session_state.all_flights)} Flights")

    # Prompt display + Make Edits (AI searches only)
    if st.session_state.get('search_mode') != 'manual':
        if 'editing_prompt_main' not in st.session_state:
            st.session_state.editing_prompt_main = False

        if not st.session_state.editing_prompt_main:
            col_prompt, col_btn = st.columns([5, 1])
            with col_prompt:
                st.info(f"**Your prompt:** {st.session_state.get('original_prompt', '')}")
            with col_btn:
                if st.button("Make Edits", key="edit_prompt_main_btn", use_container_width=True):
                    st.session_state.editing_prompt_main = True
                    st.rerun()
        else:
            edited_prompt_main = st.text_area(
                "Edit your prompt:",
                value=st.session_state.get('original_prompt', ''),
                height=120,
                key="edited_prompt_main"
            )
            col_save1, col_cancel = st.columns([1, 1])
            with col_save1:
                if st.button("Save", key="save_prompt_main"):
                    st.session_state.original_prompt = edited_prompt_main
                    st.session_state.editing_prompt_main = False
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="cancel_prompt_main"):
                    st.session_state.editing_prompt_main = False
                    st.rerun()

    if total_flights <= 20:
        st.info(f"There are {total_flights} flights in your results — please check and rank **all {total_flights}** of them.")
    else:
        st.info("There are more than 20 flights in your results — please select and rank your **top 20**.")

    # ------------------------------------------------------------------
    # SIDEBAR FILTERS
    # ------------------------------------------------------------------
    _render_sidebar_filters()

    # Apply filters
    filtered_outbound = apply_filters(
        st.session_state.all_flights,
        airlines=st.session_state.filter_airlines,
        connections=st.session_state.filter_connections,
        price_range=st.session_state.filter_price_range,
        duration_range=st.session_state.filter_duration_range,
        departure_range=st.session_state.filter_departure_time_range,
        arrival_range=st.session_state.filter_arrival_time_range,
        origins=st.session_state.filter_origins,
        destinations=st.session_state.filter_destinations,
    )

    # ------------------------------------------------------------------
    # SINGLE PANEL LAYOUT
    # ------------------------------------------------------------------
    st.markdown('<div id="top-of-page"></div>', unsafe_allow_html=True)
    st.markdown('<div id="outbound-flights"></div>', unsafe_allow_html=True)
    st.markdown("## Outbound Flights")

    st.markdown(NEON_METRIC_CSS, unsafe_allow_html=True)

    col_flights, col_ranking = st.columns([2, 1])

    with col_flights:
        st.markdown("#### All Available Flights")

        if len(filtered_outbound) < len(st.session_state.all_flights):
            st.info(f"Filters applied: Showing {len(filtered_outbound)} of {len(st.session_state.all_flights)} flights")

        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            arrow = "" if st.session_state.sort_price_dir_single == 'asc' else ""
            if st.button(f"Sort by Price {arrow}", key="sort_price_single", use_container_width=True):
                reverse = st.session_state.sort_price_dir_single == 'desc'
                st.session_state.all_flights = sorted(
                    st.session_state.all_flights, key=lambda x: x['price'], reverse=reverse
                )
                st.session_state.sort_price_dir_single = 'desc' if st.session_state.sort_price_dir_single == 'asc' else 'asc'
                st.rerun()
        with col_sort2:
            arrow = "" if st.session_state.sort_duration_dir_single == 'asc' else ""
            if st.button(f"Sort by Duration {arrow}", key="sort_duration_single", use_container_width=True):
                reverse = st.session_state.sort_duration_dir_single == 'desc'
                st.session_state.all_flights = sorted(
                    st.session_state.all_flights, key=lambda x: x['duration_min'], reverse=reverse
                )
                st.session_state.sort_duration_dir_single = 'desc' if st.session_state.sort_duration_dir_single == 'asc' else 'asc'
                st.rerun()

        for idx, flight in enumerate(filtered_outbound):
            flight_key = f"{flight['id']}_{flight['departure_time']}"
            is_selected = any(
                f"{f['id']}_{f['departure_time']}" == flight_key
                for f in st.session_state.selected_flights
            )

            col1, col2 = st.columns([1, 5])

            with col1:
                chk_key = f"chk_single_{idx}_v{st.session_state.checkbox_version}"
                if chk_key not in st.session_state:
                    st.session_state[chk_key] = is_selected
                selected = st.checkbox(
                    "Select flight",
                    key=chk_key,
                    label_visibility="collapsed",
                    disabled=(not st.session_state[chk_key] and len(st.session_state.selected_flights) >= rank_limit)
                )
                if selected and not is_selected:
                    if len(st.session_state.selected_flights) < rank_limit:
                        st.session_state.selected_flights.append(flight)
                        st.rerun()
                elif not selected and is_selected:
                    st.session_state.selected_flights = [
                        f for f in st.session_state.selected_flights
                        if f"{f['id']}_{f['departure_time']}" != flight_key
                    ]
                    st.rerun()

            with col2:
                dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                dept_time_display = dept_dt.strftime("%I:%M %p")
                arr_time_display = arr_dt.strftime("%I:%M %p")
                dept_date_display = dept_dt.strftime("%a, %b %d")
                dh = flight['duration_min'] // 60
                dm = flight['duration_min'] % 60
                duration_display = f"{dh} hr {dm} min" if dh > 0 else f"{dm} min"
                airline_name = get_airline_name(flight['airline'])
                stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"
                neon_class = "metric-neon" if idx == 0 else ""

                st.markdown(f"""
                <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                    <span class="{neon_class}" style="font-weight: 700;">{format_price(flight['price'])}</span> &bull;
                    <span class="{neon_class}" style="font-weight: 600;">{duration_display}</span> &bull;
                    <span class="{neon_class}" style="font-weight: 500;">{stops_text}</span> &bull;
                    <span class="{neon_class}" style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    <span class="{neon_class}">{airline_name} {flight['flight_number']}</span> |
                    <span class="{neon_class}">{flight['origin']} &rarr; {flight['destination']}</span> |
                    <span class="{neon_class}">{dept_date_display}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

    with col_ranking:
        _render_ranking_column(rank_limit)


def _render_sidebar_filters():
    """Render all sidebar filter widgets for the main flight list."""
    with st.sidebar:
        st.markdown(_FILTER_HEADING_CSS, unsafe_allow_html=True)
        st.markdown('<h2><span class="filter-heading-neon">Filters</span></h2>', unsafe_allow_html=True)

        all_flights = st.session_state.all_flights
        unique_airlines = sorted(set(f['airline'] for f in all_flights))
        airline_names_map = {code: get_airline_name(code) for code in unique_airlines}
        unique_connections = sorted(set(f['stops'] for f in all_flights))

        prices = [f['price'] for f in all_flights]
        min_price, max_price = (min(prices), max(prices)) if prices else (0, 1000)
        durations = [f['duration_min'] for f in all_flights]
        min_duration, max_duration = (min(durations), max(durations)) if durations else (0, 1440)

        unique_origins = sorted(set(f['origin'] for f in all_flights))
        unique_destinations = sorted(set(f['destination'] for f in all_flights))
        show_origin_filter = len(unique_origins) > 1
        show_dest_filter = len(unique_destinations) > 1

        rc = st.session_state.filter_reset_counter

        with st.expander("Airlines", expanded=False):
            selected_airlines = []
            for code in unique_airlines:
                if st.checkbox(airline_names_map[code], key=f"airline_{code}_{rc}"):
                    selected_airlines.append(code)
            st.session_state.filter_airlines = selected_airlines if selected_airlines else None

        if show_origin_filter:
            with st.expander("Origin Airport", expanded=False):
                selected_origins = []
                for origin in unique_origins:
                    if st.checkbox(origin, key=f"origin_{origin}_{rc}"):
                        selected_origins.append(origin)
                st.session_state.filter_origins = selected_origins if selected_origins else None

        if show_dest_filter:
            with st.expander("Destination Airport", expanded=False):
                selected_destinations = []
                for dest in unique_destinations:
                    if st.checkbox(dest, key=f"dest_{dest}_{rc}"):
                        selected_destinations.append(dest)
                st.session_state.filter_destinations = selected_destinations if selected_destinations else None

        with st.expander("Connections", expanded=False):
            selected_connections = []
            for conn in unique_connections:
                label = "Direct" if conn == 0 else f"{conn} stop{'s' if conn > 1 else ''}"
                if st.checkbox(label, key=f"conn_{conn}_{rc}"):
                    selected_connections.append(conn)
            st.session_state.filter_connections = selected_connections if selected_connections else None

        if len(all_flights) <= 1:
            st.caption("Only one flight — filters not applicable.")
        else:
            with st.expander("Price Range", expanded=False):
                price_range = st.slider(
                    "Select price range",
                    min_value=float(min_price), max_value=float(max_price),
                    value=(float(min_price), float(max_price)),
                    step=10.0, format="$%.0f",
                    key=f"filter_price_slider_{rc}"
                )
                st.session_state.filter_price_range = (
                    price_range if price_range != (float(min_price), float(max_price)) else None
                )

        def hours_to_time(h):
            hours = int(h)
            mins = int((h - hours) * 60)
            return f"{hours:02d}:{mins:02d}"

        if len(all_flights) > 1:
            with st.expander("Flight Duration", expanded=False):
                duration_range = st.slider(
                    "Select duration range",
                    min_value=int(min_duration), max_value=int(max_duration),
                    value=(int(min_duration), int(max_duration)),
                    step=30, format="%d min",
                    key=f"filter_duration_slider_{rc}"
                )
                min_h, min_m = divmod(duration_range[0], 60)
                max_h, max_m = divmod(duration_range[1], 60)
                st.caption(f"{min_h}h {min_m}m - {max_h}h {max_m}m")
                st.session_state.filter_duration_range = (
                    duration_range if duration_range != (int(min_duration), int(max_duration)) else None
                )

        if len(all_flights) > 1:
            with st.expander("Departure Time", expanded=False):
                dept_range = st.slider(
                    "Select departure time range",
                    min_value=0.0, max_value=24.0, value=(0.0, 24.0),
                    step=0.5, format="%.1f",
                    key=f"filter_departure_slider_{rc}"
                )
                st.caption(f"{hours_to_time(dept_range[0])} - {hours_to_time(dept_range[1])}")
                st.session_state.filter_departure_time_range = (
                    dept_range if dept_range != (0.0, 24.0) else None
                )

            with st.expander("Arrival Time", expanded=False):
                arr_range = st.slider(
                    "Select arrival time range",
                    min_value=0.0, max_value=24.0, value=(0.0, 24.0),
                    step=0.5, format="%.1f",
                    key=f"filter_arrival_slider_{rc}"
                )
                st.caption(f"{hours_to_time(arr_range[0])} - {hours_to_time(arr_range[1])}")
                st.session_state.filter_arrival_time_range = (
                    arr_range if arr_range != (0.0, 24.0) else None
                )

        if st.button("Clear All Filters", use_container_width=True):
            st.session_state.filter_airlines = None
            st.session_state.filter_connections = None
            st.session_state.filter_price_range = None
            st.session_state.filter_duration_range = None
            st.session_state.filter_departure_time_range = None
            st.session_state.filter_arrival_time_range = None
            st.session_state.filter_origins = None
            st.session_state.filter_destinations = None
            st.session_state.filter_reset_counter += 1
            st.session_state.checkbox_version += 1


def _render_ranking_column(rank_limit: int):
    """Render the drag-to-rank column and submit button."""
    st.markdown(f"#### Your Top {rank_limit} (Drag to Rank)")
    st.markdown(f"**{len(st.session_state.selected_flights)}/{rank_limit} selected**")

    if not st.session_state.selected_flights:
        st.info("Check boxes on the left to select flights")
        return

    if len(st.session_state.selected_flights) > rank_limit:
        st.session_state.selected_flights = st.session_state.selected_flights[:rank_limit]
        st.rerun()

    draggable_items = []
    for i, flight in enumerate(st.session_state.selected_flights):
        airline_name = get_airline_name(flight['airline'])
        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
        dh = flight['duration_min'] // 60
        dm = flight['duration_min'] % 60
        duration_display = f"{dh}h {dm}m" if dh > 0 else f"{dm}m"
        stops = int(flight.get('stops', 0))
        stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
        item = (
            f"#{i+1}: {format_price(flight['price'])} \u2022 {duration_display} \u2022 {stops_text}\n"
            f"{dept_dt.strftime('%I:%M %p')} - {arr_dt.strftime('%I:%M %p')}\n"
            f"{airline_name} {flight['flight_number']}\n"
            f"{flight['origin']} \u2192 {flight['destination']} | {dept_dt.strftime('%a, %b %d')}"
        )
        draggable_items.append(item)

    sorted_items = sort_items(
        draggable_items,
        multi_containers=False,
        direction='vertical',
        key=f"single_sort_v{st.session_state.single_sort_version}_n{len(st.session_state.selected_flights)}"
    )

    if sorted_items and sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
        new_order = []
        for si in sorted_items:
            rank = int(si.split(':')[0].replace('#', '')) - 1
            if rank < len(st.session_state.selected_flights):
                new_order.append(st.session_state.selected_flights[rank])
        if len(new_order) == len(st.session_state.selected_flights):
            st.session_state.selected_flights = new_order
            st.session_state.single_sort_version += 1
            st.rerun()

    st.markdown("---")

    if len(st.session_state.selected_flights) == rank_limit and not st.session_state.outbound_submitted:
        if st.button("Submit Rankings", key="submit_single", type="primary", use_container_width=True):
            print("[DEBUG] Submit button clicked - Preparing for review")
            csv_data = generate_flight_csv(
                st.session_state.all_flights,
                st.session_state.selected_flights,
                k=rank_limit
            )
            st.session_state.csv_data_outbound = csv_data
            st.session_state.outbound_submitted = True
            st.session_state.csv_generated = True
            st.rerun()
    elif st.session_state.outbound_submitted:
        st.success("Rankings submitted")
    else:
        remaining = rank_limit - len(st.session_state.selected_flights)
        st.info(f"Select {remaining} more flight{'s' if remaining != 1 else ''}")
