"""
Completion / post-submission section.

Covers everything that renders once outbound_submitted is True:
  - Review-before-save screen
  - Database save
  - Cross-validation (Groups B/C re-rankings)
  - Completion page with CSV download
  - Countdown timer
  - "New Search" reset button
  - Error fallback when save fails
"""
import streamlit as st
from datetime import datetime
from frontend.utils import apply_filters, get_airline_name, format_price
from phases import is_phase_token


def _trigger_backup(token: str):
    """Import and call trigger_backup_on_completion from app-level helpers."""
    try:
        from app import trigger_backup_on_completion
        trigger_backup_on_completion(token)
    except Exception:
        pass


def render_completion_section():
    """
    Render the post-submission flow.  Must only be called when
    st.session_state.outbound_submitted is True.
    """
    ready_to_save = st.session_state.get('csv_generated')

    # ------------------------------------------------------------------
    # 1. REVIEW SCREEN (before saving to DB)
    # ------------------------------------------------------------------
    if (ready_to_save
            and not st.session_state.get('search_id')
            and not st.session_state.get('db_save_error')
            and not st.session_state.get('review_confirmed')):

        st.markdown("---")
        st.markdown("## Review Your Results")
        st.markdown("Before finalizing your submission, please review your prompt and ensure it accurately and fully captures your preferences over relevant considerations.")

        st.markdown("### Your Current Prompt:")
        st.info(st.session_state.get('original_prompt', ''))

        with st.expander("Review Questions", expanded=True):
            st.markdown("""
            - Did you mention **all** the factors that influenced how you ranked flights (e.g., price, duration, stops, timing, location, airline)?
            - Are your trade-offs and priorities clear? For example, if you chose a pricier nonstop over a cheaper connection, does your prompt reflect that?
            - Is there anything you considered while ranking that isn't captured in your prompt?
            """)

        st.markdown("### Edit Your Prompt (Optional)")
        edited_prompt = st.text_area(
            "If you'd like to revise your prompt, edit it here:",
            value=st.session_state.get('original_prompt', ''),
            height=150,
            key="edited_prompt"
        )

        if st.button("Confirm & Submit Final Results", type="primary", use_container_width=True):
            if edited_prompt != st.session_state.get('original_prompt', ''):
                st.session_state.original_prompt = edited_prompt
                try:
                    from backend.db import save_prompt_attempt
                    save_prompt_attempt(
                        st.session_state.get('prolific_id', 'anonymous'),
                        edited_prompt,
                        is_edit=True,
                        edit_source="confirmation",
                    )
                except Exception:
                    pass
            st.session_state.review_confirmed = True

            if st.session_state.get('token'):
                from backend.db import save_participant_progress
                save_participant_progress(
                    prolific_id=st.session_state.token,
                    session_id=st.session_state.session_id,
                    prompt=st.session_state.get('original_prompt', ''),
                )
                print(f"[PILOT] Saved review confirmation for {st.session_state.token}")

            st.rerun()

        st.stop()

    # ------------------------------------------------------------------
    # 2. SAVE TO DATABASE (after review confirmed)
    # ------------------------------------------------------------------
    if (ready_to_save
            and not st.session_state.get('search_id')
            and not st.session_state.get('db_save_error')
            and st.session_state.get('review_confirmed')):

        st.info("Saving your rankings...")
        try:
            if not st.session_state.selected_flights:
                st.session_state.db_save_error = "No selected flights available"
            else:
                from backend.db import save_rankings
                success = save_rankings(
                    prolific_id=st.session_state.token,
                    selected_flights=st.session_state.selected_flights,
                    prompt=st.session_state.get('original_prompt', ''),
                )
                if success:
                    st.session_state.search_id = st.session_state.token
                    st.session_state.countdown_started = True
                    print(f"[DB] Rankings saved for {st.session_state.token}")
                    st.rerun()
                else:
                    st.session_state.db_save_error = "save_rankings returned False"
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            st.session_state.db_save_error = str(e)

    # ------------------------------------------------------------------
    # 3. SUCCESS + DOWNLOAD + CROSS-VALIDATION
    # ------------------------------------------------------------------
    if st.session_state.get('search_id'):
        st.success("All rankings submitted successfully!")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 3a. CROSS-VALIDATION
        # ------------------------------------------------------------------
        if not st.session_state.get('cross_validation_completed'):
            _render_cross_validation(None)

        # ------------------------------------------------------------------
        # 3b. COMPLETION PAGE
        # ------------------------------------------------------------------
        if (st.session_state.get('cross_validation_completed')
                and not st.session_state.get('completion_page_dismissed')):
            _render_completion_page()

        # ------------------------------------------------------------------
        # 3c. COUNTDOWN TIMER
        # ------------------------------------------------------------------
        if (st.session_state.get('cross_validation_completed')
                and st.session_state.get('survey_completed')
                and st.session_state.get('completion_page_dismissed')
                and st.session_state.get('countdown_started')
                and not st.session_state.get('countdown_completed')):
            _render_countdown()

    # ------------------------------------------------------------------
    # 4. DB ERROR / MISSING SEARCH ID FALLBACK
    # ------------------------------------------------------------------
    elif st.session_state.get('db_save_error'):
        st.warning("Rankings submitted but database save failed")
        st.error(f"Database error: {st.session_state.db_save_error}")
    else:
        st.error("Something went wrong saving your rankings. Please refresh the page and try again, or contact listen.cornell@gmail.com.")



# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_cross_validation(_unused):
    """Render cross-validation: participant re-ranks flights from a seed prompt."""
    import json as _json

    # Assign seed prompt once per session
    if not st.session_state.get('cv_seed_prompt_id'):
        from backend.db import get_next_seed_prompt
        seed = get_next_seed_prompt(st.session_state.get('token', ''))
        if not seed:
            # No seed prompts loaded — skip CV gracefully
            st.session_state.cross_validation_completed = True
            st.session_state.all_reranks_completed = True
            if not st.session_state.get('backup_triggered'):
                st.session_state.backup_triggered = True
                _trigger_backup(st.session_state.get('token', 'unknown'))
            return
        st.session_state.cv_seed_prompt_id = seed['id']
        st.session_state.cv_seed_prompt_data = seed

    seed = st.session_state.cv_seed_prompt_data
    flights = _json.loads(seed['flights_json']) if isinstance(seed['flights_json'], str) else seed['flights_json']
    rank_limit = min(20, len(flights))

    for key, default in [
        ('cross_val_selected_flights', []),
        ('cv_checkbox_version', 0),
        ('cv_sort_version', 0),
        ('cv_sort_price_dir', 'asc'),
        ('cv_sort_duration_dir', 'asc'),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("---")
    st.markdown("## One More Task")
    st.markdown("Before you finish, please rank the flights shown for another participant's search prompt.")
    st.info(f"**Their search prompt:**\n\n> {seed['prompt_text']}")

    if len(flights) <= 20:
        st.info(f"There are {len(flights)} flights — please check and rank **all {len(flights)}** of them.")
    else:
        st.info("There are more than 20 flights — please select and rank the **top 20** that best match this prompt.")

    st.markdown("---")
    _render_cv_flight_list(flights, rank_limit)
    return

    # Dead code below — kept for reference only
    # Dim everything above this section
    st.markdown("""
    <style>
        .stApp > header,
        .main > div:not(:has(.cross-validation-section)) {
            opacity: 0.1;
            pointer-events: none;
        }
        [data-testid="stSidebar"] {
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        .cross-validation-section {
            background: white;
            position: relative;
            z-index: 1000;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="cross-validation-section">', unsafe_allow_html=True)

    if rerank_targets:
        completed_reranks = st.session_state.get('completed_reranks', [])
        remaining_targets = [t for t in rerank_targets if t not in completed_reranks]
        current_rerank_num = len(completed_reranks) + 1
        total_reranks = len(rerank_targets)

        if not remaining_targets:
            st.session_state.all_reranks_completed = True
            st.session_state.cross_validation_completed = True
            if not st.session_state.get('backup_triggered'):
                st.session_state.backup_triggered = True
                _trigger_backup(st.session_state.get('token', 'unknown'))
            st.rerun()

        st.progress(len(completed_reranks) / total_reranks)
        st.markdown(f"### Re-ranking {current_rerank_num} of {total_reranks}")
        st.markdown("*Help us by ranking flights for another user's search*")

        current_target = remaining_targets[0]
        if ('cross_val_data' not in st.session_state
                or st.session_state.get('current_cv_target') != current_target):
            st.session_state.cross_val_data = get_assigned_search_for_validation(
                st.session_state.get('token'), current_target
            )
            st.session_state.current_cv_target = current_target
            st.session_state.cross_val_selected_flights = []
            st.session_state.cv_checkbox_version = st.session_state.get('cv_checkbox_version', 0) + 1
    else:
        st.markdown("### Help Validate Another Search")
        st.markdown("*Before completing your session, please help us by ranking flights for another user's search*")
        if 'cross_val_data' not in st.session_state:
            st.session_state.cross_val_data = get_previous_search_for_validation(
                st.session_state.session_id,
                st.session_state.get('token')
            )

    if st.session_state.get('cross_val_data') is not None:
        if not st.session_state.cross_val_data:
            st.info("No previous searches available for validation. You're one of the first users!")
            st.session_state.cross_validation_completed = True
            if not st.session_state.get('backup_triggered'):
                st.session_state.backup_triggered = True
                _trigger_backup(st.session_state.get('token', 'unknown'))
            st.rerun()
        else:
            _render_cv_flight_selection(rerank_targets)

    st.markdown('</div>', unsafe_allow_html=True)


def _render_cv_flight_list(flights: list, rank_limit: int):
    """Flight list + ranking column for the new seed-prompt cross-validation."""
    from streamlit_sortables import sort_items

    col_flights, col_ranking = st.columns([2, 1])

    with col_flights:
        st.markdown("#### Flights")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            arrow = "↑" if st.session_state.cv_sort_price_dir == 'asc' else "↓"
            if st.button(f"Sort by Price {arrow}", key="cv2_sort_price", use_container_width=True):
                flights[:] = sorted(flights, key=lambda x: x['price'],
                                    reverse=(st.session_state.cv_sort_price_dir == 'desc'))
                st.session_state.cv_sort_price_dir = 'desc' if st.session_state.cv_sort_price_dir == 'asc' else 'asc'
                st.session_state.cv_seed_prompt_data = {
                    **st.session_state.cv_seed_prompt_data,
                    'flights_json': __import__('json').dumps(flights),
                }
                st.rerun()
        with col_s2:
            arrow = "↑" if st.session_state.cv_sort_duration_dir == 'asc' else "↓"
            if st.button(f"Sort by Duration {arrow}", key="cv2_sort_dur", use_container_width=True):
                flights[:] = sorted(flights, key=lambda x: x['duration_min'],
                                    reverse=(st.session_state.cv_sort_duration_dir == 'desc'))
                st.session_state.cv_sort_duration_dir = 'desc' if st.session_state.cv_sort_duration_dir == 'asc' else 'asc'
                st.session_state.cv_seed_prompt_data = {
                    **st.session_state.cv_seed_prompt_data,
                    'flights_json': __import__('json').dumps(flights),
                }
                st.rerun()

        for idx, flight in enumerate(flights):
            flight_key = f"{flight['id']}_{flight['departure_time']}"
            is_selected = any(
                f"{f['id']}_{f['departure_time']}" == flight_key
                for f in st.session_state.cross_val_selected_flights
            )

            c1, c2 = st.columns([1, 5])
            with c1:
                chk_key = f"cv2_chk_{idx}_v{st.session_state.cv_checkbox_version}"
                if chk_key not in st.session_state:
                    st.session_state[chk_key] = is_selected
                selected = st.checkbox(
                    "Select",
                    key=chk_key,
                    label_visibility="collapsed",
                    disabled=(not st.session_state[chk_key] and
                              len(st.session_state.cross_val_selected_flights) >= rank_limit),
                )
                if selected and not is_selected:
                    if len(st.session_state.cross_val_selected_flights) < rank_limit:
                        st.session_state.cross_val_selected_flights.append(flight)
                        st.rerun()
                elif not selected and is_selected:
                    st.session_state.cross_val_selected_flights = [
                        f for f in st.session_state.cross_val_selected_flights
                        if f"{f['id']}_{f['departure_time']}" != flight_key
                    ]
                    st.rerun()

            with c2:
                dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                arr_dt  = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                dh, dm  = divmod(flight['duration_min'], 60)
                dur_str = f"{dh} hr {dm} min" if dh else f"{dm} min"
                airline_name = get_airline_name(flight['airline'])
                stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"

                cabin = flight.get('cabin') or ''
                cabin_display = cabin.replace('_', ' ').title() if cabin else 'Economy'
                bags = flight.get('checked_bags', 0) or 0
                bags_display = f"{bags} bag{'s' if bags != 1 else ''} included"
                layovers = flight.get('layover_airports') or []
                layover_display = f"Via {', '.join(layovers)}" if layovers else 'Nonstop'
                extras = ' | '.join([cabin_display, bags_display, layover_display])

                st.markdown(f"""
                <div style="line-height:1.4;padding:0.4rem 0;border-bottom:1px solid #eee;">
                <div style="font-size:1.1em;margin-bottom:0.2rem;">
                    <b>{format_price(flight['price'])}</b> &bull;
                    <span>{dur_str}</span> &bull;
                    <span>{stops_text}</span> &bull;
                    <span>{dept_dt.strftime('%I:%M %p')} - {arr_dt.strftime('%I:%M %p')}</span>
                </div>
                <div style="font-size:0.9em;color:#666;">
                    {airline_name} {flight['flight_number']} |
                    {flight['origin']} &rarr; {flight['destination']} |
                    {dept_dt.strftime('%a, %b %d')}
                </div>
                {f'<div style="font-size:0.85em;color:#888;">{extras}</div>' if extras else ''}
                </div>
                """, unsafe_allow_html=True)

    with col_ranking:
        st.markdown(f"#### Top {rank_limit} (Drag to Rank)")
        st.markdown(f"**{len(st.session_state.cross_val_selected_flights)}/{rank_limit} selected**")

        if not st.session_state.cross_val_selected_flights:
            st.info("Check boxes on the left to select flights")
            return

        if len(st.session_state.cross_val_selected_flights) > rank_limit:
            st.session_state.cross_val_selected_flights = st.session_state.cross_val_selected_flights[:rank_limit]
            st.rerun()

        draggable_items = []
        for i, f in enumerate(st.session_state.cross_val_selected_flights):
            dept_dt = datetime.fromisoformat(f['departure_time'].replace('Z', '+00:00'))
            arr_dt  = datetime.fromisoformat(f['arrival_time'].replace('Z', '+00:00'))
            dh, dm  = divmod(f['duration_min'], 60)
            dur_str = f"{dh}h {dm}m" if dh else f"{dm}m"
            stops   = "Nonstop" if f['stops'] == 0 else f"{f['stops']} stop{'s' if f['stops']>1 else ''}"
            draggable_items.append(
                f"#{i+1}: {format_price(f['price'])} • {dur_str} • {stops}\n"
                f"{dept_dt.strftime('%I:%M %p')} - {arr_dt.strftime('%I:%M %p')}\n"
                f"{get_airline_name(f['airline'])} {f['flight_number']}\n"
                f"{f['origin']} → {f['destination']} | {dept_dt.strftime('%a, %b %d')}"
            )

        sorted_items = sort_items(
            draggable_items,
            multi_containers=False,
            direction='vertical',
            key=f"cv2_sort_v{st.session_state.cv_sort_version}_n{len(st.session_state.cross_val_selected_flights)}"
        )

        if sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
            new_order = []
            for si in sorted_items:
                rank = int(si.split(':')[0].replace('#', '')) - 1
                if rank < len(st.session_state.cross_val_selected_flights):
                    new_order.append(st.session_state.cross_val_selected_flights[rank])
            if len(new_order) == len(st.session_state.cross_val_selected_flights):
                st.session_state.cross_val_selected_flights = new_order
                st.session_state.cv_sort_version += 1
                st.rerun()

        st.markdown("---")
        if len(st.session_state.cross_val_selected_flights) == rank_limit:
            if st.button("Submit Cross-Validation Rankings", key="cv2_submit",
                         type="primary", use_container_width=True):
                from backend.db import save_cv_rankings
                success = save_cv_rankings(
                    reviewer_prolific_id=st.session_state.get('token', ''),
                    seed_prompt_id=st.session_state.cv_seed_prompt_id,
                    selected_flights=st.session_state.cross_val_selected_flights,
                )
                if success:
                    st.session_state.cross_validation_completed = True
                    st.session_state.all_reranks_completed = True
                    if not st.session_state.get('backup_triggered'):
                        st.session_state.backup_triggered = True
                        _trigger_backup(st.session_state.get('token', 'unknown'))
                    st.rerun()
                else:
                    st.error("Failed to save. Please try again.")
        else:
            remaining = rank_limit - len(st.session_state.cross_val_selected_flights)
            st.info(f"Select {remaining} more flight{'s' if remaining != 1 else ''}")


def _render_cv_flight_selection(rerank_targets):
    """Render the flight-selection UI inside cross-validation."""
    from streamlit_sortables import sort_items

    cross_val = st.session_state.cross_val_data
    cv_flights = cross_val['flights']

    # Init CV session state
    for key, default in [
        ('cross_val_selected_flights', []),
        ('cv_filter_reset_counter', 0),
        ('cv_sort_version', 0),
        ('cv_checkbox_version', 0),
        ('cv_sort_price_dir', 'asc'),
        ('cv_sort_duration_dir', 'asc'),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.info(f"**Another user's search prompt:**\n\n> {cross_val['prompt']}")
    st.markdown("**Task:** Review their flights and select your top 5 that best match their needs, then drag to rank them.")
    st.markdown("---")

    st.markdown(f"### Found {len(cv_flights)} Flights")
    st.markdown("**Select your top 5 flights and drag to rank them**")

    # Sidebar filters for CV
    unique_airlines_cv = sorted(set(f['airline'] for f in cv_flights))
    airline_names_map_cv = {code: get_airline_name(code) for code in unique_airlines_cv}
    unique_connections_cv = sorted(set(f['stops'] for f in cv_flights))
    prices_cv = [f['price'] for f in cv_flights]
    min_price_cv, max_price_cv = (min(prices_cv), max(prices_cv)) if prices_cv else (0, 1000)
    durations_cv = [f['duration_min'] for f in cv_flights]
    min_duration_cv, max_duration_cv = (min(durations_cv), max(durations_cv)) if durations_cv else (0, 1440)

    def hours_to_time(h):
        hours = int(h)
        mins = int((h - hours) * 60)
        return f"{hours:02d}:{mins:02d}"

    with st.sidebar:
        st.markdown("---")
        st.markdown('<h2><span class="filter-heading-neon">Filters</span></h2>', unsafe_allow_html=True)

        with st.expander("Airlines", expanded=False):
            selected_airlines_cv = []
            for code in unique_airlines_cv:
                if st.checkbox(airline_names_map_cv[code], key=f"cv_airline_{code}_{st.session_state.cv_filter_reset_counter}"):
                    selected_airlines_cv.append(code)

        with st.expander("Connections", expanded=False):
            selected_connections_cv = []
            for conn in unique_connections_cv:
                label = "Direct" if conn == 0 else f"{conn} stop{'s' if conn > 1 else ''}"
                if st.checkbox(label, key=f"cv_conn_{conn}_{st.session_state.cv_filter_reset_counter}"):
                    selected_connections_cv.append(conn)

        with st.expander("Price Range", expanded=False):
            price_range_cv = st.slider(
                "Select price range",
                min_value=float(min_price_cv), max_value=float(max_price_cv),
                value=(float(min_price_cv), float(max_price_cv)),
                step=10.0, format="$%.0f",
                key=f"cv_price_{st.session_state.cv_filter_reset_counter}"
            )

        with st.expander("Flight Duration", expanded=False):
            duration_range_cv = st.slider(
                "Select duration range",
                min_value=int(min_duration_cv), max_value=int(max_duration_cv),
                value=(int(min_duration_cv), int(max_duration_cv)),
                step=30, format="%d min",
                key=f"cv_duration_{st.session_state.cv_filter_reset_counter}"
            )
            min_h, min_m = divmod(duration_range_cv[0], 60)
            max_h, max_m = divmod(duration_range_cv[1], 60)
            st.caption(f"{min_h}h {min_m}m - {max_h}h {max_m}m")

        with st.expander("Departure Time", expanded=False):
            dept_range_cv = st.slider(
                "Select departure time range",
                min_value=0.0, max_value=24.0, value=(0.0, 24.0),
                step=0.5, format="%.1f",
                key=f"cv_dept_{st.session_state.cv_filter_reset_counter}"
            )
            st.caption(f"{hours_to_time(dept_range_cv[0])} - {hours_to_time(dept_range_cv[1])}")

        with st.expander("Arrival Time", expanded=False):
            arr_range_cv = st.slider(
                "Select arrival time range",
                min_value=0.0, max_value=24.0, value=(0.0, 24.0),
                step=0.5, format="%.1f",
                key=f"cv_arr_{st.session_state.cv_filter_reset_counter}"
            )
            st.caption(f"{hours_to_time(arr_range_cv[0])} - {hours_to_time(arr_range_cv[1])}")

        if st.button("Clear All Filters", use_container_width=True, key="clear_cv"):
            st.session_state.cv_filter_reset_counter += 1
            st.rerun()

    filtered_cv = apply_filters(
        cv_flights,
        airlines=selected_airlines_cv if selected_airlines_cv else None,
        connections=selected_connections_cv if selected_connections_cv else None,
        price_range=price_range_cv if price_range_cv != (float(min_price_cv), float(max_price_cv)) else None,
        duration_range=duration_range_cv if duration_range_cv != (int(min_duration_cv), int(max_duration_cv)) else None,
        departure_range=dept_range_cv if dept_range_cv != (0.0, 24.0) else None,
        arrival_range=arr_range_cv if arr_range_cv != (0.0, 24.0) else None,
    )

    col_flights_cv, col_ranking_cv = st.columns([2, 1])

    with col_flights_cv:
        st.markdown("#### All Flights")
        if len(filtered_cv) < len(cv_flights):
            st.info(f"Filters applied: Showing {len(filtered_cv)} of {len(cv_flights)} flights")

        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            arrow = "" if st.session_state.cv_sort_price_dir == 'asc' else ""
            if st.button(f"Sort by Price {arrow}", key="cv_sort_price", use_container_width=True):
                reverse = st.session_state.cv_sort_price_dir == 'desc'
                filtered_cv[:] = sorted(filtered_cv, key=lambda x: x['price'], reverse=reverse)
                st.session_state.cv_sort_price_dir = 'desc' if st.session_state.cv_sort_price_dir == 'asc' else 'asc'
                st.rerun()
        with col_sort2:
            arrow = "" if st.session_state.cv_sort_duration_dir == 'asc' else ""
            if st.button(f"Sort by Duration {arrow}", key="cv_sort_dur", use_container_width=True):
                reverse = st.session_state.cv_sort_duration_dir == 'desc'
                filtered_cv[:] = sorted(filtered_cv, key=lambda x: x['duration_min'], reverse=reverse)
                st.session_state.cv_sort_duration_dir = 'desc' if st.session_state.cv_sort_duration_dir == 'asc' else 'asc'
                st.rerun()

        for idx, flight in enumerate(filtered_cv):
            flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
            is_selected = any(
                f"{f['id']}_{f['departure_time']}" == flight_unique_key
                for f in st.session_state.cross_val_selected_flights
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                selected = st.checkbox(
                    "Select flight",
                    value=is_selected,
                    key=f"cv_chk_{idx}_v{st.session_state.cv_checkbox_version}",
                    label_visibility="collapsed",
                    disabled=(not is_selected and len(st.session_state.cross_val_selected_flights) >= 5)
                )
                if selected and not is_selected:
                    if len(st.session_state.cross_val_selected_flights) < 5:
                        st.session_state.cross_val_selected_flights.append(flight)
                elif not selected and is_selected:
                    st.session_state.cross_val_selected_flights = [
                        f for f in st.session_state.cross_val_selected_flights
                        if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                    ]

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

                st.markdown(f"""
                <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                    <span style="font-weight: 700;">{format_price(flight['price'])}</span> •
                    <span style="font-weight: 600;">{duration_display}</span> •
                    <span style="font-weight: 500;">{stops_text}</span> •
                    <span style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    <span>{airline_name} {flight['flight_number']}</span> |
                    <span>{flight['origin']} &rarr; {flight['destination']}</span> |
                    <span>{dept_date_display}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

    with col_ranking_cv:
        st.markdown("#### Top 5 (Drag to Rank)")
        st.markdown(f"**{len(st.session_state.cross_val_selected_flights)}/5 selected**")

        if st.session_state.cross_val_selected_flights:
            if len(st.session_state.cross_val_selected_flights) > 5:
                st.session_state.cross_val_selected_flights = st.session_state.cross_val_selected_flights[:5]
                st.rerun()

            draggable_items = []
            for i, flight in enumerate(st.session_state.cross_val_selected_flights):
                airline_name = get_airline_name(flight['airline'])
                dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                dh = flight['duration_min'] // 60
                dm = flight['duration_min'] % 60
                duration_display = f"{dh}h {dm}m" if dh > 0 else f"{dm}m"
                stops = int(flight.get('stops', 0))
                stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
                item = (
                    f"#{i+1}: {format_price(flight['price'])} • {duration_display} • {stops_text}\n"
                    f"{dept_dt.strftime('%I:%M %p')} - {arr_dt.strftime('%I:%M %p')}\n"
                    f"{airline_name} {flight['flight_number']}\n"
                    f"{flight['origin']} -> {flight['destination']} | {dept_dt.strftime('%a, %b %d')}"
                )
                draggable_items.append(item)

            sorted_items = sort_items(
                draggable_items,
                multi_containers=False,
                direction='vertical',
                key=f"cv_sort_v{st.session_state.cv_sort_version}_n{len(st.session_state.cross_val_selected_flights)}"
            )

            if sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
                new_order = []
                for si in sorted_items:
                    rank = int(si.split(':')[0].replace('#', '')) - 1
                    if rank < len(st.session_state.cross_val_selected_flights):
                        new_order.append(st.session_state.cross_val_selected_flights[rank])
                if len(new_order) == len(st.session_state.cross_val_selected_flights):
                    st.session_state.cross_val_selected_flights = new_order
                    st.session_state.cv_sort_version += 1
                    st.rerun()

            st.markdown("---")
            cols = st.columns(5)
            for i, flight in enumerate(st.session_state.cross_val_selected_flights):
                with cols[i]:
                    if st.button("X", key=f"cv_remove_{i}_{flight['id']}", help=f"Remove #{i+1}"):
                        fuk = f"{flight['id']}_{flight['departure_time']}"
                        st.session_state.cross_val_selected_flights = [
                            f for f in st.session_state.cross_val_selected_flights
                            if f"{f['id']}_{f['departure_time']}" != fuk
                        ]
                        st.session_state.cv_checkbox_version += 1
                        st.rerun()

            if len(st.session_state.cross_val_selected_flights) == 5:
                remaining = len([
                    t for t in st.session_state.get('rerank_targets', [])
                    if t not in st.session_state.get('completed_reranks', [])
                ]) - 1
                button_label = (
                    f"Submit Rankings ({remaining} more to go)" if remaining > 0
                    else "Submit Final Rankings"
                )

                if st.button(button_label, key="cv_submit", type="primary", use_container_width=True):
                    _submit_cross_validation(cross_val, rerank_targets)
        else:
            st.caption("No flights selected yet")


def _submit_cross_validation(cross_val, rerank_targets):
    """Save cross-validation results and advance state."""
    from backend.db import save_cross_validation, save_session_progress

    completed_reranks = st.session_state.get('completed_reranks', [])
    rerank_sequence = len(completed_reranks) + 1 if rerank_targets else None
    source_token = cross_val.get('source_token')

    success = save_cross_validation(
        reviewer_session_id=st.session_state.session_id,
        reviewed_session_id=cross_val['session_id'],
        reviewed_search_id=cross_val['search_id'],
        reviewed_prompt=cross_val['prompt'],
        reviewed_flights=cross_val['flights'],
        selected_flights=st.session_state.cross_val_selected_flights,
        reviewer_token=st.session_state.get('token'),
        rerank_sequence=rerank_sequence,
        source_token=source_token
    )

    if success:
        if rerank_targets:
            current_target = st.session_state.get('current_cv_target')
            if current_target:
                if 'completed_reranks' not in st.session_state:
                    st.session_state.completed_reranks = []
                st.session_state.completed_reranks.append(current_target)
                st.session_state.current_rerank_index = len(st.session_state.completed_reranks)

                save_session_progress(st.session_state.get('token'), {
                    'current_rerank_index': st.session_state.current_rerank_index,
                    'completed_reranks_json': st.session_state.completed_reranks,
                })
                print(f"[PILOT] Completed re-ranking for {current_target} "
                      f"({st.session_state.current_rerank_index}/{len(st.session_state.rerank_targets)})")

                del st.session_state.cross_val_data
                del st.session_state.current_cv_target
                st.session_state.cross_val_selected_flights = []

                if len(st.session_state.completed_reranks) >= len(st.session_state.rerank_targets):
                    st.session_state.all_reranks_completed = True
                    st.session_state.cross_validation_completed = True
                    save_session_progress(st.session_state.get('token'), {
                        'all_reranks_completed': 1,
                        'current_phase': 'complete',
                    })
                    if not st.session_state.get('backup_triggered'):
                        st.session_state.backup_triggered = True
                        _trigger_backup(st.session_state.get('token', 'unknown'))
                    if not st.session_state.get('token_marked_used'):
                        from backend.db import mark_token_used
                        t = st.session_state.get('token')
                        if t and t.upper() not in ["DEMO", "DATA"] and not is_phase_token(t):
                            mark_token_used(t)
                            st.session_state.token_marked_used = True
                            print(f"[PILOT] Token {t} marked as used (all re-rankings complete)")
                    st.success("All re-rankings complete! Thank you!")
                else:
                    st.success("Submitted! Moving to next re-ranking...")
        else:
            st.session_state.cross_validation_completed = True
            if not st.session_state.get('backup_triggered'):
                st.session_state.backup_triggered = True
                _trigger_backup(st.session_state.get('token', 'unknown'))
            st.success("Thank you for helping validate!")

        st.rerun()
    else:
        st.error("Failed to save. Please try again.")


_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C1D07BSK"


def _render_completion_page():
    """Show the post-study survey, then redirect to Prolific on submit."""
    import streamlit.components.v1 as components

    # If survey already submitted, show redirect (handles page re-renders)
    if st.session_state.get('post_survey_completed'):
        st.markdown("---")
        st.markdown("# All Done!")
        st.markdown("---")
        st.link_button(
            "Click here to complete your submission on Prolific",
            _COMPLETION_URL,
            type="primary",
            use_container_width=True,
        )
        components.html(
            f'<script>window.top.location.href = "{_COMPLETION_URL}";</script>',
            height=0,
        )
        return

    st.markdown("---")
    st.markdown("# Thank You for Participating!")
    st.markdown("Your rankings have been saved successfully.")
    st.markdown("---")
    st.markdown("### One Last Question")
    st.markdown(
        "Was anything unclear or could anything be improved about the study? "
        "Please share any feedback below."
    )

    with st.form("post_survey_form"):
        feedback = st.text_area(
            label="Your feedback",
            placeholder="Share any thoughts, confusion, or suggestions here...",
            height=150,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Submit & Finish", type="primary", use_container_width=True
        )
    if submitted:
        _save_post_survey_feedback(feedback.strip())
        st.session_state.post_survey_completed = True
        st.session_state.completion_page_dismissed = True
        st.rerun()


def _save_post_survey_feedback(feedback: str):
    """Persist post-survey feedback to the database."""
    try:
        from backend.db import save_post_survey
        prolific_id = st.session_state.get('token', '')
        save_post_survey(prolific_id=prolific_id, feedback=feedback)
        print(f"[POST_SURVEY] Saved feedback for {prolific_id}")
    except Exception as e:
        print(f"[POST_SURVEY] Failed to save feedback: {e}")


def _render_countdown():
    """Render the 120-second countdown then mark token used."""
    import time
    st.info("Please note your Search ID above. This session will end in:")

    countdown_placeholder = st.empty()
    progress_placeholder = st.empty()

    for remaining in range(120, 0, -1):
        countdown_placeholder.markdown(f"### {remaining} seconds")
        progress_placeholder.progress((120 - remaining) / 120)
        time.sleep(1)

    countdown_placeholder.markdown("### 0 seconds")
    progress_placeholder.progress(1.0)

    t = st.session_state.token
    if t and t.upper() not in ["DEMO", "DATA"] and not is_phase_token(t):
        from backend.db import mark_token_used
        mark_token_used(t)
        st.session_state.countdown_completed = True
        st.warning("Session ended. Thank you for your participation!")
        st.info("This link can no longer be used. To participate again, please request a new token from the research team.")
        st.stop()
    else:
        st.session_state.countdown_completed = True
        st.success("Thank you! You can continue using this link to submit more rankings.")
        tok = st.session_state.token or ''
        if tok.upper() == "DEMO":
            st.info("This is a DEMO token - you can use it as many times as you want!")
        elif tok.upper() == "DATA":
            st.info("This is a DATA collection token - you can use it for unlimited submissions!")
        elif is_phase_token(tok):
            st.info("Thank you for your participation! You may close this window now.")
        _reset_demo_data_state()
        st.info("Refresh the page to start a new search!")
        st.stop()


def _reset_demo_data_state():
    """Reset submission state for DEMO/DATA tokens to allow re-use."""
    st.session_state.outbound_submitted = False
    st.session_state.csv_generated = False
    st.session_state.countdown_started = False
    st.session_state.countdown_completed = False
    st.session_state.selected_flights = []
    st.session_state.all_flights = []
    st.session_state.review_confirmed = False
    st.session_state.cross_validation_completed = False
    st.session_state.cross_val_data = None
    st.session_state.cross_val_selected = []
    st.session_state.survey_completed = False
    st.session_state.completion_page_dismissed = False


def _reset_for_new_search():
    """Reset all submission state when user clicks 'New Search'."""
    st.session_state.all_flights = []
    st.session_state.selected_flights = []
    st.session_state.csv_generated = False
    st.session_state.outbound_submitted = False
    st.session_state.csv_data_outbound = None
    st.session_state.parsed_params = None
    st.session_state.review_confirmed = False
    st.session_state.search_id = None
    st.session_state.db_save_error = None
    st.session_state.survey_completed = False
    st.session_state.completion_page_dismissed = False
    st.session_state.cross_validation_completed = False
    st.session_state.cross_val_data = None
    st.session_state.cross_val_selected_flights = []
    st.session_state.cv_filter_reset_counter = 0
    st.session_state.cv_sort_version = 0
    st.session_state.cv_checkbox_version = 0
    st.session_state.cv_sort_price_dir = 'asc'
    st.session_state.cv_sort_duration_dir = 'asc'
    if 'flight_prompt_input' in st.session_state:
        del st.session_state.flight_prompt_input
