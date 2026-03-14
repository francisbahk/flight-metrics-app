"""
Export comprehensive session data to CSV.
Each row represents one flight shown to the participant.
"""

import csv
import json
from backend.db import (
    SessionLocal, Search, CrossValidation, FlightShown, UserRanking
)


def export_session_to_csv(token: str, output_file: str = None) -> str:
    """
    Export all data for a participant token to CSV.
    Includes their flights + rankings, and any cross-validation rankings.

    Returns path to generated CSV file.
    """
    if not output_file:
        output_file = f"{token}_data.csv"

    db = SessionLocal()

    try:
        searches = db.query(Search).filter(
            (Search.session_id == token) | (Search.completion_token == token)
        ).all()

        if not searches:
            print(f"No data found for token {token}")
            return None

        all_rows = []

        for search in searches:
            search_id = search.search_id
            prompt = search.user_prompt

            all_flights = search.listen_ranked_flights_json or search.amadeus_flights_json or []
            if not all_flights:
                print(f"No flight data for search_id={search_id}")
                continue

            # User rankings: flight_id -> rank
            user_rankings = {}
            for r in db.query(UserRanking).filter_by(search_id=search_id).all():
                fs = db.query(FlightShown).filter_by(id=r.flight_id).first()
                if fs:
                    user_rankings[fs.flight_data.get('id')] = r.user_rank

            # Cross-validation records (up to 4)
            cross_vals = db.query(CrossValidation).filter_by(
                reviewer_session_id=search.session_id
            ).order_by(CrossValidation.rerank_sequence.asc()).all()

            cv_data = []
            for cv_idx in range(4):
                if cv_idx < len(cross_vals):
                    cv = cross_vals[cv_idx]
                    cv_rankings = {}
                    if cv.selected_flights_data:
                        for rank, flight in enumerate(cv.selected_flights_data, 1):
                            cv_rankings[flight.get('id')] = rank
                    reviewed_search = db.query(Search).filter(
                        Search.session_id == cv.reviewed_session_id
                    ).first()
                    cv_flights = []
                    if reviewed_search:
                        cv_flights = reviewed_search.listen_ranked_flights_json or reviewed_search.amadeus_flights_json or []
                    cv_data.append({
                        'source_token': cv.source_token or cv.reviewed_session_id,
                        'rerank_sequence': cv.rerank_sequence,
                        'prompt': cv.reviewed_prompt,
                        'flights': cv_flights,
                        'rankings': cv_rankings,
                    })
                else:
                    cv_data.append({'source_token': None, 'rerank_sequence': None, 'prompt': None, 'flights': [], 'rankings': {}})

            max_cv_flights = max((len(cv['flights']) for cv in cv_data), default=0)
            num_rows = max(len(all_flights), max_cv_flights)

            for idx in range(num_rows):
                row = {
                    'prompt': prompt if idx == 0 else '',
                    'id': token if idx == 0 else '',
                }

                if idx < len(all_flights):
                    f = all_flights[idx]
                    fid = f.get('id')
                    row.update({
                        'unique_id': fid, 'rank': user_rankings.get(fid, ''),
                        'airline': f.get('airline', ''), 'origin': f.get('origin', ''),
                        'destination': f.get('destination', ''),
                        'departure_time': f.get('departure_time', ''),
                        'arrival_time': f.get('arrival_time', ''),
                        'stops': f.get('stops', ''), 'price': f.get('price', ''),
                        'duration_min': f.get('duration_min', ''),
                    })
                else:
                    row.update({'unique_id': '', 'rank': '', 'airline': '', 'origin': '',
                                'destination': '', 'departure_time': '', 'arrival_time': '',
                                'stops': '', 'price': '', 'duration_min': ''})

                for n in range(1, 5):
                    cv = cv_data[n - 1]
                    p = f'cv{n}_'
                    row[f'{p}source_token'] = cv['source_token'] if idx == 0 else ''
                    row[f'{p}rerank_sequence'] = cv['rerank_sequence'] if idx == 0 else ''
                    row[f'{p}prompt'] = cv['prompt'] if idx == 0 else ''
                    if idx < len(cv['flights']):
                        cf = cv['flights'][idx]
                        cfid = cf.get('id')
                        row.update({
                            f'{p}unique_id': cfid, f'{p}rank': cv['rankings'].get(cfid, ''),
                            f'{p}airline': cf.get('airline', ''), f'{p}origin': cf.get('origin', ''),
                            f'{p}destination': cf.get('destination', ''),
                            f'{p}departure_time': cf.get('departure_time', ''),
                            f'{p}arrival_time': cf.get('arrival_time', ''),
                            f'{p}stops': cf.get('stops', ''), f'{p}price': cf.get('price', ''),
                            f'{p}duration_min': cf.get('duration_min', ''),
                        })
                    else:
                        row.update({f'{p}unique_id': '', f'{p}rank': '', f'{p}airline': '',
                                    f'{p}origin': '', f'{p}destination': '',
                                    f'{p}departure_time': '', f'{p}arrival_time': '',
                                    f'{p}stops': '', f'{p}price': '', f'{p}duration_min': ''})

                all_rows.append(row)

        if not all_rows:
            print(f"No data to export for token {token}")
            return None

        fieldnames = [
            'prompt', 'id',
            'unique_id', 'rank', 'airline', 'origin', 'destination',
            'departure_time', 'arrival_time', 'stops', 'price', 'duration_min',
        ]
        for n in range(1, 5):
            p = f'cv{n}_'
            fieldnames += [f'{p}source_token', f'{p}rerank_sequence', f'{p}prompt',
                           f'{p}unique_id', f'{p}rank', f'{p}airline', f'{p}origin',
                           f'{p}destination', f'{p}departure_time', f'{p}arrival_time',
                           f'{p}stops', f'{p}price', f'{p}duration_min']

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"✓ Exported {len(all_rows)} rows to {output_file}")
        return output_file

    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python export_session_data.py <token> [output_file.csv]")
        sys.exit(1)
    token = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    export_session_to_csv(token, output_file)
