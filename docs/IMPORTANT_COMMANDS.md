# View Session Data by Token

## Check a token's search data
```bash
python3 -c "
from backend.db import SessionLocal, Search
db = SessionLocal()
TOKEN = 'GA01'  # <-- Change this

s = db.query(Search).filter(Search.completion_token == TOKEN).first()
if s:
    print(f'Token: {TOKEN}')
    print(f'Search ID: {s.search_id}')
    print(f'Session ID: {s.session_id}')
    print(f'Prompt: {s.user_prompt}')
    print(f'Flights: {len(s.listen_ranked_flights_json) if s.listen_ranked_flights_json else 0}')
    print(f'Created: {s.created_at}')
else:
    print(f'No search found for {TOKEN}')
db.close()
"
```

## Check a token's cross-validations (Groups B/C)
```bash
python3 -c "
from backend.db import SessionLocal, CrossValidation
db = SessionLocal()
TOKEN = 'GB01'  # <-- Change this

cvs = db.query(CrossValidation).filter(CrossValidation.reviewer_token == TOKEN).all()
print(f'Cross-validations by {TOKEN}: {len(cvs)}')
for cv in cvs:
    print(f'  #{cv.rerank_sequence}: reviewed {cv.source_token}')
db.close()
"
```

## Check token usage status
```bash
python3 -c "
from backend.db import SessionLocal, AccessToken
db = SessionLocal()
TOKEN = 'GA01'  # <-- Change this

t = db.query(AccessToken).filter(AccessToken.token == TOKEN).first()
if t:
    print(f'{TOKEN}: {\"USED\" if t.is_used else \"AVAILABLE\"}')
else:
    print(f'{TOKEN}: NOT FOUND')
db.close()
"
```

## List all pilot tokens status
```bash
python3 migrate_pilot_study.py --list
```

## Export a token's data to CSV
```bash
python3 export_session_data.py GA01
```
