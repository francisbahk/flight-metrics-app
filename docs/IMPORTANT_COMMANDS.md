# Important Commands for Data Verification

## Check Database Data

### Check a specific token's data
```bash
python3 -c "
from backend.db import SessionLocal, Search, AccessToken, CrossValidation
db = SessionLocal()

TOKEN = 'GA01'  # Change this to check different tokens

# Check token usage
token = db.query(AccessToken).filter(AccessToken.token == TOKEN).first()
print(f'Token {TOKEN}: is_used={token.is_used if token else \"NOT FOUND\"}')

# Check search data
search = db.query(Search).filter(Search.completion_token == TOKEN).order_by(Search.created_at.desc()).first()
if search:
    print(f'\\nSearch found:')
    print(f'  search_id: {search.search_id}')
    print(f'  session_id: {search.session_id}')
    print(f'  user_prompt: {search.user_prompt[:100]}...' if search.user_prompt else 'None')
    print(f'  has flights: {len(search.listen_ranked_flights_json) if search.listen_ranked_flights_json else 0}')
    print(f'  created_at: {search.created_at}')
else:
    print(f'No search found for {TOKEN}')

# Check cross-validations (for Groups B/C)
cv_count = db.query(CrossValidation).filter(CrossValidation.reviewer_token == TOKEN).count()
print(f'\\nCross-validations by {TOKEN}: {cv_count}')

db.close()
"
```

### List all pilot tokens and their status
```bash
python3 migrate_pilot_study.py --list
```

### Count all searches
```bash
python3 -c "
from backend.db import SessionLocal, Search
db = SessionLocal()
count = db.query(Search).count()
print(f'Total searches: {count}')
db.close()
"
```

### View recent searches
```bash
python3 -c "
from backend.db import SessionLocal, Search
db = SessionLocal()
searches = db.query(Search).order_by(Search.created_at.desc()).limit(10).all()
print('Recent searches:')
for s in searches:
    print(f'  {s.search_id}: {s.completion_token or \"no token\"} - {s.created_at}')
db.close()
"
```

---

## S3 Backup Commands

### Run a manual backup
```bash
python3 backup_db.py
```

### Check S3 backups (using AWS CLI)
```bash
# Install AWS CLI first: brew install awscli
# Configure: aws configure (enter your AWS credentials)

aws s3 ls s3://flight-app-db-backups/
```

### Check S3 backups (using Python - if boto3 works)
```bash
python3 -c "
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)

response = s3.list_objects_v2(Bucket='flight-app-db-backups', MaxKeys=10)
if 'Contents' in response:
    for obj in response['Contents']:
        print(f'{obj[\"Key\"]} - {obj[\"Size\"]/1024:.1f} KB')
else:
    print('No backups found')
"
```

---

## Export Session Data

### Export a token's data to CSV
```bash
python3 export_session_data.py GA01
```

### Export simplified rankings CSV
```bash
python3 -c "
from export_session_data import export_simple_rankings_csv
export_simple_rankings_csv('GA01', output_file='GA01_rankings.csv')
print('Exported to GA01_rankings.csv')
"
```

---

## Database Connection Test

```bash
python3 -c "
from backend.db import test_connection
if test_connection():
    print('Database connection: OK')
else:
    print('Database connection: FAILED')
"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Check token data | `python3 -c "..."` (see above) |
| List all tokens | `python3 migrate_pilot_study.py --list` |
| Manual backup | `python3 backup_db.py` |
| Export to CSV | `python3 export_session_data.py TOKEN` |
| Test DB connection | See above |
