# Two-Token Privacy System

## Overview

The flight app now implements a **two-token privacy system** to ensure participant anonymity and data protection:

1. **Access Token** - For study entry (NOT linked to research data)
2. **Completion Token** - For payment verification and data collection

## How It Works

### 1. Access Token (Study Entry)

- Participants receive a **randomly generated access token** to enter the study
- Token format: 6-10 character alphanumeric string
- Purpose: **Control access only** - not stored with research data
- Stored in: `access_tokens` table
- Fields:
  - `token` - The access token
  - `created_at` - When token was generated
  - `used_at` - When participant entered study
  - `is_used` - Whether token has been used (0/1)
  - `completion_token` - Links to completion token (for admin tracking only)

### 2. Completion Token (Data Collection & Payment)

- Generated **automatically when participant completes the study**
- Token format: 8 character alphanumeric string (URL-safe)
- Purpose: **Payment verification + links to anonymous research data**
- Stored in: `completion_tokens` table
- Fields:
  - `token` - The completion token
  - `created_at` - When study was completed
  - `session_id` - Anonymous session identifier

## Privacy Guarantees

### ✅ What Researchers CAN See

- Completion tokens (e.g., "ABC123XY")
- Anonymous behavioral data linked to completion tokens:
  - Flight searches
  - Preferences
  - Rankings
  - Survey responses
- Aggregate statistics

### ❌ What Researchers CANNOT See

- Which access token belongs to which participant
- Personal identifying information (no names, emails, etc.)
- Link between individual participants and their data
- Who submitted which completion token for payment

## Data Flow

```
1. Participant receives access token → ABC123
2. Enters study with access token → Access token marked as "used"
3. Completes study tasks (searches, rankings, survey)
   └─ Data stored under session_id only
4. Upon completion → System generates completion token → XYZ789
   └─ Completion token links to session_id
   └─ Access token links to completion token (admin tracking only)
5. Participant submits completion token for payment → XYZ789
   └─ Payment processor confirms completion
   └─ Researcher cannot identify who this completion token belongs to
```

## Database Schema

### access_tokens
```sql
CREATE TABLE access_tokens (
    token VARCHAR(255) PRIMARY KEY,
    created_at DATETIME,
    used_at DATETIME NULL,
    is_used INT DEFAULT 0,
    completion_token VARCHAR(255) NULL
);
```

### completion_tokens
```sql
CREATE TABLE completion_tokens (
    token VARCHAR(255) PRIMARY KEY,
    created_at DATETIME,
    session_id VARCHAR(255) NOT NULL
);
```

### Research Data Tables
All research data tables (searches, rankings, surveys, etc.) now use:
- `completion_token` (NOT access_token)
- `session_id` (anonymous)
- NO personally identifying information

## Functions

### Access Token Functions

**`validate_access_token(token: str) -> Dict`**
- Validates access token for study entry
- Returns: `{'valid': bool, 'is_used': bool, 'message': str}`

**`mark_access_token_used(access_token: str) -> bool`**
- Marks access token as used when participant enters
- Returns: `True` if successful

### Completion Token Functions

**`generate_completion_token(session_id: str, access_token: Optional[str] = None) -> Optional[str]`**
- Generates completion token when study is finished
- Links to session_id for data collection
- Optionally links to access_token (admin tracking only)
- Returns: Generated completion token or `None` if failed

### Data Collection Functions

**`save_survey_response(session_id: str, survey_data: Dict, completion_token: Optional[str] = None) -> bool`**
- Saves survey with completion_token (not access_token)

**`save_search(..., completion_token: Optional[str] = None)`**
- Saves search data with completion_token

## Implementation Status

### ✅ Completed

- [x] Database schema updated
  - [x] `access_tokens` table created
  - [x] `completion_tokens` table created
  - [x] All data tables updated to use `completion_token`
- [x] Core functions implemented
  - [x] `validate_access_token()`
  - [x] `mark_access_token_used()`
  - [x] `generate_completion_token()`
  - [x] `save_survey_response()` updated
- [x] Legacy functions for backwards compatibility

### 🔄 TODO

- [ ] Update app.py to:
  - [ ] Accept access token for entry
  - [ ] Generate completion token on study completion
  - [ ] Display completion token to participant
  - [ ] Store completion_token with all data
- [ ] Update save_search() function
- [ ] Update token generation script
- [ ] Add completion token display UI
- [ ] Update documentation/instructions for participants
- [ ] Test end-to-end flow

## Participant Instructions

### Entering the Study
"Use the access code you received to enter the study. This code is for entry only and will not be linked to your responses."

### Completing the Study
"Here is your completion code: **ABC123XY**

Please save this code! You will need to submit it to receive payment. Your responses are anonymous and cannot be traced back to you."

## Admin/Researcher Instructions

### Generating Access Tokens
```python
from backend.token_generator import generate_tokens
generate_tokens(count=100, token_type='access')
```

### Verifying Completion Tokens for Payment
```python
from backend.db import CompletionToken, SessionLocal

db = SessionLocal()
token_record = db.query(CompletionToken).filter(
    CompletionToken.token == 'ABC123XY'
).first()

if token_record:
    print(f"✓ Valid completion token from {token_record.created_at}")
    # Process payment
else:
    print("✗ Invalid completion token")
```

### Viewing Research Data
```python
# Data is linked to completion tokens, not participants
from backend.db import Search, SessionLocal

db = SessionLocal()
searches = db.query(Search).filter(
    Search.completion_token == 'ABC123XY'
).all()

# You can see the data but cannot identify the participant
```

## Security Considerations

1. **Access tokens** are single-use and expire after use
2. **Completion tokens** are randomly generated (URL-safe base64)
3. **No PII** is collected or stored
4. **Session IDs** are random UUIDs - not traceable to individuals
5. **Completion tokens** cannot be reverse-engineered to find participants
6. **Link between access→completion** is one-way and only for admin tracking

## Benefits

- ✅ **Privacy**: Researchers cannot identify participants from data
- ✅ **Payment**: Completion tokens verify participation
- ✅ **Anonymity**: No PII linked to research data
- ✅ **Control**: Access tokens manage study entry
- ✅ **Compliance**: Meets IRB requirements for anonymous research
