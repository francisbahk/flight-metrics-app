# 📚 Algorithm Upload Guide

## 🎯 Overview

The app allows you to **upload custom ranking algorithms** to compare different flight sorting strategies. Your algorithm will be added to the dropdown menu alongside built-in algorithms like "Cheapest", "LISTEN-U", etc.

**Location in App:** Scroll to the bottom → "📤 Upload Custom Algorithm" section

**Code Location:** [streamlit_app.py:436-570](streamlit_app.py#L436-L570)

---

## ✅ Required Format

Your algorithm **MUST** follow this exact format:

### Function Signature
```python
def rank_flights(flights, preferences):
    """
    Your algorithm description here.

    Args:
        flights: List of flight dictionaries
        preferences: Dictionary with user preferences

    Returns:
        List of flights sorted by your ranking (best first)
    """
    # Your ranking logic here
    return sorted(flights, key=lambda x: ...)
```

### Requirements:
1. ✅ Function **must** be named `rank_flights`
2. ✅ Must take **exactly 2 parameters**: `flights` and `preferences`
3. ✅ Must **return a list** of the same flight dictionaries
4. ✅ Return list should be **sorted** (best flights first)

---

## 📦 Input: What You Receive

### 1. `flights` Parameter

A **list of dictionaries**, where each dictionary represents one flight:

```python
[
    {
        'id': '1',
        'price': 299.99,
        'currency': 'USD',
        'duration_min': 360,          # 6 hours in minutes
        'stops': 0,                   # 0 = direct, 1 = one stop, etc.
        'departure_time': '2024-11-15T08:00:00',
        'arrival_time': '2024-11-15T14:00:00',
        'airline': 'AA',              # Airline code
        'flight_number': '1234',
        'origin': 'JFK',              # Origin airport code
        'destination': 'LAX'          # Destination airport code
    },
    {
        'id': '2',
        'price': 450.00,
        'duration_min': 240,          # 4 hours (faster but more expensive)
        'stops': 1,
        ...
    },
    # ... more flights
]
```

**Available Fields:**
- `id` (str) - Unique identifier
- `price` (float) - Total price in currency
- `currency` (str) - Usually "USD"
- `duration_min` (int) - Total flight duration in minutes
- `stops` (int) - Number of stops (0 = direct)
- `departure_time` (str) - ISO datetime format
- `arrival_time` (str) - ISO datetime format
- `airline` (str) - 2-letter airline code (e.g., "AA", "UA", "DL")
- `flight_number` (str) - Flight number
- `origin` (str) - 3-letter airport code
- `destination` (str) - 3-letter airport code

### 2. `preferences` Parameter

A **dictionary** with user preferences extracted from their natural language query:

```python
{
    'original_query': 'I want to fly from NYC to LA cheap and fast',
    'prefer_cheap': True,      # User mentioned "cheap", "budget", etc.
    'prefer_fast': True,       # User mentioned "fast", "quick", etc.
    'prefer_nonstop': False,   # User mentioned "direct", "nonstop", etc.
    'prefer_comfort': False,   # User mentioned "comfortable", etc.
    'max_stops': None          # Optional: max number of stops
}
```

**You can use these to adapt your ranking!**

---

## 📤 Output: What You Return

**Return a list of the SAME flight dictionaries, sorted by your preference.**

```python
# Example: Sort by price (cheapest first)
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])
```

**Important:**
- ✅ Return the **same dictionaries** (don't modify them)
- ✅ Put **best flights first** (index 0 = best, index -1 = worst)
- ✅ Return **all flights** (don't filter them out)

---

## 💡 Example Algorithms

### Example 1: Simple Price Ranking
```python
def rank_flights(flights, preferences):
    """Sort by price (cheapest first)."""
    return sorted(flights, key=lambda x: x['price'])
```

**Saves as:** `cheapest.py`

---

### Example 2: Direct Flights First, Then Price
```python
def rank_flights(flights, preferences):
    """
    Prioritize direct flights, then sort by price.
    Two-tier sorting: stops first, then price.
    """
    return sorted(flights, key=lambda x: (x['stops'], x['price']))
```

**Saves as:** `direct_then_cheap.py`

---

### Example 3: Multi-Factor Scoring
```python
def rank_flights(flights, preferences):
    """
    Balanced score considering price, duration, and stops.
    Lower score = better flight.
    """
    def calculate_score(flight):
        # Normalize price (assume $500 average)
        price_score = flight['price'] / 500

        # Normalize duration (assume 5 hours average)
        duration_score = flight['duration_min'] / 300

        # Penalty for stops
        stop_penalty = flight['stops'] * 0.5

        # Combined score (lower = better)
        return price_score + duration_score + stop_penalty

    return sorted(flights, key=calculate_score)
```

**Saves as:** `balanced.py`

---

### Example 4: Preference-Aware Algorithm
```python
def rank_flights(flights, preferences):
    """
    Adapts ranking based on user's stated preferences.
    """
    # If user wants cheap flights
    if preferences.get('prefer_cheap'):
        return sorted(flights, key=lambda x: x['price'])

    # If user wants fast flights
    elif preferences.get('prefer_fast'):
        return sorted(flights, key=lambda x: x['duration_min'])

    # If user wants direct flights
    elif preferences.get('prefer_nonstop'):
        return sorted(flights, key=lambda x: (x['stops'], x['price']))

    # Default: balanced approach
    else:
        def score(flight):
            return (flight['price'] / 500 +
                    flight['duration_min'] / 300 +
                    flight['stops'] * 0.5)
        return sorted(flights, key=score)
```

**Saves as:** `adaptive.py`

---

### Example 5: Advanced Weighted Scoring
```python
def rank_flights(flights, preferences):
    """
    Sophisticated scoring with configurable weights.
    """
    # Configure weights based on preferences
    if preferences.get('prefer_cheap'):
        price_weight = 0.7
        duration_weight = 0.2
        stop_weight = 0.1
    elif preferences.get('prefer_fast'):
        price_weight = 0.2
        duration_weight = 0.7
        stop_weight = 0.1
    else:
        price_weight = 0.4
        duration_weight = 0.3
        stop_weight = 0.3

    def weighted_score(flight):
        # Normalize each factor to 0-1 range
        price_norm = min(flight['price'] / 1000, 1.0)
        duration_norm = min(flight['duration_min'] / 600, 1.0)
        stop_norm = min(flight['stops'] / 2, 1.0)

        # Calculate weighted sum
        score = (price_weight * price_norm +
                 duration_weight * duration_norm +
                 stop_weight * stop_norm)

        return score

    return sorted(flights, key=weighted_score)
```

**Saves as:** `weighted.py`

---

## 🚀 How to Upload

### Step 1: Create Your Algorithm File

Create a new file (e.g., `my_algorithm.py`) with your `rank_flights` function:

```python
def rank_flights(flights, preferences):
    # Your custom logic here
    return sorted(flights, key=lambda x: x['price'])
```

### Step 2: Upload in the App

1. Scroll to bottom of the app
2. Find **"📤 Upload Custom Algorithm"** section
3. Click **"Browse files"** and select your `.py` file
4. Enter a **unique name** (e.g., "My Custom Ranker")
5. Click **"➕ Add Algorithm"**

### Step 3: Use Your Algorithm

1. Scroll back up to **"Step 2: Choose Two Algorithms"**
2. Select your algorithm from the dropdown (it will be at the bottom)
3. Choose another algorithm to compare against
4. Click **"🚀 Search Flights & Show Interleaved Rankings"**

Your custom algorithm will now rank flights alongside the other algorithm!

---

## 🧪 Testing Your Algorithm

### Quick Local Test

Before uploading, test your algorithm:

```python
# test_algorithm.py

# Copy your rank_flights function here
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])

# Create test data
test_flights = [
    {'id': '1', 'price': 500, 'duration_min': 300, 'stops': 1},
    {'id': '2', 'price': 300, 'duration_min': 400, 'stops': 0},
    {'id': '3', 'price': 400, 'duration_min': 250, 'stops': 2},
]

test_preferences = {
    'prefer_cheap': True,
    'prefer_fast': False,
}

# Run your algorithm
result = rank_flights(test_flights, test_preferences)

# Print results
print("Ranked flights:")
for i, flight in enumerate(result, 1):
    print(f"{i}. Flight {flight['id']} - ${flight['price']}")

# Expected output for price-based ranking:
# 1. Flight 2 - $300
# 2. Flight 3 - $400
# 3. Flight 1 - $500
```

Run it:
```bash
python test_algorithm.py
```

---

## ❌ Common Errors

### Error 1: "File must contain a function named 'rank_flights'"

**Problem:** Function is named something else

**Fix:**
```python
# ❌ Wrong
def my_ranker(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])

# ✅ Correct
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])
```

---

### Error 2: "rank_flights must take exactly 2 parameters"

**Problem:** Wrong number of parameters

**Fix:**
```python
# ❌ Wrong (too many parameters)
def rank_flights(flights, preferences, max_results):
    return sorted(flights, key=lambda x: x['price'])

# ❌ Wrong (too few parameters)
def rank_flights(flights):
    return sorted(flights, key=lambda x: x['price'])

# ✅ Correct
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])
```

---

### Error 3: KeyError when accessing flight fields

**Problem:** Trying to access a field that might not exist

**Fix:**
```python
# ❌ Wrong (crashes if 'price' is missing)
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])

# ✅ Correct (provides default if missing)
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x.get('price', float('inf')))
```

**Use `.get()` with defaults:**
```python
flight.get('price', float('inf'))      # Default to infinity
flight.get('stops', 0)                 # Default to 0
flight.get('duration_min', 999999)     # Default to large number
```

---

### Error 4: Not returning a list

**Problem:** Function doesn't return anything or returns wrong type

**Fix:**
```python
# ❌ Wrong (doesn't return anything)
def rank_flights(flights, preferences):
    sorted(flights, key=lambda x: x['price'])
    # Missing return!

# ✅ Correct
def rank_flights(flights, preferences):
    return sorted(flights, key=lambda x: x['price'])
```

---

## 🎓 Advanced Tips

### Tip 1: Handle Missing Fields Safely

```python
def rank_flights(flights, preferences):
    def safe_score(flight):
        price = flight.get('price', float('inf'))
        duration = flight.get('duration_min', float('inf'))
        stops = flight.get('stops', 999)

        # Only score if we have valid data
        if price == float('inf') or duration == float('inf'):
            return float('inf')  # Push to end

        return price / 500 + duration / 300 + stops * 0.5

    return sorted(flights, key=safe_score)
```

---

### Tip 2: Use Preferences Intelligently

```python
def rank_flights(flights, preferences):
    # Extract user's original query
    query = preferences.get('original_query', '').lower()

    # Detect keywords in query
    is_business_trip = 'business' in query or 'work' in query
    is_vacation = 'vacation' in query or 'holiday' in query

    if is_business_trip:
        # Business travelers prefer: fast, direct, morning flights
        return sorted(flights, key=lambda x: (x['stops'], x['duration_min']))
    elif is_vacation:
        # Vacationers prefer: cheap, any time
        return sorted(flights, key=lambda x: x['price'])
    else:
        # Default balanced ranking
        return sorted(flights, key=lambda x: x['price'] + x['stops'] * 100)
```

---

### Tip 3: Parse Departure Times

```python
from datetime import datetime

def rank_flights(flights, preferences):
    """Prefer morning flights."""

    def morning_score(flight):
        dep_time = flight.get('departure_time', '')

        try:
            # Parse ISO datetime string
            dt = datetime.fromisoformat(dep_time.replace('Z', '+00:00'))
            hour = dt.hour

            # Prefer 6 AM - 10 AM (low score = better)
            if 6 <= hour <= 10:
                time_score = 0
            elif 11 <= hour <= 14:
                time_score = 1
            else:
                time_score = 2
        except:
            time_score = 999  # Invalid time = worst

        # Combine with price
        price = flight.get('price', float('inf')) / 500

        return time_score + price

    return sorted(flights, key=morning_score)
```

---

## 📁 Example Files

I've created example algorithms for you in the `example_algorithms/` folder:

1. **[cheapest_direct.py](example_algorithms/cheapest_direct.py)** - Direct flights first, then by price
2. **[balanced_scorer.py](example_algorithms/balanced_scorer.py)** - Multi-factor balanced scoring
3. **[preference_based.py](example_algorithms/preference_based.py)** - Adapts to user preferences

You can upload any of these directly to test the feature!

---

## 🔍 How It Works Internally

When you upload an algorithm:

1. **File is read** as text ([line 529](streamlit_app.py#L529))
2. **Code is executed** in an isolated namespace ([line 533](streamlit_app.py#L533))
3. **Function is extracted** from namespace ([line 541](streamlit_app.py#L541))
4. **Signature is validated** (must have 2 parameters) ([line 545](streamlit_app.py#L545))
5. **Stored in session** for use in dropdowns ([line 549](streamlit_app.py#L549))

When you run the search:
1. Both selected algorithms are called with the same flight list
2. Each returns a ranked list
3. Results are **interleaved**: A1, B1, A2, B2, A3, B3...
4. Displayed in a table showing which algorithm picked each flight

---

## 🆘 Need Help?

**Issue:** Algorithm not working as expected?
- Add `print()` statements to debug
- Test with the local test script above
- Check that you're using `.get()` for safe field access

**Issue:** Can't see uploaded algorithm in dropdown?
- Make sure you clicked "➕ Add Algorithm"
- Check for error messages below the upload button
- Try refreshing the page (algorithms persist in session)

**Issue:** Want to modify an uploaded algorithm?
- Click "Remove" next to the algorithm name
- Upload the modified version with the same name

---

**Ready to create your own ranking algorithm?** 🚀

Start with one of the simple examples above and customize it to your needs!
