# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### Step 1: Database Setup (2 minutes)

```bash
# Create database
mysql -u root -p
```

In MySQL shell:
```sql
CREATE DATABASE flights;
EXIT;
```

Load schema:
```bash
mysql -u root -p flights < database/schema.sql
```

### Step 2: Configure Environment (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required: AMADEUS_API_KEY, AMADEUS_API_SECRET, MYSQL_PASSWORD
```

### Step 3: Install Dependencies (1 minute)

Backend:
```bash
pip install -r requirements.txt
```

Frontend:
```bash
cd frontend
npm install
cd ..
```

### Step 4: Run the Application (1 minute)

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Step 5: Access the App

Open your browser to: **http://localhost:3000**

Backend API docs: **http://localhost:8000/docs**

## 🎯 First Flight Search

1. Click a quick example button (e.g., "JFK → LAX")
2. Click "Search Flights"
3. Select some flights with checkboxes
4. Try "LISTEN Ranking" or "Team Draft" evaluation

## 📌 Important Notes

- **Amadeus API**: Get free test credentials at https://developers.amadeus.com/
- **MySQL**: Must be running on port 3306 (default)
- **Ports**: Backend runs on 8000, Frontend on 3000

## ❓ Having Issues?

1. **Database error**: Check MySQL is running and credentials in `.env`
2. **Amadeus error**: Verify API credentials are correct
3. **Frontend won't start**: Run `npm install` again in frontend directory
4. **Port already in use**: Stop other services or change ports in code

## 📚 What's Included

✅ Complete FastAPI backend with 14 API endpoints
✅ React frontend with 4 major components
✅ MySQL database with 4 tables
✅ Amadeus API integration
✅ LISTEN drag-and-drop ranking
✅ Team Draft interleaved evaluation
✅ Responsive design with Tailwind CSS
✅ Full error handling and validation

---

For detailed documentation, see [README.md](README.md)
