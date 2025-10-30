# Setup Status

## ✅ Step 1: Database Setup - COMPLETE!

MySQL has been installed and configured:
- ✅ MySQL 9.5.0 installed via Homebrew
- ✅ MySQL service started (running in background)
- ✅ Database 'flights' created
- ✅ Schema loaded (4 tables: flights, listen_rankings, team_draft_results, ratings)

Your MySQL connection details:
```
Host: localhost
Port: 3306
User: root
Password: (none - no password set)
Database: flights
```

## 📝 Step 2: Configure .env File - ACTION NEEDED

You need to add credentials to your `.env` file. Here's what to do:

### 1. Copy the example file (if not done):
```bash
cp .env.example .env
```

### 2. Edit `.env` and fill in these values:

#### Required MySQL Settings (ALREADY DONE FOR YOU):
```env
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=flights
```

#### Required Amadeus API Credentials (YOU NEED TO GET THESE):
```env
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
```

### How to Get Amadeus API Credentials:

1. **Go to**: https://developers.amadeus.com/
2. **Click**: "Register" or "Get your API key"
3. **Sign up** for a free account
4. **Create a new app** in the dashboard
5. **Copy** your API Key and API Secret
6. **Paste** them into your `.env` file

**Note**: Use the **TEST** environment credentials for development.

### Your .env file should look like this:
```env
# Amadeus API Credentials
AMADEUS_API_KEY=your_actual_key_from_amadeus
AMADEUS_API_SECRET=your_actual_secret_from_amadeus

# MySQL Database Configuration
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=flights

# Optional: Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

## 🎯 Next Steps (After Getting Amadeus Credentials):

Once you've added your Amadeus credentials to `.env`:

### Step 3: Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 5: Run the Application

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

Then open: http://localhost:3000

---

**Current Status**: Database ready ✅ | Need Amadeus API credentials to proceed
