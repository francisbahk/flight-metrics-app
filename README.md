# Flight Metrics Web Application

A full-stack flight search and evaluation system built with FastAPI, React, MySQL, and the Amadeus Flight API. This application enables users to search for flights, store results, and evaluate different ranking algorithms using LISTEN and Team Draft methodologies.

## 🚀 Features

- **Flight Search**: Real-time flight search using Amadeus Flight Offers Search API v2
- **Data Storage**: MySQL database with computed flight metrics (distance, duration, time-of-day)
- **LISTEN Ranking**: Drag-and-drop interface for manual flight ranking
- **Team Draft Evaluation**: Interleaved ranking comparison between two algorithms
- **Flight Metrics**: Aggregated statistics and analytics on flight data
- **Responsive UI**: Modern React interface with Tailwind CSS

## 📋 Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for MySQL database operations
- **Amadeus API** - Flight search data provider
- **Python 3.9+** - Programming language

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **react-beautiful-dnd** - Drag-and-drop functionality
- **date-fns** - Date formatting

### Database
- **MySQL 8.0+** - Relational database

## 🏗️ Project Structure

```
flight_app/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── database.py             # Database connection and configuration
│   ├── amadeus_client.py       # Amadeus API client
│   ├── models/
│   │   ├── __init__.py
│   │   └── flight.py           # SQLAlchemy ORM models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── flights.py          # Flight search endpoints
│   │   └── evaluate.py         # Evaluation endpoints
│   └── utils/
│       ├── __init__.py
│       └── parse_duration.py   # Utility functions
├── frontend/
│   ├── public/
│   │   └── index.html          # HTML template
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── index.js            # React entry point
│   │   ├── index.css           # Global styles
│   │   ├── api/
│   │   │   └── flights.js      # API client
│   │   └── components/
│   │       ├── FlightSearch.jsx    # Search form component
│   │       ├── FlightTable.jsx     # Results table component
│   │       ├── ListenRanking.jsx   # LISTEN evaluation UI
│   │       └── TeamDraft.jsx       # Team Draft evaluation UI
│   └── package.json            # Frontend dependencies
├── database/
│   └── schema.sql              # MySQL database schema
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- MySQL 8.0 or higher
- Amadeus API credentials ([Get them here](https://developers.amadeus.com/))

### 1. Clone and Navigate

```bash
cd flight_app
```

### 2. Database Setup

Create the MySQL database:

```bash
mysql -u root -p
```

```sql
CREATE DATABASE flights;
EXIT;
```

Load the schema:

```bash
mysql -u root -p flights < database/schema.sql
```

### 3. Backend Setup

Create and configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=flights
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 4. Frontend Setup

Navigate to frontend directory and install dependencies:

```bash
cd frontend
npm install
```

### 5. Running the Application

#### Option A: Development Mode (Recommended)

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000`

#### Option B: Production Mode

Build the frontend:

```bash
cd frontend
npm run build
```

Run the backend (which will serve the built frontend):

```bash
cd backend
python main.py
```

Access the application at `http://localhost:8000`

## 📚 API Documentation

Once the backend is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Flight Search
- `GET /api/flights/search` - Search flights via Amadeus API
- `GET /api/flights/all` - Get all stored flights with filters
- `GET /api/flights/metrics` - Get aggregated flight metrics
- `GET /api/flights/{id}` - Get specific flight by ID

#### Evaluation
- `POST /api/evaluate/listen/ranking` - Submit LISTEN ranking
- `GET /api/evaluate/listen/rankings` - Get LISTEN rankings
- `POST /api/evaluate/teamdraft/start` - Start Team Draft session
- `POST /api/evaluate/teamdraft/submit` - Submit Team Draft preferences
- `GET /api/evaluate/teamdraft/results/{id}` - Get Team Draft results
- `POST /api/evaluate/rating` - Submit individual flight rating

#### System
- `GET /api/health` - Health check
- `GET /api/info` - System information

## 🎯 Usage Guide

### 1. Search for Flights

1. Enter origin and destination airport codes (e.g., JFK, LAX)
2. Select departure date
3. Set number of adults and max results
4. Click "Search Flights"

### 2. LISTEN Ranking Evaluation

1. Select flights using checkboxes in the results table
2. Click "LISTEN Ranking" button
3. Drag and drop flights to rank them (best to worst)
4. Add optional notes
5. Submit ranking

### 3. Team Draft Evaluation

1. Select flights using checkboxes in the results table
2. Click "Team Draft" button
3. Configure algorithm names (or use defaults)
4. Start evaluation
5. For each flight presented, click "Yes" (like) or "No" (dislike)
6. View final scores and winner

## 🗄️ Database Schema

### `flights` Table
Stores flight offers with computed metrics:
- Basic flight info (origin, destination, times, price)
- Computed metrics (duration_min, distances, time-of-day seconds)
- Raw Amadeus API response (JSON)

### `listen_rankings` Table
Stores LISTEN evaluation results:
- User ID and prompt
- Original flight IDs
- User's ranked order
- Timestamp and notes

### `team_draft_results` Table
Stores Team Draft evaluation results:
- Algorithm names and rankings
- Interleaved list
- User preferences (yes/no for each flight)
- Computed scores

### `ratings` Table
Stores individual flight ratings:
- User ID and flight ID
- Rating value (1-5)
- Timestamp

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AMADEUS_API_KEY` | Amadeus API key | Required |
| `AMADEUS_API_SECRET` | Amadeus API secret | Required |
| `MYSQL_USER` | MySQL username | root |
| `MYSQL_PASSWORD` | MySQL password | Required |
| `MYSQL_HOST` | MySQL host | localhost |
| `MYSQL_PORT` | MySQL port | 3306 |
| `MYSQL_DB` | MySQL database name | flights |
| `BACKEND_HOST` | Backend server host | 0.0.0.0 |
| `BACKEND_PORT` | Backend server port | 8000 |

### Frontend Configuration

To change the API endpoint in production, set the `REACT_APP_API_URL` environment variable:

```bash
REACT_APP_API_URL=https://your-api-domain.com/api npm run build
```

## 🧪 Testing

### Backend Testing

Test database connection:

```bash
cd backend
python -c "from database import test_connection; test_connection()"
```

Test API health:

```bash
curl http://localhost:8000/api/health
```

### Frontend Testing

```bash
cd frontend
npm test
```

## 🐛 Troubleshooting

### Database Connection Issues

- Verify MySQL is running: `mysql -u root -p`
- Check credentials in `.env` file
- Ensure database exists: `SHOW DATABASES;`

### Amadeus API Issues

- Verify API credentials are correct
- Check API quota/limits on Amadeus dashboard
- Ensure using test environment credentials for development

### CORS Issues

- Backend is configured to allow `localhost:3000` and `localhost:8000`
- For production, update CORS settings in `backend/main.py`

### Frontend Build Issues

- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 16+)

## 📝 Development Notes

### Adding New Airport Coordinates

Edit `backend/utils/parse_duration.py` and add to `AIRPORT_COORDINATES` dictionary:

```python
AIRPORT_COORDINATES = {
    "ABC": (latitude, longitude),
    # ...
}
```

### Custom Ranking Algorithms

Modify `frontend/src/components/TeamDraft.jsx` to implement custom ranking logic:

```javascript
// Algorithm A: Custom logic
const rankingA = [...selectedFlights]
  .sort((a, b) => yourCustomLogic(a, b))
  .map((f) => f.id);
```

## 🤝 Contributing

This is a complete implementation following the specifications. To extend:

1. Add new evaluation methods in `backend/routes/evaluate.py`
2. Create new React components in `frontend/src/components/`
3. Add database migrations for schema changes

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔗 Resources

- [Amadeus API Documentation](https://developers.amadeus.com/self-service/category/flights)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at `/docs` when backend is running
3. Verify all dependencies are installed correctly

---

**Built with FastAPI, React, and Amadeus API** ✈️
