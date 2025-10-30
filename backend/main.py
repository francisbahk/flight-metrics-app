"""
FastAPI main application entry point.
Flight Metrics Web Application Backend.
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Support both development mode (python backend/main.py) and production mode (uvicorn backend.main:app)
try:
    # Try relative imports first (for uvicorn backend.main:app)
    from .database import test_connection, init_db
    from .routes import flights, evaluate
except ImportError:
    # Fall back to absolute imports (for python backend/main.py)
    from database import test_connection, init_db
    from routes import flights, evaluate

# Create FastAPI application
app = FastAPI(
    title="Flight Metrics API",
    description="Flight search and evaluation system using Amadeus API",
    version="1.0.0",
)

# Configure CORS - Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:8000",  # Same origin
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(flights.router)
app.include_router(evaluate.router)


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    Test database connection and initialize tables if needed.
    """
    import sys
    print("\n" + "=" * 60, flush=True)
    print("Starting Flight Metrics API Server", flush=True)
    print("=" * 60, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"FastAPI version: {app.version}", flush=True)
    print(f"PORT env var: {os.getenv('PORT', 'not set')}", flush=True)
    print(f"MYSQL_HOST: {os.getenv('MYSQL_HOST', 'localhost')}", flush=True)
    print(f"MYSQL_PORT: {os.getenv('MYSQL_PORT', '3306')}", flush=True)
    print(f"MYSQL_DB: {os.getenv('MYSQL_DB', 'flights')}", flush=True)

    try:
        # Test database connection
        test_connection()

        # Initialize database tables (creates if they don't exist)
        # Comment out if you prefer to use schema.sql manually
        # init_db()

        print("✓ Server startup complete", flush=True)
        print("=" * 60 + "\n", flush=True)

    except Exception as e:
        print(f"✗ Startup error: {str(e)}", flush=True)
        print("⚠ Server will start anyway (database not required for health check)", flush=True)
        print("=" * 60 + "\n", flush=True)
        # Don't raise exception - allow server to start even if DB is not ready
        # This allows troubleshooting via health check endpoint


@app.get("/")
async def root():
    """
    Root endpoint - simple health check that doesn't require database.
    """
    return {
        "service": "Flight Metrics API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "flights": "/api/flights/*",
            "evaluate": "/api/evaluate/*"
        }
    }


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is running and database connection.
    """
    print("Health check called", flush=True)
    try:
        test_connection()
        db_status = "connected"
        print("Database connection: OK", flush=True)
    except Exception as e:
        db_status = "disconnected"
        print(f"Database connection: FAILED - {str(e)}", flush=True)

    return {
        "status": "healthy",
        "service": "Flight Metrics API",
        "version": "1.0.0",
        "database": db_status,
    }


@app.get("/api/info")
async def system_info():
    """
    System information endpoint.
    """
    import sys
    import platform

    return {
        "service": "Flight Metrics API",
        "version": "1.0.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "endpoints": {
            "flights": "/api/flights/*",
            "evaluate": "/api/evaluate/*",
            "health": "/api/health",
            "docs": "/docs",
        },
    }


# Serve React frontend if build exists
# This allows running the entire app from a single server in production
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"

if frontend_build_path.exists():
    # Serve static files from React build
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """
        Serve React application for all non-API routes.
        This enables client-side routing.
        """
        # Don't serve React for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # Serve index.html for all other routes
        index_path = frontend_build_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        raise HTTPException(status_code=404, detail="Frontend not found")


# Development server entry point
if __name__ == "__main__":
    import uvicorn

    # Get host and port from environment or use defaults
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))

    print(f"\n🚀 Starting development server on http://{host}:{port}")
    print(f"📚 API documentation available at http://{host}:{port}/docs")
    print(f"🔍 Interactive API docs at http://{host}:{port}/redoc\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
