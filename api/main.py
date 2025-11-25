"""
FastAPI Backend for Flight Ranking Research Application
Replaces Streamlit with REST API + React frontend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

# Import routers
from .routers import search, ranking, lilo, evaluation, tracking

app = FastAPI(
    title="Flight Ranking API",
    description="Research platform for comparing flight ranking algorithms (LISTEN-U, LILO) with human recommendations",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://your-frontend-domain.com"  # Production (update later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(ranking.router, prefix="/api", tags=["Ranking"])
app.include_router(lilo.router, prefix="/api/lilo", tags=["LILO"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(tracking.router, prefix="/api/tracking", tags=["Tracking"])

# Serve React build (production)
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_build / "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        """Serve React frontend"""
        return FileResponse(str(frontend_build / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend_routes(full_path: str):
        """Serve React frontend for all routes (SPA)"""
        file_path = frontend_build / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_build / "index.html"))


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Development mode
    )