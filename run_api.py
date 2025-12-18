#!/usr/bin/env python3
"""
Startup script for running the FastAPI backend with uvicorn.
"""
import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))

    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )
