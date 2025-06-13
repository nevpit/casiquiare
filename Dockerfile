# -------------------
# Stage 1: Build Frontend
# -------------------
    FROM node:18-alpine AS frontend-build

    # Create app directory and install frontend deps
    WORKDIR /app/interface/frontend
    COPY interface/frontend/package*.json ./
    RUN npm install
    
    # Build production React assets
    COPY interface/frontend/ ./
    RUN npm run build
    
    # -------------------
    # Stage 2: Build Backend + Final Image
    # -------------------
    FROM python:3.12-slim
    
    # Set working dir for the entire app
    WORKDIR /app
    
    # Install system libs for rasterio, GDAL, etc. (if needed)
    RUN apt-get update \
        && apt-get install -y --no-install-recommends \
           gdal-bin libgdal-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy and install Python requirements
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy backend code
    COPY interface/backend ./backend
    
    # Copy built frontend from stage 1 into the static folder
    COPY --from=frontend-build /app/interface/frontend/build \
         ./interface/frontend/build
    
    # Copy any other code (CLI, agents, utils, etc.)
    COPY ./*.py ./
    COPY agents ./agents
    COPY io_utils ./io_utils
    
    # Expose Flask port
    EXPOSE 5000
    
    # Env vars for Flask and interface settings
    ENV PYTHONUNBUFFERED=1 \
        FLASK_APP=backend.app:create_app \
        FLASK_ENV=production \
        FLASK_RUN_HOST=0.0.0.0 \
        # (optional) override these if you mount data volumes
        EYES_OUTPUT_DIR=/app/data/detections \
        EYES_TILE_DIR=/app/data/tiles
    
    # Default command: serve the Flask app (which also serves the React build)
    CMD ["flask", "run"]    