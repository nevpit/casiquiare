# Casiquiare Web Interface

This directory contains the Flask backend and React frontend that power the experimental Casiquiare viewer.

## Quickstart

Follow these steps to build the interface and launch the server.

1. **Install Python requirements**

   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Generate map tiles**

   Map tiles must be created from your DEM/hillshade/NDVI rasters. A helper is provided in `processing.tiles` which calls `gdal2tiles`.

   ```bash
   python - <<'PY'
   from processing import generate_default_tiles
   generate_default_tiles(
       dtm='path/to/dtm.tif',
       hillshade='path/to/hillshade.tif',
       ndvi='path/to/ndvi.tif',
       tiles_root='interface/backend/static/tiles'
   )
   PY
   ```

3. **Build the frontend**

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Run the development server**

   ```bash
   export FLASK_APP=backend.app:create_app
   flask run
   ```

The Flask server will serve the built React application and the generated tiles.
