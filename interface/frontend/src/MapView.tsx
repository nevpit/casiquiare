import React, { useEffect, useState, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  LayersControl,
  GeoJSON,
} from 'react-leaflet';
import Drawer from '@mui/material/Drawer';
import Typography from '@mui/material/Typography';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import 'leaflet/dist/leaflet.css';

const { BaseLayer } = LayersControl;

const typeColors: Record<string, string> = {
  mound: '#e41a1c',
  outline: '#377eb8',
  unknown: '#4daf4a',
};

const MapView: React.FC = () => {
  const [detections, setDetections] = useState<any | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<any | null>(null);
  const [snippetUrl, setSnippetUrl] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 2000;
  const style = { height: '100vh', width: '100%' } as React.CSSProperties;

  useEffect(() => {
    fetch('/eyes/detections')
      .then((res) => {
        if (!res.ok) {
          if (res.status === 404) {
            setMsg('No detections found');
          } else {
            setMsg('Failed to load detections');
          }
          throw new Error('Failed to load detections');
        }
        return res.json();
      })
      .then((data) => setDetections(data))
      .catch(() => setDetections(null));
  }, []);

  useEffect(() => {
    setPage(0);
  }, [detections]);

  const numPages = useMemo(() => {
    if (!detections || detections.type !== 'FeatureCollection') return 1;
    return Math.ceil(detections.features.length / PAGE_SIZE);
  }, [detections]);

  const pagedData = useMemo(() => {
    if (!detections || detections.type !== 'FeatureCollection') return detections;
    if (detections.features.length <= PAGE_SIZE) return detections;
    const start = page * PAGE_SIZE;
    const end = start + PAGE_SIZE;
    return { ...detections, features: detections.features.slice(start, end) };
  }, [detections, page]);

  useEffect(() => {
    if (!selectedFeature) {
      setSnippetUrl(null);
      return;
    }
    const snippetPath = selectedFeature.properties?.snippet_path;
    if (!snippetPath) {
      setSnippetUrl(null);
      return;
    }
    fetch(`/eyes/snippets/${snippetPath}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load snippet');
        return res.blob();
      })
      .then((blob) => setSnippetUrl(URL.createObjectURL(blob)))
      .catch(() => setSnippetUrl(null));
  }, [selectedFeature]);

  const featureStyle = (feature: any): import('leaflet').PathOptions => {
    const t = feature?.properties?.feature_type || 'unknown';
    const color = typeColors[t] || '#888';
    const conf = feature?.properties?.confidence ?? 0;
    return {
      color,
      weight: 2,
      stroke: true,
      fillColor: color,
      fillOpacity: 0.2 + Math.max(0, Math.min(conf, 1)) * 0.8,
    };
  };

  const onEachFeature = (feature: any, layer: any) => {
    layer.on({
      click: () => setSelectedFeature(feature),
    });
  };

  return (
    <>
      <MapContainer center={[-3, -60]} zoom={10} style={style}>
        <LayersControl position="topright">
          <BaseLayer checked name="DTM">
            <TileLayer url="/tiles/dtm/{z}/{x}/{y}.png" />
          </BaseLayer>
          <BaseLayer name="Hillshade">
            <TileLayer url="/tiles/hillshade/{z}/{x}/{y}.png" />
          </BaseLayer>
          <BaseLayer name="NDVI">
            <TileLayer url="/tiles/ndvi/{z}/{x}/{y}.png" />
          </BaseLayer>
        </LayersControl>
        {pagedData && (
          <GeoJSON
            data={pagedData}
            style={featureStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
      {numPages > 1 && (
        <div
          style={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            background: 'rgba(255,255,255,0.8)',
            padding: 4,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <Button
            size="small"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(p - 1, 0))}
          >
            Prev
          </Button>
          <span>{page + 1} / {numPages}</span>
          <Button
            size="small"
            disabled={page >= numPages - 1}
            onClick={() => setPage((p) => Math.min(p + 1, numPages - 1))}
          >
            Next
          </Button>
        </div>
      )}
      <Drawer
        anchor="right"
        open={!!selectedFeature}
        onClose={() => setSelectedFeature(null)}
      >
        <div style={{ width: 300, padding: 16 }}>
          {selectedFeature && (
            <>
              <Typography variant="h6" gutterBottom>
                Feature {selectedFeature.properties?.id}
              </Typography>
              <pre>
                {JSON.stringify(selectedFeature.properties, null, 2)}
              </pre>
              {snippetUrl && (
                <img
                  src={snippetUrl}
                  alt="snippet"
                  style={{ width: '100%', marginTop: 8 }}
                />
              )}
            </>
          )}
        </div>
      </Drawer>
      <Snackbar
        open={!!msg}
        autoHideDuration={3000}
        onClose={() => setMsg(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="info" sx={{ width: '100%' }}>
          {msg}
        </Alert>
      </Snackbar>
    </>
  );
};

export default MapView;
