import React, { useEffect, useState } from 'react';
import {
  MapContainer,
  TileLayer,
  LayersControl,
  GeoJSON,
} from 'react-leaflet';
import Drawer from '@mui/material/Drawer';
import Typography from '@mui/material/Typography';
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
  const style = { height: '100vh', width: '100%' } as React.CSSProperties;

  useEffect(() => {
    fetch('/eyes/detections')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load detections');
        return res.json();
      })
      .then((data) => setDetections(data))
      .catch(() => setDetections(null));
  }, []);

  useEffect(() => {
    if (!selectedFeature) {
      setSnippetUrl(null);
      return;
    }
    const id = selectedFeature.properties?.id;
    if (id == null) {
      setSnippetUrl(null);
      return;
    }
    fetch(`/eyes/snippets/${id}.png`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load snippet');
        return res.blob();
      })
      .then((blob) => setSnippetUrl(URL.createObjectURL(blob)))
      .catch(() => setSnippetUrl(null));
  }, [selectedFeature]);

  const featureStyle = (feature: any) => {
    const t = feature?.properties?.feature_type || 'unknown';
    const color = typeColors[t] || '#888';
    const conf = feature?.properties?.confidence ?? 0;
    return {
      color,
      weight: 1,
      fillColor: color,
      fillOpacity: 0.2 + Math.max(0, Math.min(conf, 1)) * 0.8,
    } as React.CSSProperties;
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
        {detections && (
          <GeoJSON
            data={detections}
            style={featureStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
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
    </>
  );
};

export default MapView;
