import React from 'react';
import { MapContainer, TileLayer, LayersControl } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const { BaseLayer } = LayersControl;

const MapView: React.FC = () => {
  const style = { height: '100vh', width: '100%' } as React.CSSProperties;
  return (
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
    </MapContainer>
  );
};

export default MapView;
