import React from 'react';
import Sidebar from './Sidebar';
import MapView from './MapView';
import './App.css';

function App() {
  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <Sidebar />
      <div style={{ flexGrow: 1 }}>
        <MapView />
      </div>
    </div>
  );
}

export default App;
