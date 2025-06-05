import React, { useEffect, useState } from 'react';
import Button from '@mui/material/Button';

const LogsPanel: React.FC = () => {
  const [logs, setLogs] = useState<string>('');

  const fetchLogs = () => {
    fetch('/eyes/logs?n=100')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load logs');
        return res.text();
      })
      .then((text) => setLogs(text))
      .catch(() => setLogs(''));
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  return (
    <div>
      <Button variant="contained" onClick={fetchLogs} style={{ marginBottom: 8 }}>
        Refresh
      </Button>
      <pre
        style={{
          maxHeight: 300,
          overflowY: 'auto',
          backgroundColor: '#f5f5f5',
          padding: 8,
        }}
      >
        {logs}
      </pre>
    </div>
  );
};

export default LogsPanel;
