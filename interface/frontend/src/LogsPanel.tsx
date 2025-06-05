import React, { useEffect, useState } from 'react';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

const LogsPanel: React.FC = () => {
  const [logs, setLogs] = useState<string>('');
  const [msg, setMsg] = useState<string | null>(null);

  const fetchLogs = () => {
    fetch('/eyes/logs?n=100')
      .then((res) => {
        if (!res.ok) {
          if (res.status === 404) {
            setMsg('No logs found');
          } else {
            setMsg('Failed to load logs');
          }
          throw new Error('Failed to load logs');
        }
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
    </div>
  );
};

export default LogsPanel;
