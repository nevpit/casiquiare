import React, { useEffect, useState } from 'react';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import ReactMarkdown from 'react-markdown';

const SummaryPanel: React.FC = () => {
  const [summary, setSummary] = useState<string>('');
  const [msg, setMsg] = useState<string | null>(null);

  const fetchSummary = () => {
    fetch('/eyes/summary')
      .then((res) => {
        if (!res.ok) {
          if (res.status === 404) {
            setMsg('No summary found');
          } else {
            setMsg('Failed to load summary');
          }
          throw new Error('Failed to load summary');
        }
        return res.text();
      })
      .then((text) => setSummary(text))
      .catch(() => setSummary(''));
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  return (
    <div>
      <Button variant="contained" onClick={fetchSummary} style={{ marginBottom: 8 }}>
        Refresh
      </Button>
      <div style={{ maxHeight: 300, overflowY: 'auto', padding: 8 }}>
        <ReactMarkdown>{summary}</ReactMarkdown>
      </div>
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

export default SummaryPanel;
