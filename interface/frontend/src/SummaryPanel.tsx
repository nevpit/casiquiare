import React, { useEffect, useState } from 'react';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

function renderMarkdown(text: string): string {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped
    .split(/\n\n+/)
    .map((block) => {
      const line = block.trim();
      if (line.startsWith('### ')) {
        return `<h3>${line.slice(4)}</h3>`;
      }
      if (line.startsWith('## ')) {
        return `<h2>${line.slice(3)}</h2>`;
      }
      if (line.startsWith('# ')) {
        return `<h1>${line.slice(2)}</h1>`;
      }
      if (line.startsWith('- ')) {
        const items = line.split(/\n-/).map((l) => l.replace(/^\-\s*/, '').trim());
        const lis = items.map((t) => `<li>${t}</li>`).join('');
        return `<ul>${lis}</ul>`;
      }
      return `<p>${line}</p>`;
    })
    .join('');
}

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
      <div
        style={{ maxHeight: 300, overflowY: 'auto', padding: 8 }}
        dangerouslySetInnerHTML={{ __html: renderMarkdown(summary) }}
      />
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
