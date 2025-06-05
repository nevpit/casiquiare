import React, { useEffect, useState } from 'react';
import Button from '@mui/material/Button';

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

  const fetchSummary = () => {
    fetch('/eyes/summary')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load summary');
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
    </div>
  );
};

export default SummaryPanel;
