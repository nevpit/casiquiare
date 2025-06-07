import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import LogsPanel from './LogsPanel';
import SummaryPanel from './SummaryPanel';

const Sidebar: React.FC = () => {
  const [tab, setTab] = useState(0);
  const handleChange = (_: React.SyntheticEvent, newValue: number) => {
    setTab(newValue);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <Tabs
        orientation="vertical"
        value={tab}
        onChange={handleChange}
        sx={{ borderRight: 1, borderColor: 'divider', minWidth: 120 }}
      >
        <Tab label="Logs" />
        <Tab label="Summary" />
      </Tabs>
      <Box sx={{ flexGrow: 1, p: 2 }}>
        {tab === 0 && <LogsPanel />}
        {tab === 1 && <SummaryPanel />}
      </Box>
    </Box>
  );
};

export default Sidebar;
