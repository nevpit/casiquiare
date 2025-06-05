import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders map tab', () => {
  render(<App />);
  const tab = screen.getByRole('tab', { name: /map/i });
  expect(tab).toBeInTheDocument();
});
