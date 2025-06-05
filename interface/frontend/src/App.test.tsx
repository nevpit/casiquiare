import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders logs tab', () => {
  render(<App />);
  const tab = screen.getByRole('tab', { name: /logs/i });
  expect(tab).toBeInTheDocument();
});
