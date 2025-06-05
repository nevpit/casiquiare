import React from 'react';
import { render, screen } from '@testing-library/react';
import LogsPanel from './LogsPanel';

beforeEach(() => {
  // mock fetch
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      text: () => Promise.resolve('logs'),
    } as Response)
  ) as jest.Mock;
});

test('renders refresh button', () => {
  render(<LogsPanel />);
  const btn = screen.getByRole('button', { name: /refresh/i });
  expect(btn).toBeInTheDocument();
});

test('shows no data message on 404', async () => {
  (global.fetch as jest.Mock).mockResolvedValueOnce({
    ok: false,
    status: 404,
    text: () => Promise.resolve(''),
  } as Response);

  render(<LogsPanel />);
  const alert = await screen.findByText(/no logs found/i);
  expect(alert).toBeInTheDocument();
});
