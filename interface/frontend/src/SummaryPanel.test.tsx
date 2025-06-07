import React from 'react';
import { render, screen } from '@testing-library/react';
import SummaryPanel from './SummaryPanel';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      text: () => Promise.resolve('# Title'),
    } as Response)
  ) as jest.Mock;
});

test('renders refresh button', () => {
  render(<SummaryPanel />);
  const btn = screen.getByRole('button', { name: /refresh/i });
  expect(btn).toBeInTheDocument();
});

test('renders markdown heading', async () => {
  render(<SummaryPanel />);
  const heading = await screen.findByRole('heading', { level: 1 });
  expect(heading).toHaveTextContent('Title');
});

test('shows no data message on 404', async () => {
  (global.fetch as jest.Mock).mockResolvedValueOnce({
    ok: false,
    status: 404,
    text: () => Promise.resolve(''),
  } as Response);

  render(<SummaryPanel />);
  const alert = await screen.findByText(/no summary found/i);
  expect(alert).toBeInTheDocument();
});
