import { useState, useEffect } from 'react';

interface DashboardMetrics {
  request_by_provider: Array<Record<string, number>>;
  request_by_model: Array<Record<string, number>>;
  total_cost_by_provider: Array<Record<string, number>>;
  total_cost_by_model: Array<Record<string, number>>;
  average_latency: Array<Record<string, number>>;
  average_ttft: Array<Record<string, number>>;
  average_itl: Array<Record<string, number>>;
  average_tps: Array<Record<string, number>>;
}

export function useDashboardFetch() {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    request_by_provider: [],
    request_by_model: [],
    total_cost_by_provider: [],
    total_cost_by_model: [],
    average_latency: [],
    average_ttft: [],
    average_itl: [],
    average_tps: []
  });

  useEffect(() => {
    async function fetchDashboardMetrics() {
      const queryParams = new URLSearchParams({ year: '2024', month: '1' });
      const response = await fetch(
        `http://${process.env.NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST}:${process.env.NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT}/api/tracking/dashboard/metrics?${queryParams}`
      );
      const data = await response.json();
      setMetrics(data);
    }
    fetchDashboardMetrics();
  }, []);

  return { metrics };
}
