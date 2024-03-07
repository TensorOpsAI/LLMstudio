import { useState, useEffect } from 'react';

interface DashboardMetrics {
  request_by_provider: Array<Record<string, number>>;
  request_by_model: Array<Record<string, number>>;
  total_cost_by_provider: Array<Record<string, number>>;
  total_cost_by_model: Array<Record<string, number>>;
  average_latency: Array<Record<string, number>>;
}

export function useDashboardFetch() {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    request_by_provider: [],
    request_by_model: [],
    total_cost_by_provider: [],
    total_cost_by_model: [],
    average_latency: [],
  });

  useEffect(() => {
    async function fetchDashboardMetrics() {
      const queryParams = new URLSearchParams({ year: '2024', month: '1' });
      const response = await fetch(
        `http://localhost:8080/api/tracking/dashboard/metrics?${queryParams}`
      );
      const data = await response.json();
      setMetrics(data);
    }
    fetchDashboardMetrics();
  }, []);

  return { metrics };
}
