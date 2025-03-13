import type { PageLoad } from "./$types";

interface Metric {
  metric: string;
  value: number;
}

interface Metrics {
  metrics: Metric[];
}

export const load: PageLoad = async ({ fetch }) => {
  const trainResponse = await fetch("http://localhost:8000/metrics/train");
  const trainMetrics: Metrics = await trainResponse.json();

  const testResponse = await fetch("http://localhost:8000/metrics/train");
  const testMetrics: Metrics = await testResponse.json();

  return {
    trainMetrics,
    testMetrics,
  };
};
