'use client';
import { AreaChart, DonutChart, BarChart, Card, Title } from '@tremor/react';
import { useRef } from 'react';
import { useDashboardFetch } from '@/app/(llm)/Dashboard/hooks/useDashboardFetch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function Dashboard() {
  const { metrics } = useDashboardFetch();
  const toggleRef = useRef(null);

  if (metrics) {
    const providers =
      metrics.request_by_model.length > 0
        ? Object.keys(metrics.request_by_provider[0]).filter(
            (k) => k !== 'date'
          )
        : [];
    const models =
      metrics.request_by_model.length > 0
        ? Object.keys(metrics.request_by_model[0]).filter((k) => k !== 'date')
        : [];

    return (
      <>
        <div className='flex flex-col gap-4 p-4'>
          <h1 className='text-3xl font-bold'>Dashboard</h1>
          <div className='flex gap-4'>
            <Card>
              <Title>Requests by Provider</Title>
              <AreaChart
                className='mt-4 h-72'
                data={metrics.request_by_provider}
                index='date'
                categories={providers}
                yAxisWidth={30}
                connectNulls={true}
              />
            </Card>
            <Card>
              <Title>Requests by Model</Title>
              <AreaChart
                className='mt-4 h-72'
                data={metrics.request_by_model}
                index='date'
                categories={models}
                yAxisWidth={30}
                connectNulls={true}
              />
            </Card>
          </div>
          <div className='flex gap-4'>
            <Card>
              <div className='flex w-full justify-between'>
                <Title>Total Cost</Title>
                <Tabs
                  className='-translate-y-1.5'
                  defaultValue='model'
                  ref={toggleRef}
                >
                  <TabsList>
                    <TabsTrigger value='model'>Model</TabsTrigger>
                    <TabsTrigger value='provider'>Provider</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
              <DonutChart
                className='mt-6'
                data={metrics.total_cost_by_model}
                category='cost'
                index='name'
              />
            </Card>
            <Card>
              <Title>Average Latency</Title>
              <BarChart
                className='mt-6'
                data={metrics.average_latency}
                categories={['latency']}
                index='name'
                yAxisWidth={48}
              />
            </Card>
            <Card>
              <Title>Average Time To First Token</Title>
              <BarChart
                className='mt-6'
                data={metrics.average_ttft}
                categories={['ttft']}
                index='name'
                yAxisWidth={48}
              />
            </Card>
            <Card>
              <Title>Average Inter Token Latency</Title>
              <BarChart
                className='mt-6'
                data={metrics.average_itl}
                categories={['itl']}
                index='name'
                yAxisWidth={48}
              />
            </Card>
            <Card>
              <Title>Average Tokens Per Second</Title>
              <BarChart
                className='mt-6'
                data={metrics.average_tps}
                categories={['tps']}
                index='name'
                yAxisWidth={48}
              />
            </Card>
          </div>
        </div>
      </>
    );
  }
}
