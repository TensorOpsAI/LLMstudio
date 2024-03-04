import { useEffect } from 'react';
import { toast } from 'sonner';
import { useStore } from '@/app/(llm)/playground/store';

export function useLogsFetch() {
  const { status, setLogs } = useStore();

  useEffect(() => {
    async function fetchLogs() {
      try {
        const response = await fetch(
          `http://${process.env.NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST}:${process.env.NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT}/api/tracking/logs`
        );
        const data = await response.json();
        setLogs(data);
      } catch (e) {
        toast.error('Tracking API is not available');
      }
    }

    fetchLogs();
  }, [status, setLogs]);
}
