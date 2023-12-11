import { useEffect } from 'react';
import { toast } from 'sonner';
import { useStore } from '@/app/(llm)/playground/store';

export function useLogsFetch() {
  const { status, setLogs } = useStore();

  useEffect(() => {
    async function fetchLogs() {
      try {
        const response = await fetch('http://localhost:8080/api/tracking/logs');
        const data = await response.json();
        setLogs(data);
      } catch (e) {
        toast.error('Tracking API is not available');
      }
    }

    fetchLogs();
  }, [status, setLogs]);
}
