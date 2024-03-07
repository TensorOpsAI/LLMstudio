import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useStore } from '@/app/(llm)/playground/store';

export interface ParameterType {
  id: string;
  name: string;
  type: string;
  default: number;
  min?: number;
  max?: number;
  step?: number;
}

export function useParameterFetch() {
  const [parameters, setParameters] = useState<ParameterType[]>([]);
  const { provider } = useStore();

  useEffect(() => {
    async function fetchParameters() {
      try {
        const queryParams = new URLSearchParams({
          provider: provider.replace(/\s+/g, '').toLowerCase(),
        });
        const response = await fetch(
          `http://${process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST}:${process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT}/api/engine/parameters?${queryParams}`
        );
        const data: ParameterType[] = await response.json();
        setParameters(data);
      } catch (e) {
        toast.error('LLMstudio Engine is not running');
      }
    }

    if (provider) fetchParameters();
  }, [provider]);

  return { parameters };
}
