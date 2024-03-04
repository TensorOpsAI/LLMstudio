import { useCallback } from 'react';
import { useStore } from '@/app/(llm)/playground/store';

export const useExport = (): ((selected?: any[]) => void) => {
  const { logs } = useStore();

  const exportLogs = useCallback(
    (selected?: any[]): void => {
      fetch(
        `http://${process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST}:${process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT}/api/export`,
        {
          method: 'POST',
          headers: {
            Accept: 'application/json, text/plain',
            'Content-Type': 'application/json;charset=UTF-8',
          },
          body: JSON.stringify(selected || logs),
        }
      )
        .then((response) => response.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = 'parameters.csv';
          link.click();
          URL.revokeObjectURL(url);
        });
    },
    [logs]
  );

  return exportLogs;
};
