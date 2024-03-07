import { useCallback } from 'react';
import { useStore } from '@/app/(llm)/playground/store';
import { generateStream } from '@/lib/utils';
import { toast } from 'sonner';

export const useChat = (): (() => Promise<void>) => {
  const { input, provider, model, parameters, setOutput, setStatus } =
    useStore();

  const submitChat = useCallback(async () => {
    setStatus('waiting');
    setOutput('');

    generateStream(
      `http://${process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST}:${
        process.env.NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT
      }/api/engine/chat/${provider.replace(/\s+/g, '').toLowerCase()}`,
      {
        chat_input: input,
        model: model,
        is_stream: true,
        has_end_token: true,
        parameters: parameters,
      }
    )
      .then(async (stream) => {
        for await (const chunk of stream)
          setOutput(
            chunk.includes('<END_TOKEN>')
              ? chunk.split('<END_TOKEN>')[0]
              : chunk,
            true
          );
        setStatus('done');
      })
      .catch((e: Error) => {
        toast.error(e.message);
        setStatus('error');
      });
  }, [input, model, parameters, provider, setStatus, setOutput]);

  return submitChat;
};
