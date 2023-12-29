'use client';
import { useRef } from 'react';
import { useStore } from '@/app/(llm)/playground/store';
import { useChat } from '@/app/(llm)/playground/hooks/useChat';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';

export default function Input() {
  const { input, status, setInput, setStatus } = useStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const submitChat = useChat();

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setStatus('loading');
    textareaRef.current?.blur();
    submitChat();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter')
      onSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    setStatus('idle');
  };

  return (
    <form onSubmit={onSubmit} className='flex h-full flex-col'>
      <div className='flex h-full flex-grow flex-col  rounded-lg border'>
        <div className='flex items-center justify-between p-3'>
          <h1 className='text-2xl font-bold'>Input</h1>
          <Button
            variant={'secondary'}
            type='submit'
            disabled={status == 'waiting'}
          >
            {status == 'waiting' && (
              <Loader2 className='mr-2 h-4 w-4 animate-spin' />
            )}
            Submit
          </Button>
        </div>
        <div className='flex-grow rounded-lg px-4 py-2'>
          <textarea
            id='input'
            ref={textareaRef}
            value={input}
            onKeyDown={onKeyDown}
            onChange={onInputChange}
            className='h-full w-full resize-none bg-[var(--background)] px-0 focus:outline-none'
            placeholder='Insert your prompt here...'
            required
          ></textarea>
        </div>
      </div>
    </form>
  );
}
