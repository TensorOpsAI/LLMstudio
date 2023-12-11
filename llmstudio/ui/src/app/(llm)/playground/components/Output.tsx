'use client';
import { useEffect, useRef } from 'react';
import { useStore } from '@/app/(llm)/playground/store';
import Markdown from '@/components/Markdown';

export default function Output() {
  const { output, status } = useStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current)
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
  });

  const getStatusColor = (status: string) => {
    if (status === 'idle') return 'bg-slate-400';
    if (status === 'waiting') return 'bg-yellow-400';
    if (status === 'done') return 'bg-green-500';
    if (status === 'error') return 'bg-red-600';
  };

  return (
    <div className='flex h-full flex-col rounded-lg border'>
      <div className='mb-2 flex items-center justify-between p-3 '>
        <h4 className='text-2xl font-bold'>Output</h4>
        <div className={`h-3 w-3 rounded-full ${getStatusColor(status)}`}></div>
      </div>
      <div className='w-full flex-grow overflow-auto px-4 py-2'>
        <Markdown code={output} codeCopyable className='w-full' />
      </div>
    </div>
  );
}
