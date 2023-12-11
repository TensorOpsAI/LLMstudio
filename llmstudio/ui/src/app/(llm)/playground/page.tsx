import Image from 'next/image';
import {
  Input,
  Output,
  ModelSelector,
  Parameters,
  DataTable,
} from '@/app/(llm)/playground/components';

export default function Playground() {
  return (
    <>
      <div className='flex flex-col gap-4 p-4'>
        <div className='flex items-stretch gap-4'>
          <ModelSelector />
          <Parameters />
        </div>
        <div className='flex h-[500px] items-stretch gap-4'>
          <div className='w-1/2'>
            <Input />
          </div>
          <div className='w-1/2'>
            <Output />
          </div>
        </div>
      </div>
      <div className='flex flex-col gap-4 p-4'>
        <DataTable />
      </div>
    </>
  );
}
