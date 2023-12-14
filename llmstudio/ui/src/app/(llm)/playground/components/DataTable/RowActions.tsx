'use client';

import { DotsHorizontalIcon } from '@radix-ui/react-icons';
import { Row } from '@tanstack/react-table';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

import { Log, useStore } from '@/app/(llm)/playground/store';

interface DataTableRowActionsProps<TData> {
  row: Row<TData>;
}

export function DataTableRowActions<TData extends Log>({
  row,
}: DataTableRowActionsProps<TData>) {
  const { setInput, setOutput, setModel, setProvider, setParameter } =
    useStore();

  const restore = () => {
    setInput(row.original.chat_input);
    setOutput(row.original.chat_output);
    setModel(row.original.model);
    setProvider(row.original.provider);
    Object.entries(row.original.parameters).forEach(([key, value]) =>
      setParameter(key, value)
    );
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant='ghost'
          className='flex h-8 w-8 p-0 data-[state=open]:bg-muted'
        >
          <DotsHorizontalIcon className='h-4 w-4' />
          <span className='sr-only'>Open menu</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align='end' className='w-[160px]'>
        <DropdownMenuItem onClick={restore}>Restore</DropdownMenuItem>
        <DropdownMenuItem>
          <div className='text-red-600'>Delete</div>
          <DropdownMenuShortcut>⌘⌫</DropdownMenuShortcut>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
