'use client';

import { ColumnDef } from '@tanstack/react-table';

import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';

import { DataTableColumnHeader } from '@/app/(llm)/playground/components/DataTable/ColumnHeader';
import { DataTableRowActions } from '@/app/(llm)/playground/components/DataTable/RowActions';

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';

import CodeBlock from '@/components/CodeBlock';
import { Maximize2 } from 'lucide-react';

import { Button } from '@/components/ui/button';

export const columns: ColumnDef<any>[] = [
  {
    accessorKey: 'context',
    header: () => <></>,
    cell: ({ row }) => (
      <Sheet key={row.id}>
        <SheetTrigger>
          <Button variant='outline' size='icon'>
            <Maximize2 className='h-4 w-4' />
          </Button>
        </SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Context</SheetTitle>
            <SheetDescription>
              {row.original.context.map((item, index) => (
                <div key={index} className='my-2'>
                  <div>
                    <strong>Role:</strong> {item.role}
                  </div>
                  {item.content && (
                    <div>
                      <strong>Content:</strong> {item.content}
                    </div>
                  )}
                  {item.function_call && (
                    <CodeBlock
                      language='json'
                      value={JSON.stringify(item.function_call, null, 2)}
                      className='my-2'
                    />
                  )}
                  {item.name && (
                    <div>
                      <strong>Function Name:</strong> {item.name}
                    </div>
                  )}
                </div>
              ))}
            </SheetDescription>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    ),
    enablePinning: true,
  },
  {
    id: 'select',
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() ||
          (table.getIsSomePageRowsSelected() && 'indeterminate')
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label='Select all'
        className='translate-y-[2px]'
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        aria-label='Select row'
        className='translate-y-[2px]'
      />
    ),
    enablePinning: true,
  },
  {
    accessorKey: 'log_id',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='ID' />
    ),
    cell: ({ row }) => <div className='w-[15px]'>{row.original.log_id}</div>,
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: 'chat_input',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Input' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          {/* {row.original.label && (
            <Badge variant='outline'>{row.original.label}</Badge>
          )} */}
          <span className='max-w-[200px] truncate font-medium'>
            {row.original.chat_input}
          </span>
        </div>
      );
    },
    enableSorting: false,
  },
  {
    accessorKey: 'chat_output',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Output' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[200px] truncate font-medium'>
            {row.original.chat_output}
          </span>
        </div>
      );
    },
    enableSorting: false,
  },
  {
    accessorKey: 'model',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Model' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <Badge variant='outline'>{row.original.provider}</Badge>
          <span className='max-w-[150px] truncate font-medium'>
            {row.original.model}
          </span>
        </div>
      );
    },
    enableSorting: false,
  },
  {
    accessorKey: 'cost_usd',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Cost' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            ${row.original.metrics.cost_usd.toFixed(6)}
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'input_tokens',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Input Tokens' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[50px] truncate font-medium'>
            {row.original.metrics.input_tokens}
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'output_tokens',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Output Tokens' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[50px] truncate font-medium'>
            {row.original.metrics.output_tokens}
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'latency_s',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Latency' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            {row.original.metrics.latency_s.toFixed(3)}s
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'ttft',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='TTFT' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            {row.original.metrics.time_to_first_token_s.toFixed(3)}s
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'itl',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='ITL' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            {row.original.metrics.inter_token_latency_s.toFixed(3)}s
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'tps',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='T/S' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            {row.original.metrics.tokens_per_second.toFixed(0)}
          </span>
        </div>
      );
    },
  },
  {
    id: 'actions',
    cell: ({ row }) => <DataTableRowActions row={row} />,
  },
];
