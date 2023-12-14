'use client';

import { ColumnDef } from '@tanstack/react-table';

import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';

import { DataTableColumnHeader } from '@/app/(llm)/playground/components/DataTable/ColumnHeader';
import { DataTableRowActions } from '@/app/(llm)/playground/components/DataTable/RowActions';

export const columns: ColumnDef<any>[] = [
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
    enableSorting: false,
    enableHiding: false,
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
    accessorKey: 'cost',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Cost' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            ${row.original.metrics.cost.toFixed(6)}
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
    accessorKey: 'latency',
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title='Latency' />
    ),
    cell: ({ row }) => {
      return (
        <div className='flex space-x-2'>
          <span className='max-w-[100px] truncate font-medium'>
            {row.original.metrics.latency.toFixed(3)}s
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
            {row.original.metrics.time_to_first_token.toFixed(3)}s
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
            {row.original.metrics.inter_token_latency.toFixed(3)}s
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
