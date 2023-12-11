'use client';

import { Table } from '@tanstack/react-table';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

import { useExport } from '@/app/(llm)/playground/hooks/useExport';

interface DataTableToolbarProps<TData> {
  table: Table<TData>;
}

export function DataTableToolbar<TData>({
  table,
}: DataTableToolbarProps<TData>) {
  const exportLogs = useExport();

  return (
    <div className='flex items-end justify-between'>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant='ghost' className='flex data-[state=open]:bg-muted'>
            Export data
            <span className='sr-only'>Export</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align='end' className='w-[160px]'>
          <DropdownMenuItem onClick={() => exportLogs()}>
            Export all
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() =>
              exportLogs(
                table.getSelectedRowModel().rows.map((row) => row.original)
              )
            }
          >
            Export selected
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
