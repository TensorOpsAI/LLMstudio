'use client';
import { CheckIcon } from '@radix-ui/react-icons';

import { cn } from '@/lib/utils';
import { CommandItem } from '@/components/ui/command';

interface ModelItemProps {
  id: string;
  isSelected: boolean;
  onSelect: () => void;
}

export function ModelItem({ id, isSelected, onSelect }: ModelItemProps) {
  return (
    <CommandItem
      onSelect={onSelect}
      className='aria-selected:bg-primary aria-selected:text-primary-foreground'
    >
      {id}
      <CheckIcon
        className={cn(
          'ml-auto h-4 w-4',
          isSelected ? 'opacity-100' : 'opacity-0'
        )}
      />
    </CommandItem>
  );
}
