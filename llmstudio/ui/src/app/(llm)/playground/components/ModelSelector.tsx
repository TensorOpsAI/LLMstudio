'use client';
import React from 'react';
import { PopoverProps } from '@radix-ui/react-popover';
import { CaretSortIcon, CheckIcon } from '@radix-ui/react-icons';

import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandList,
} from '@/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';

import { ModelItem } from '@/app/(llm)/playground/components/ModelItem';
import { useModelFetch } from '@/app/(llm)/playground/hooks/useModelFetch';
import { useStore } from '@/app/(llm)/playground/store';

interface ModelSelectorProps extends PopoverProps {}

export default function ModelSelector(props: ModelSelectorProps) {
  const [open, setOpen] = React.useState(false);
  const { providers } = useModelFetch();
  const { model, setModel, setProvider } = useStore();

  return (
    <div className='flex items-center gap-2'>
      <Label htmlFor='model'>Model: </Label>
      <Popover open={open} onOpenChange={setOpen} {...props}>
        <PopoverTrigger asChild>
          <Button
            variant='outline'
            role='combobox'
            aria-expanded={open}
            aria-label='Select a model'
            className='w-full justify-between'
          >
            {model ? model : 'Select a model...'}
            <CaretSortIcon className='ml-2 h-4 w-4 shrink-0 opacity-50' />
          </Button>
        </PopoverTrigger>
        <PopoverContent align='end' className='w-[250px] p-0'>
          <Command loop>
            <CommandList className='h-[var(--cmdk-list-height)] max-h-[400px]'>
              <CommandInput placeholder='Search Models...' />
              <CommandEmpty>No Models found.</CommandEmpty>
              {providers.map((provider) => (
                <CommandGroup key={provider.name} heading={provider.name}>
                  {provider.models.map((singleModel) => (
                    <ModelItem
                      key={singleModel}
                      id={singleModel}
                      isSelected={model === singleModel}
                      onSelect={() => {
                        setProvider(provider.name);
                        setModel(singleModel);
                        setOpen(false);
                      }}
                    />
                  ))}
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
