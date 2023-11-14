import * as React from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { CaretSortIcon, CheckIcon } from "@radix-ui/react-icons";

import { cn } from "../../../lib/utils";
import { useMutationObserver } from "../../../hooks/useMutationObserver";
import { Button } from "../../../components/primitives/Button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../../../components/primitives/Command";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../../../components/primitives/HoverCard";
import { Label } from "../../../components/primitives/Label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../../../components/primitives/Popover";

import { models, types } from "./../assets/modelsConfig";

export default function ModelSelector() {
  const { model, setModel } = usePlaygroundStore();
  const [open, setOpen] = React.useState(false);
  const [selectedModel, setSelectedModel] = React.useState(
    models.filter((m) => m.name === model)[0]
  );
  const [peekedModel, setPeekedModel] = React.useState(
    models.filter((m) => m.name === model)[0]
  );

  React.useEffect(() => {
    setSelectedModel(models.filter((m) => m.name === model)[0]);
  }, [model]);

  return (
    <div className="grid gap-2">
      <HoverCard openDelay={200}>
        <HoverCardTrigger asChild>
          <Label htmlFor="model">Model</Label>
        </HoverCardTrigger>
        <HoverCardContent
          align="start"
          className="w-[260px] text-sm"
          side="left"
        >
          The model which will generate the completion. Some models are suitable
          for natural language tasks, others specialize in code. Learn more.
        </HoverCardContent>
      </HoverCard>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            aria-label="Select a model"
            className="w-full justify-between"
          >
            <span className="w-24 truncate whitespace-nowrap">
              {selectedModel ? selectedModel.name : "Select a model..."}
            </span>

            <CaretSortIcon className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-[250px] p-0">
          <HoverCard>
            {/* <HoverCardContent
              side="left"
              align="start"
              forceMount
              className="min-h-[280px]"
            >
              <div className="grid gap-2">
                <h4 className="font-medium leading-none">{peekedModel.name}</h4>
                <div className="text-sm text-muted-foreground">
                  {peekedModel.description}
                </div>
                {peekedModel.strengths ? (
                  <div className="mt-4 grid gap-2">
                    <h5 className="text-sm font-medium leading-none">
                      Strengths
                    </h5>
                    <ul className="text-sm text-muted-foreground">
                      {peekedModel.strengths}
                    </ul>
                  </div>
                ) : null}
              </div>
            </HoverCardContent> */}
            <Command loop>
              <CommandList className="h-[var(--cmdk-list-height)] max-h-[400px]">
                <CommandInput placeholder="Search Models..." />
                <CommandEmpty>No Models found.</CommandEmpty>
                <HoverCardTrigger />
                {types.map((type) => (
                  <CommandGroup key={type} heading={type}>
                    {models
                      .filter((singleModel) => singleModel.type === type)
                      .map((singleModel) => (
                        <ModelItem
                          key={singleModel.id}
                          singleModel={singleModel}
                          isSelected={selectedModel?.id === singleModel.id}
                          onPeek={(singleModel) => setPeekedModel(singleModel)}
                          onSelect={() => {
                            setSelectedModel(singleModel);
                            setModel(singleModel.name);
                            setOpen(false);
                          }}
                        />
                      ))}
                  </CommandGroup>
                ))}
              </CommandList>
            </Command>
          </HoverCard>
        </PopoverContent>
      </Popover>
    </div>
  );
}

function ModelItem({ singleModel, isSelected, onSelect, onPeek }) {
  const ref = React.useRef(null);

  useMutationObserver(ref, (mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "attributes") {
        if (mutation.attributeName === "aria-selected") {
          onPeek(singleModel);
        }
      }
    }
  });

  return (
    <CommandItem
      key={singleModel.id}
      onSelect={onSelect}
      ref={ref}
      className="aria-selected:bg-primary aria-selected:text-primary-foreground"
    >
      {singleModel.name}
      <CheckIcon
        className={cn(
          "ml-auto h-4 w-4",
          isSelected ? "opacity-100" : "opacity-0"
        )}
      />
    </CommandItem>
  );
}
