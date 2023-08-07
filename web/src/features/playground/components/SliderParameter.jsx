import * as React from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../../../components/primitives/HoverCard";
import { Label } from "../../../components/primitives/Label";
import { Slider } from "../../../components/primitives/Slider";

export default function SliderParameter({
  id,
  name,
  defaultValue,
  min,
  max,
  step,
  description,
}) {
  const { parameters, setParameter } = usePlaygroundStore();
  return (
    <div className="grid gap-2 pt-2">
      <HoverCard openDelay={200}>
        <HoverCardTrigger asChild>
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">{name}</Label>
              <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground hover:border-border">
                {parameters[id]}
              </span>
            </div>
            <Slider
              id={id}
              min={min}
              max={max}
              value={[parameters[id]]}
              step={step}
              onValueChange={(value) => setParameter(id, value[0])}
              className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              aria-label="Temperature"
            />
          </div>
        </HoverCardTrigger>
        <HoverCardContent
          align="start"
          className="w-[260px] text-sm"
          side="left"
        >
          {description}
        </HoverCardContent>
      </HoverCard>
    </div>
  );
}
