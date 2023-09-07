import { DotsHorizontalIcon } from "@radix-ui/react-icons";

import { Button } from "../../../../components/primitives/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../../../../components/primitives/DropdownMenu";
import { usePlaygroundStore } from "../../stores/PlaygroundStore";

export function DataTableRowActions({ row }) {
  const { setExecution, modelName, setModelName } = usePlaygroundStore();

  const restoreExecution = () => {
    console.log(modelName);
    setExecution(
      row.original.chatInput,
      row.original.chatOutput,
      row.original.modelName,
      row.original.parameters
    );
    console.log(modelName);
    setModelName(row.original.modelName);
    console.log(modelName);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="flex h-8 w-8 p-0 data-[state=open]:bg-muted"
        >
          <DotsHorizontalIcon className="h-4 w-4" />
          <span className="sr-only">Open menu</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[160px]">
        <DropdownMenuItem onClick={restoreExecution}>Restore</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem className="text-red-600">Delete</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
