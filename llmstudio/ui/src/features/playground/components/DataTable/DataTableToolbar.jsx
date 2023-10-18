import { Button } from "../../../../components/primitives/Button";
import { useExport } from "../../api/useExport";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../../../../components/primitives/DropdownMenu";

export function DataTableToolbar({ table }) {
  const exportCSV = useExport();

  return (
    <div className="flex items-end justify-between">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="secondary"
            className="flex data-[state=open]:bg-muted"
          >
            Export data
            <span className="sr-only">Export</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-[160px]">
          <DropdownMenuItem onClick={() => exportCSV()}>
            Export all
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() =>
              exportCSV(
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
