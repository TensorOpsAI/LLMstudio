// import { Cross2Icon } from "@radix-ui/react-icons";

// import { Button } from "../../../../components/primitives/Button";
// import { Input } from "../../../../components/primitives/Input";
// import { DataTableViewOptions } from "./DataTableViewOptions";

// import { priorities, statuses } from "./Data";
// import { DataTableFacetedFilter } from "./DataTableFacetedFilter";

export function DataTableToolbar({ table }) {
  return <></>;
  // const isFiltered = table.getState().columnFilters.length > 0;

  // return (
  //   <div className="flex items-center justify-between">
  //     <div className="flex flex-1 items-center space-x-2">
  //       <Input
  //         placeholder="Filter tasks..."
  //         value={table.getColumn("title")?.getFilterValue() ?? ""}
  //         onChange={(event) =>
  //           table.getColumn("title")?.setFilterValue(event.target.value)
  //         }
  //         className="h-8 w-[150px] lg:w-[250px]"
  //       />
  //       {table.getColumn("id") && (
  //         <DataTableFacetedFilter
  //           column={table.getColumn("id")}
  //           title="ID"
  //           options={["id"]}
  //         />
  //       )}
  //       {isFiltered && (
  //         <Button
  //           variant="ghost"
  //           onClick={() => table.resetColumnFilters()}
  //           className="h-8 px-2 lg:px-3"
  //         >
  //           Reset
  //           <Cross2Icon className="ml-2 h-4 w-4" />
  //         </Button>
  //       )}
  //     </div>
  //     <DataTableViewOptions table={table} />
  //   </div>
  // );
}
