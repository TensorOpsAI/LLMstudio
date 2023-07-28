import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { useExport } from "../api/useExport";

import { DataTable } from "./DataTable/DataTable";
import { columns } from "./DataTable/Columns";
import { tasks } from "./DataTable/Tasks";

export default function Executions() {
  const { executions } = usePlaygroundStore();
  const exportCSV = useExport();

  return <DataTable data={tasks} columns={columns} />;
}
