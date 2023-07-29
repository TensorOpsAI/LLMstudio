import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { useExport } from "../api/useExport";

import { DataTable } from "./DataTable/DataTable";
import { columns } from "./DataTable/Columns";

export default function Executions() {
  const { executions } = usePlaygroundStore();
  const exportCSV = useExport();

  return <DataTable data={executions} columns={columns} />;
}
