import { usePlaygroundStore } from "../stores/PlaygroundStore";

import { DataTable } from "./DataTable/DataTable";
import { columns } from "./DataTable/Columns";

export default function Executions() {
  const { executions } = usePlaygroundStore();

  return <DataTable data={executions} columns={columns} />;
}
