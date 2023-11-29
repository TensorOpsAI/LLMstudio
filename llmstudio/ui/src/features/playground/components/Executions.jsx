import React, { useEffect } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";

import { DataTable } from "./DataTable/DataTable";
import { columns } from "./DataTable/Columns";

export default function Executions() {
  const { executions, addExecution } = usePlaygroundStore();

  useEffect(() => {
    fetch("http://localhost:8000/logs")
      .then((response) => response.json())
      .then((data) => {
        data.forEach((item) => {
          addExecution(
            item["chat_input"],
            item["chat_output"],
            item["usage"]["input_tokens"],
            item["usage"]["output_tokens"],
            item["usage"]["cost"],
            item["metrics"]["latency"],
            item["metrics"]["time_to_first_token"],
            item["metrics"]["inter_token_latency"],
            item["metrics"]["tokens_per_second"],
            item["model"],
            item["parameters"]
          );
        });
      });
  }, [addExecution]);

  return <DataTable data={executions} columns={columns} />;
}
