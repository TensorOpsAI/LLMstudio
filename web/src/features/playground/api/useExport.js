import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";

export const useExport = () => {
  const { executions } = usePlaygroundStore();

  const exportCSV = useCallback(
    (selected) => {
      console.log(selected);
      fetch(`http://localhost:3001/export`, {
        method: "POST",
        headers: {
          Accept: "application/json, text/plain",
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify(selected ? selected : executions),
      })
        .then((response) => response.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = "parameters.csv";
          link.click();
          URL.revokeObjectURL(url);
        });
    },
    [executions]
  );

  return exportCSV;
};
