import { useEffect } from "react";
import Input from "../components/Input";
import Output from "../components/Output";
import Parameters from "../components/Parameters";
import Executions from "../components/Executions";
import { Toaster } from "sonner";
import { usePlaygroundStore } from "../stores/PlaygroundStore";

export default function Playground() {
  const { setExecutions } = usePlaygroundStore();

  useEffect(() => {
    fetch("http://localhost:8000/logs", {
      method: "get",
    })
      .then((res) => res.text())
      .then((res) => {
        setExecutions(
          res
            .split("\n")
            .filter((line) => line.trim() !== "")
            .map((line) => JSON.parse(line))
        );
      });
  }, [setExecutions]);

  return (
    <>
      <Toaster richColors />
      <div className="flex p-4 flex-col gap-4">
        <div className="flex items-stretch gap-4">
          <Input className="flex-1" />
          <Output className="flex-1" />
          <Parameters className="flex-none w-44" />
        </div>
        <div>
          <Executions />
        </div>
      </div>
    </>
  );
}
