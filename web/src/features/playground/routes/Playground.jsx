import Input from "../components/Input";
import Output from "../components/Output";
import Parameters from "../components/Parameters";
import Executions from "../components/Executions";
import { Toaster } from "sonner";

export default function Playground() {
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
