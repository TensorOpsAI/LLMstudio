import Input from "./Input";
import Output from "./Output";
import Parameters from "./Parameters";
import Executions from "./Executions";

export default function Editor() {
  return (
    <div className="editor--container">
      <div className="editor--top">
        <Input />
        <Output />
        {/* <Parameters /> */}
      </div>
      <div className="editor--bottom">
        <Executions />
      </div>
    </div>
  );
}
