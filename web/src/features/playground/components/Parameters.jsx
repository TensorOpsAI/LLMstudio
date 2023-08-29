import ModelSelector from "./ModelSelector";
import ApiSettings from "./ApiSettings";
import SliderParameter from "./SliderParameter";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

import { parameters } from "./../assets/modelsConfig";

export default function Parameters({ className }) {
  const { modelName } = usePlaygroundStore();

  return (
    <div className={className}>
      <div className="hidden flex-col space-y-4 sm:flex md:order-2">
        <ApiSettings />
        <ModelSelector />
        {parameters[getChatProvider(modelName)].map((parameter) => (
          <SliderParameter
            key={parameter.id}
            id={parameter.id}
            name={parameter.name}
            defaultValue={parameter.defaultValue}
            min={parameter.min}
            max={parameter.max}
            step={parameter.step}
            description={parameter.description}
          />
        ))}
      </div>
    </div>
  );
}
