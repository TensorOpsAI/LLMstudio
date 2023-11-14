import { useEffect } from "react";
import ModelSelector from "./ModelSelector";
import ApiSettings from "./ApiSettings";
import SliderParameter from "./SliderParameter";
import CheckboxParameter from "./CheckboxParameter";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

import { parameters } from "./../assets/modelsConfig";

export default function Parameters({ className }) {
  const { model, isStream, setIsStream } = usePlaygroundStore();

  // useEffect(() => {}, [model]);

  return (
    <div className={className}>
      <div className="hidden flex-col space-y-4 sm:flex md:order-2">
        <div className="flex items-center gap-3">
          <ApiSettings />
          <CheckboxParameter
            value={isStream}
            setter={setIsStream}
            description={"Stream"}
          />
        </div>
        <ModelSelector />
        {parameters[getChatProvider(model)].map(
          (parameter) =>
            parameter.models.includes(model) && (
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
            )
        )}
      </div>
    </div>
  );
}
