// create jsons with definitions for parameteres + ranges for each model supported
import { useEffect, useRef } from "react";
import InputLabel from "@mui/material/InputLabel";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";

export default function Parameters({ parameters, setParameters }) {
  function updateRange(target, value, min, max) {
    const newTarget =
      target.type === "range"
        ? target
        : target.parentElement.parentElement.querySelector(
            'input[type="range"]'
          );

    newTarget.style.backgroundSize =
      ((value - min) * 100) / (max - min) + "% 100%";
  }

  function updateParameters(value, parameter, min, max) {
    setParameters({
      ...parameters,
      [parameter]:
        min & max ? (value > max ? max : value < min ? min : value) : value,
    });
  }

  const parametersRef = useRef();

  useEffect(() => {
    parametersRef.current
      .querySelectorAll('input[type="range"]')
      .forEach((target) => {
        updateRange(target, target.value, target.min, target.max);
      });
  }, [parameters]);

  return (
    <div className="parameters--container" ref={parametersRef}>
      <span className="parameters--title">Parameters</span>
      <div>
        <FormControl sx={{ m: 1, minWidth: 120 }}>
          <InputLabel htmlFor="grouped-native-select">Model</InputLabel>
          <Select
            native
            value={parameters.model}
            id="grouped-native-select"
            label="Model"
            onChange={(e) => {
              updateParameters(e.target.value, "model");
            }}
          >
            <option aria-label="None" value="" />
            <optgroup label="OpenAI">
              <option value={"gpt-3.5-turbo"}>gpt-3.5-turbo</option>
              <option value={"gpt-4"}>gpt-4</option>
            </optgroup>
            <optgroup label="Vertex AI">
              <option value={"text-bison@001"}>text-bison@001</option>
              <option value={"chat-bison@001"}>chat-bison@001</option>
            </optgroup>
          </Select>
        </FormControl>
      </div>
      {parameters.model.includes("gpt") && (
        <>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>API Key</span>
            </div>
            <div className="parameters--single-bottom">
              <input
                type="text"
                placeholder="Insert your API Key"
                onBlur={(e) => updateParameters(e.target.value, "apiKey")}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Temperature</span>
              <input
                type="text"
                value={parameters.temperature}
                onChange={(e) => {
                  updateParameters(e.target.value, "temperature", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={parameters.temperature}
                onChange={(e) => {
                  updateParameters(e.target.value, "temperature", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Maximum length</span>
              <input
                type="text"
                value={parameters.maximumLength}
                onChange={(e) => {
                  updateParameters(e.target.value, "maximumLength", 0, 2048);
                  updateRange(e.target, e.target.value, 0, 2048);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="2048"
                step="4"
                value={parameters.maximumLength}
                onChange={(e) => {
                  updateParameters(e.target.value, "maximumLength", 0, 2048);
                  updateRange(e.target, e.target.value, 0, 2048);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Top P</span>
              <input
                type="text"
                value={parameters.topP}
                onChange={(e) => {
                  updateParameters(e.target.value, "topP", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={parameters.topP}
                onChange={(e) => {
                  updateParameters(e.target.value, "topP", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Frequency Penalty</span>
              <input
                type="text"
                value={parameters.frequencyPenalty}
                onChange={(e) => {
                  updateParameters(e.target.value, "frequencyPenalty", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={parameters.frequencyPenalty}
                onChange={(e) => {
                  updateParameters(e.target.value, "frequencyPenalty", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Presence Penalty</span>
              <input
                type="text"
                value={parameters.presencePenalty}
                onChange={(e) => {
                  updateParameters(e.target.value, "presencePenalty", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={parameters.presencePenalty}
                onChange={(e) => {
                  updateParameters(e.target.value, "presencePenalty", 0, 2);
                  updateRange(e.target, e.target.value, 0, 2);
                }}
              ></input>
            </div>
          </div>
        </>
      )}
      {!parameters.model.includes("gpt") && (
        <>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>API Key</span>
            </div>
            <div className="parameters--single-bottom">
              <input
                type="text"
                placeholder="Insert your API Key"
                onBlur={(e) => updateParameters(e.target.value, "apiKey")}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Temperature</span>
              <input
                type="text"
                value={parameters.temperature}
                onChange={(e) => {
                  updateParameters(e.target.value, "temperature", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={parameters.temperature}
                onChange={(e) => {
                  updateParameters(e.target.value, "temperature", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Maximum length</span>
              <input
                type="text"
                value={parameters.maximumLength}
                onChange={(e) => {
                  updateParameters(e.target.value, "maximumLength", 1, 1024);
                  updateRange(e.target, e.target.value, 1, 1024);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="1"
                max="1024"
                step="4"
                value={parameters.maximumLength}
                onChange={(e) => {
                  updateParameters(e.target.value, "maximumLength", 1, 1024);
                  updateRange(e.target, e.target.value, 1, 1024);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Top P</span>
              <input
                type="text"
                value={parameters.topP}
                onChange={(e) => {
                  updateParameters(e.target.value, "topP", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={parameters.topP}
                onChange={(e) => {
                  updateParameters(e.target.value, "topP", 0, 1);
                  updateRange(e.target, e.target.value, 0, 1);
                }}
              ></input>
            </div>
          </div>
          <div className="parameters--single">
            <div className="parameters--single-top">
              <span>Top K</span>
              <input
                type="text"
                value={parameters.topK}
                onChange={(e) => {
                  updateParameters(e.target.value, "topK", 1, 40);
                  updateRange(e.target, e.target.value, 1, 40);
                }}
              />
            </div>
            <div className="parameters--single-bottom">
              <input
                type="range"
                min="1"
                max="40"
                step="0.5"
                value={parameters.topK}
                onChange={(e) => {
                  updateParameters(e.target.value, "topK", 1, 40);
                  updateRange(e.target, e.target.value, 1, 40);
                }}
              ></input>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
