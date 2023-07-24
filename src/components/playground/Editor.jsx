import { useState, useReducer } from "react";
import Input from "./Input";
import Output from "./Output";
import Parameters from "./Parameters";
import History from "./History";
import { getEncoding } from "js-tiktoken";

export default function Editor(props) {
  const [responseStatus, setResponseStatus] = useState("idle");
  const [executions, setExecutions] = useState([]);
  const [parameters, setParameters] = useState({
    model: "gpt3",
    temperature: 1,
    maximumLength: 256,
    topP: 1,
    topK: 40,
    frequencyPenalty: 0,
    presencePenalty: 0,
  });
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useReducer(
    (response, chunk) => (chunk ? response + chunk : ""),
    ""
  );

  async function handleExport() {
    fetch(`http://127.0.0.1:3001/export`, {
      method: "POST",
      headers: {
        Accept: "application/json, text/plain",
        "Content-Type": "application/json;charset=UTF-8",
      },
      body: JSON.stringify(executions),
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
  }

  async function handlePromptSubmit() {
    setResponse(null);
    setResponseStatus("waiting");

    if (parameters.model !== "palm") {
      fetch(`http://127.0.0.1:3001/chat/openai`, {
        method: "POST",
        headers: {
          Accept: "application/json, text/plain",
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify({
          prompt: prompt,
          model: "gpt-4",
          apiKey: parameters.apiKey,
          temperature: parameters.temperature,
          maximumLength: parameters.maximumLength,
          topP: parameters.topP,
          frequencyPenalty: parameters.frequencyPenalty,
          presencePenalty: parameters.presencePenalty,
          stream: true,
        }),
      }).then((res) => {
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let output = "";
        let completionTokens = 0;

        reader.read().then(function pump({ done, value }) {
          if (done) {
            setResponseStatus("done");
            const promptTokens = getEncoding("gpt2").encode(prompt).length;
            setExecutions([
              ...executions,
              {
                id: executions.length + 1,
                input: prompt,
                output: output,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens,
                totalCost:
                  promptTokens * (0.0015 / 1000) +
                  completionTokens * (0.002 / 1000),
                timestamp: Date.now(),
                model: parameters.model,
                parameters: {
                  temperature: parameters.temperature,
                  maximumLength: parameters.maximumLength,
                  topP: parameters.topP,
                  frequencyPenalty: parameters.frequencyPenalty,
                  presencePenalty: parameters.presencePenalty,
                },
              },
            ]);
            return;
          }

          const chunk = decoder.decode(value);
          if (chunk.startsWith("<END_TOKEN>")) console.log(chunk.split(","));
          output += chunk;
          completionTokens++;
          setResponse(chunk);
          return reader.read().then(pump);
        });
      });
    } else {
      fetch(`http://127.0.0.1:3001/chat/palm`, {
        method: "POST",
        headers: {
          Accept: "application/json, text/plain",
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify({
          prompt: prompt,
          temperature: parameters.temperature,
          maximumLength: parameters.maximumLength,
          topP: parameters.topP,
          topK: parameters.topK,
        }),
      }).then((res) => {
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let output = "";
        let completionTokens = 0;

        reader.read().then(function pump({ done, value }) {
          if (done) {
            setResponseStatus("done");
            const promptTokens = getEncoding("gpt2").encode(prompt).length;
            setExecutions([
              ...executions,
              {
                id: executions.length + 1,
                input: prompt,
                output: output,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens,
                totalCost:
                  promptTokens * (0.0015 / 1000) +
                  completionTokens * (0.002 / 1000),
                timestamp: Date.now(),
                model: parameters.model,
                parameters: {
                  temperature: parameters.temperature,
                  maximumLength: parameters.maximumLength,
                  topP: parameters.topP,
                  topK: parameters.topK,
                },
              },
            ]);
            return;
          }

          const chunk = decoder.decode(value);
          output += chunk;
          completionTokens++;
          setResponse(chunk);
          return reader.read().then(pump);
        });
      });
    }
  }

  function handleExecutionRowClick(params) {
    setPrompt(params.row.input);
    setResponse(null);
    setResponse(params.row.output);
    setParameters({ ...parameters, ...params.row.parameters });
  }

  return (
    <div className="editor--container">
      <div className="editor--top">
        <Input
          prompt={prompt}
          setPrompt={setPrompt}
          handlePromptSubmit={handlePromptSubmit}
          responseStatus={responseStatus}
          setResponseStatus={setResponseStatus}
        />
        <Output response={response} responseStatus={responseStatus} />
        <Parameters parameters={parameters} setParameters={setParameters} />
      </div>
      <div className="editor--bottom">
        <History
          executions={executions}
          handleExport={handleExport}
          handleExecutionRowClick={handleExecutionRowClick}
        />
      </div>
    </div>
  );
}
