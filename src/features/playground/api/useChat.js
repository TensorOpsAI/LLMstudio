import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

export const useChat = () => {
  const {
    input,
    setOutput,
    model,
    apiKey,
    parameters,
    setResponseStatus,
    addExecution,
  } = usePlaygroundStore();

  const handleResponse = useCallback(
    (response) => {
      response.json().then((data) => {
        setOutput(data.output);
        setResponseStatus("done");
        addExecution(
          input,
          data.output,
          data.input_tokens,
          data.output_tokens,
          data.cost,
          model,
          parameters
        );
      });
    },
    [input, model, parameters, setOutput, setResponseStatus, addExecution]
  );

  const handleStream = useCallback(
    (response) => {
      setOutput("");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let output = "";
      let inputTokens = 0;
      let outputTokens = 0;
      let cost = 0;

      reader.read().then(function pump({ done, value }) {
        if (done) {
          setResponseStatus("done");
          addExecution(
            input,
            output,
            inputTokens,
            outputTokens,
            cost,
            model,
            parameters
          );
          return;
        }

        const chunk = decoder.decode(value);
        if (chunk.startsWith("<END_TOKEN>")) {
          let endChunk = chunk.split(",");
          inputTokens = endChunk[1];
          outputTokens = endChunk[2];
          cost = endChunk[3];
        } else {
          output += chunk;
          setOutput(chunk, true);
        }
        return reader.read().then(pump);
      });
    },
    [input, model, parameters, setOutput, setResponseStatus, addExecution]
  );

  const submitChat = useCallback(async () => {
    setResponseStatus("waiting");
    const chatProvider = getChatProvider(model);
    const promise = fetch(`http://localhost:3001/chat/${chatProvider}`, {
      method: "post",
      headers: {
        "Content-Type": "application/json;charset=UTF-8",
      },
      body: JSON.stringify({
        prompt: input,
        model: model,
        apiKey: apiKey,
        parameters: parameters,
        stream: chatProvider !== "vertexai",
      }),
    })
      .then((response) => {
        chatProvider === "openai"
          ? handleStream(response)
          : handleResponse(response);
      })
      .catch((error) => {
        throw new Error();
      });

    return await promise;
  }, [
    input,
    model,
    apiKey,
    parameters,
    setResponseStatus,
    handleResponse,
    handleStream,
  ]);

  return submitChat;
};
