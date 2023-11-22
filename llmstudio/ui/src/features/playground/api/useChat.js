import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

export const useChat = () => {
  const {
    chatInput,
    setChatOutput,
    model,
    apiKey,
    apiSecret,
    apiRegion,
    isStream,
    parameters,
    setResponseStatus,
    addExecution,
  } = usePlaygroundStore();

  const handleResponse = useCallback(
    (response) => {
      response.json().then((data) => {
        setChatOutput(data.chatOutput);
        setResponseStatus("done");
        addExecution(
          data.chat_input,
          data.chat_output,
          data.usage.input_tokens,
          data.usage.output_tokens,
          data.usage.cost,
          data.model,
          data.parameters
        );
      });
    },
    [
      chatInput,
      model,
      parameters,
      setChatOutput,
      setResponseStatus,
      addExecution,
    ]
  );

  const handleStream = useCallback(
    (response) => {
      setChatOutput("");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let chatOutput = "";
      let inputTokens = 0;
      let outputTokens = 0;
      let cost = 0;
      let latency = 0;
      let timeToFirstToken = 0;
      let interTokenLatency = 0;
      let tokensPerSecond = 0;

      reader.read().then(function pump({ done, value }) {
        if (done) {
          setResponseStatus("done");
          addExecution(
            chatInput,
            chatOutput,
            inputTokens,
            outputTokens,
            cost,
            latency,
            timeToFirstToken,
            interTokenLatency,
            tokensPerSecond,
            model,
            parameters
          );
          return;
        }

        const chunk = decoder.decode(value);
        if (chunk.includes("<END_TOKEN>")) {
          const [text, rawData] = chunk.split("<END_TOKEN>");
          chatOutput += text;
          setChatOutput(text, true);

          const dataPairs = rawData.split(",");
          const dataObject = {};

          dataPairs.forEach((pair) => {
            const [key, value] = pair.split("=");
            dataObject[key] = value;
          });

          inputTokens = dataObject["input_tokens"];
          outputTokens = dataObject["output_tokens"];
          cost = dataObject["cost"];
          latency = dataObject["latency"];
          timeToFirstToken = dataObject["time_to_first_token"];
          interTokenLatency = dataObject["inter_token_latency"];
          tokensPerSecond = dataObject["tokens_per_second"];
        } else {
          chatOutput += chunk;
          setChatOutput(chunk, true);
        }
        return reader.read().then(pump);
      });
    },
    [
      chatInput,
      model,
      parameters,
      setChatOutput,
      setResponseStatus,
      addExecution,
    ]
  );

  const submitChat = useCallback(async () => {
    setResponseStatus("waiting");
    const chatProvider = getChatProvider(model);
    const promise = fetch(
      `http://localhost:8000/api/engine/chat/${chatProvider}`,
      {
        method: "post",
        headers: {
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify({
          chat_input: chatInput,
          model: model,
          api_key: apiKey,
          api_secret: apiSecret,
          api_region: apiRegion,
          is_stream: isStream,
          has_end_token: true,
          parameters: {
            temperature: parameters.temperature,
            max_tokens: parameters.maxTokens,
            top_p: parameters.topP,
            top_k: parameters.topK,
            frequency_penalty: parameters.frequencyPenalty,
            presence_penalty: parameters.presencePenalty,
          },
        }),
      }
    )
      .then((response) => {
        isStream ? handleStream(response) : handleResponse(response);
      })
      .catch((error) => {
        throw new Error();
      });

    return await promise;
  }, [
    chatInput,
    model,
    apiKey,
    apiSecret,
    apiRegion,
    isStream,
    parameters,
    setResponseStatus,
    handleResponse,
    handleStream,
  ]);

  return submitChat;
};
