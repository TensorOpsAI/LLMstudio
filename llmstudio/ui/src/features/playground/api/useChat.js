import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

export const useChat = () => {
  const {
    chatInput,
    setChatOutput,
    modelName,
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
        console.log(data);
        setChatOutput(data.chatOutput);
        setResponseStatus("done");
        addExecution(
          data.chatInput,
          data.chatOutput,
          data.inputTokens,
          data.outputTokens,
          data.cost,
          data.modelName,
          data.parameters
        );
      });
    },
    [
      chatInput,
      modelName,
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

      reader.read().then(function pump({ done, value }) {
        if (done) {
          setResponseStatus("done");
          addExecution(
            chatInput,
            chatOutput,
            inputTokens,
            outputTokens,
            cost,
            modelName,
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
          chatOutput += chunk;
          setChatOutput(chunk, true);
        }
        return reader.read().then(pump);
      });
    },
    [
      chatInput,
      modelName,
      parameters,
      setChatOutput,
      setResponseStatus,
      addExecution,
    ]
  );

  const submitChat = useCallback(async () => {
    setResponseStatus("waiting");
    const chatProvider = getChatProvider(modelName);
    const promise = fetch(`http://localhost:8000/api/chat/${chatProvider}`, {
      method: "post",
      headers: {
        "Content-Type": "application/json;charset=UTF-8",
      },
      body: JSON.stringify({
        chat_input: chatInput,
        model_name: modelName,
        api_key: apiKey,
        api_secret: apiSecret,
        api_region: apiRegion,
        is_stream: isStream,
        parameters: {
          temperature: parameters.temperature,
          max_tokens: parameters.maxTokens,
          top_p: parameters.topP,
          top_k: parameters.topK,
          frequency_penalty: parameters.frequencyPenalty,
          presence_penalty: parameters.presencePenalty,
        },
      }),
    })
      .then((response) => {
        isStream ? handleStream(response) : handleResponse(response);
      })
      .catch((error) => {
        throw new Error();
      });

    return await promise;
  }, [
    chatInput,
    modelName,
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
