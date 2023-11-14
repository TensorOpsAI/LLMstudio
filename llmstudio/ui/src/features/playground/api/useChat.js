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
        console.log(data);
        setChatOutput(data.chatOutput);
        setResponseStatus("done");
        addExecution(
          data.chatInput,
          data.chatOutput,
          data.inputTokens,
          data.outputTokens,
          data.cost,
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

      reader.read().then(function pump({ done, value }) {
        if (done) {
          setResponseStatus("done");
          addExecution(
            chatInput,
            chatOutput,
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
    const promise = fetch(`http://localhost:8000/api/engine/chat/${chatProvider}`, {
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
