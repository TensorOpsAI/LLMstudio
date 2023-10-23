import { create } from "zustand";

export const usePlaygroundStore = create((set) => ({
  responseStatus: "idle",
  executions: [],
  chatInput: "",
  chatOutput: "",
  modelName: "gpt-3.5-turbo",
  apiKey: "",
  apiSecret: "",
  apiRegion: "",
  isStream: true,
  parameters: {
    temperature: 1,
    maxTokens: 256,
    topP: 1,
    topK: 40,
    frequencyPenalty: 0,
    presencePenalty: 0,
  },
  setResponseStatus: (status) => set({ responseStatus: status }),
  addExecution: (
    chatInput,
    chatOutput,
    inputTokens,
    outputTokens,
    cost,
    modelName,
    parameters
  ) =>
    set((state) => ({
      executions: [
        ...state.executions,
        {
          id: state.executions.length + 1,
          chatInput: chatInput,
          chatOutput: chatOutput,
          inputTokens: Number(inputTokens),
          outputTokens: Number(outputTokens),
          totalTokens: Number(inputTokens + outputTokens),
          cost: Number(cost),
          timestamp: new Date(),
          modelName: modelName,
          parameters: parameters,
        },
      ],
    })),
  setExecutions: (executions) => set({ executions: executions }),
  setExecution: (chatInput, chatOutput, modelName, parameters) =>
    set({
      chatInput: chatInput,
      chatOutput: chatOutput,
      modelName: modelName,
      parameters: parameters,
    }),
  setChatInput: (chatInput) => set({ chatInput: chatInput }),
  setChatOutput: (chatOutput, isChunk = false) =>
    set((state) => ({
      chatOutput: isChunk ? state.chatOutput + chatOutput : chatOutput,
    })),
  setModelName: (modelName) => set({ modelName: modelName }),
  setApiKey: (apiKey) => set({ apiKey: apiKey }),
  setApiSecret: (apiSecret) => set({ apiSecret: apiSecret }),
  setApiRegion: (apiRegion) => set({ apiRegion: apiRegion }),
  setIsStream: () =>
    set((state) => ({
      isStream: !state.isStream,
    })),
  setParameter: (parameter, value) => {
    set((state) => ({
      parameters: {
        ...state.parameters,
        [parameter]: Number(value),
      },
    }));
  },
}));
