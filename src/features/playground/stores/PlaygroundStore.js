import { create } from "zustand";

export const usePlaygroundStore = create((set) => ({
  responseStatus: "idle",
  executions: [],
  input: "tell me a joke",
  output: "",
  model: "gpt-4",
  apiKey: "",
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
    input,
    output,
    inputTokens,
    outputTokens,
    cost,
    model,
    parameters
  ) =>
    set((state) => ({
      executions: [
        ...state.executions,
        {
          id: state.executions.length + 1,
          input: input,
          output: output,
          promptTokens: inputTokens,
          completionTokens: outputTokens,
          totalTokens: inputTokens + outputTokens,
          totalCost: cost,
          timestamp: Date.now(),
          model: model,
          parameters: parameters,
        },
      ],
    })),
  setInput: (input) => set({ input: input }),
  setOutput: (output, isChunk = false) =>
    set((state) => ({ output: isChunk ? state.output + output : output })),
  setModel: (model) => set({ model: model }),
  setApiKey: (apiKey) => set({ apiKey: apiKey }),
  setParameter: (parameter, value) => {
    set((state) => ({
      parameters: {
        ...state.parameters,
        [parameter]: Number(value),
      },
    }));
  },
}));
