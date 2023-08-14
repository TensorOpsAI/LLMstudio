import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

export const useTest = () => {
  const { model } = usePlaygroundStore();

  const testApi = useCallback(
    async (apikey) => {
      const chatProvider = getChatProvider(model);
      const promise = fetch(`http://localhost:3001/api/test/${chatProvider}`, {
        method: "post",
        headers: {
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify({
          apiKey: apikey,
          model: model,
        }),
      })
        .then((res) => res.json())
        .then((res) => {
          if (res) return 1;
          else throw new Error();
        });

      return await promise;
    },
    [model]
  );

  return testApi;
};
