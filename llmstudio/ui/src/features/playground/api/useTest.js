import { useCallback } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getChatProvider } from "../utils";

export const useTest = () => {
  const { model } = usePlaygroundStore();

  const testApi = useCallback(
    async (apiKey, apiSecret, apiRegion) => {
      const chatProvider = getChatProvider(model);
      const promise = fetch(`http://localhost:8000/api/engine/validation/${chatProvider}`, {
        method: "post",
        headers: {
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify({
          api_key: apiKey,
          api_secret: apiSecret,
          api_region: apiRegion,
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
