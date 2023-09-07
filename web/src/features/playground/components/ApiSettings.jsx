import { useState } from "react";
import { useTest } from "../api/useTest";
import { Button } from "../../../components/primitives/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../../../components/primitives/Dialog";
import { GearIcon } from "@radix-ui/react-icons";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { credentials } from "./../assets/modelsConfig";
import { getChatProvider } from "../utils";
import { toast } from "sonner";

export default function ApiSettings() {
  const [open, setOpen] = useState(false);
  const {
    apiKey,
    setApiKey,
    apiSecret,
    setApiSecret,
    apiRegion,
    setApiRegion,
    modelName,
  } = usePlaygroundStore();
  const testApi = useTest();

  const onSubmit = (e) => {
    e.preventDefault();
    const apiKey = new FormData(e.target).get("apiKey");
    const apiSecret = new FormData(e.target).get("apiSecret");
    const apiRegion = new FormData(e.target).get("apiRegion");
    const promise = testApi(apiKey, apiSecret, apiRegion);
    toast.promise(promise, {
      loading: "Loading...",
      success: (data) => {
        setApiKey(apiKey);
        setApiSecret(apiSecret);
        setApiRegion(apiRegion);
        setOpen(false);
        return "API key has been updated";
      },
      error: `API key is not valid or doesn't have access to ${modelName}`,
    });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon">
          <GearIcon className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit API Key</DialogTitle>
          <DialogDescription>
            Enter your {getChatProvider(modelName, true)} API Key
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={onSubmit}>
          {credentials[modelName].needsKey && (
            <div className="px-4 py-2 my-4 bg-gray-800 rounded-lg">
              <textarea
                name="apiKey"
                rows="10"
                className="w-full px-0 text-sm text-white bg-gray-800 focus:outline-none placeholder-gray-400 resize-none"
                placeholder="Insert your API Key here..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                required
              ></textarea>
            </div>
          )}
          {credentials[modelName].needsSecret && (
            <div className="px-4 py-2 my-4 bg-gray-800 rounded-lg">
              <input
                name="apiSecret"
                rows="10"
                className="w-full px-0 text-sm text-white bg-gray-800 focus:outline-none placeholder-gray-400 resize-none"
                placeholder="Insert your API Secret here..."
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                required
              ></input>
            </div>
          )}
          {credentials[modelName].needsRegion && (
            <div className="px-4 py-2 my-4 bg-gray-800 rounded-lg">
              <input
                name="apiRegion"
                rows="10"
                className="w-full px-0 text-sm text-white bg-gray-800 focus:outline-none placeholder-gray-400 resize-none"
                placeholder="Insert your API region here..."
                value={apiRegion}
                onChange={(e) => setApiRegion(e.target.value)}
                required
              ></input>
            </div>
          )}
          <DialogFooter>
            <Button type="submit">Save changes</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
