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
import { getChatProvider } from "../utils";
import { toast } from "sonner";

export default function ApiSettings() {
  const [open, setOpen] = useState(false);
  const { apiKey, setApiKey, model } = usePlaygroundStore();
  const testApi = useTest();

  const onSubmit = (e) => {
    e.preventDefault();
    const apikey = new FormData(e.target).get("apikey");
    const promise = testApi(apikey);
    toast.promise(promise, {
      loading: "Loading...",
      success: (data) => {
        setApiKey(apikey);
        setOpen(false);
        return "API key has been updated";
      },
      error: `API key is not valid or doesn't have access to ${model}`,
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
            Enter your {getChatProvider(model)} API Key
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={onSubmit}>
          <div className="px-4 py-2 my-4 bg-gray-800 rounded-lg">
            <textarea
              name="apikey"
              rows="10"
              className="w-full px-0 text-sm text-white bg-gray-800 focus:outline-none placeholder-gray-400 resize-none"
              placeholder="Insert your API Key here..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              required
            ></textarea>
          </div>
          <DialogFooter>
            <Button type="submit">Save changes</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
