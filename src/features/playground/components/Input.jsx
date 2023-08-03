import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { useChat } from "../api/useChat";
import { toast } from "sonner";

export default function Input({ className }) {
  const { input, setInput, apiKey, responseStatus, setResponseStatus } =
    usePlaygroundStore();
  const submitChat = useChat();

  const onSubmit = (e) => {
    e.preventDefault();
    if (responseStatus !== "waiting") {
      const promise = submitChat();
      promise.catch((error) => {
        setResponseStatus("error");
        toast.error(
          apiKey === "" ? "Please set an API key" : "API key is not valid"
        );
      });
    }
  };

  const onInputChange = (e) => {
    setInput(e.target.value);
    setResponseStatus("idle");
  };

  return (
    <div className={className}>
      <form onSubmit={onSubmit}>
        <div className="border rounded-lg bg-gray-700 border-gray-600">
          <div className="flex items-center justify-between p-3 py-2 border-gray-600">
            <h4 className="text-2xl">Input</h4>
            <button
              type="submit"
              className="py-2.5 px-4 text-xs rounded-lg text-white bg-blue-700 hover:bg-blue-800 disabled:bg-gray-50 disabled:text-gray-600 transition"
              disabled={responseStatus === "waiting"}
            >
              Submit
            </button>
          </div>
          <div className="px-4 py-2 bg-gray-800 rounded-lg">
            <textarea
              id="input"
              value={input}
              onChange={onInputChange}
              rows="20"
              className="w-full px-0 text-sm text-white bg-gray-800 focus:outline-none placeholder-gray-400 resize-none"
              placeholder="Insert your prompt here..."
              required
            ></textarea>
          </div>
        </div>
      </form>
    </div>
  );
}
