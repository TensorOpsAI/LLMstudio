import { useEffect, useRef } from "react";
import { usePlaygroundStore } from "../stores/PlaygroundStore";
import { getStatusColor } from "../utils";

export default function Output({ className }) {
  const { chatOutput, responseStatus } = usePlaygroundStore();
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current)
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
  });

  return (
    <div className={className}>
      <div className="border rounded-lg bg-gray-700 border-gray-600">
        <div className="flex items-center justify-between p-3 pt-2 border-gray-600">
          <h4 className="text-2xl">Output</h4>
          <div
            className={`w-3 h-3 rounded-full ${getStatusColor(responseStatus)}`}
          ></div>
        </div>
        <div className="px-4 py-2">
          <textarea
            id="output"
            ref={textareaRef}
            value={chatOutput}
            rows="20"
            readOnly
            className="w-full px-0 text-sm bg-transparent focus:outline-none resize-none"
            required
          ></textarea>
        </div>
      </div>
    </div>
  );
}
