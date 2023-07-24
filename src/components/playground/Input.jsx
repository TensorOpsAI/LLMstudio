import { useState } from "react";

export default function Input({
  prompt,
  setPrompt,
  handlePromptSubmit,
  responseStatus,
  setResponseStatus,
}) {
  const [tokens, setTokens] = useState(0);

  function onChange(e) {
    setTokens(Math.floor(e.target.value.length / 4));
    setPrompt(e.target.value);
    if (responseStatus !== "idle") setResponseStatus("idle");
  }

  return (
    <div className="input--container">
      <div className="input--top">
        <span className="input--title">Input</span>
        <img
          src={process.env.PUBLIC_URL + "/svg/play.svg"}
          className={`input--submit ${responseStatus}`}
          alt=""
          onClick={() => {
            if (responseStatus !== "waiting") handlePromptSubmit();
          }}
        />
        <span>~{tokens} tokens</span>
      </div>
      <textarea
        className="input--textarea"
        placeholder="Insert your prompt here"
        rows={10}
        value={prompt}
        onChange={(e) => onChange(e)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            handlePromptSubmit();
          }
        }}
      ></textarea>
    </div>
  );
}
