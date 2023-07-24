export default function Output({ response, responseStatus }) {
  return (
    <div className="output--container">
      <div className="output--top">
        <span className="output--title">Output</span>
        <div className="output--status">
          <span>Status:</span>
          <div className={responseStatus}></div>
        </div>
      </div>
      <span>{response}</span>
    </div>
  );
}
