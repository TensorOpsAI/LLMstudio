export default function Header() {
  return (
    <header className="flex p-4 relative">
      <div className="inline-flex items-center">
        <img src="images/icon.png" alt="" className="w-16" />
        <span className="mr-1 ml-2 text-2xl">LLM Studio</span>
        <button className="bg-blue-600 py-1 px-5 rounded-full text-white">
          beta
        </button>
      </div>
      <div className="inline-flex items-center absolute left-1/2 -translate-x-1/2">
        <div className="inline-flex items-center py-4 px-6 gap-2 rounded-full hover:bg-slate-800 hover:cursor-pointer transition">
          <img
            className="w-5"
            src={process.env.PUBLIC_URL + "/svg/playground.svg"}
            alt=""
          />
          <span>Playground</span>
        </div>
      </div>
      {/* <div className="inline-flex items-center ml-auto gap-2">
        <img src="images/claudio.jpg" alt="" className="rounded-full w-10" />
        <span>Cláudio Lemos</span>
      </div> */}
    </header>
  );
}
