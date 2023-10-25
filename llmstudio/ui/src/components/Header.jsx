import '@dotlottie/player-component';

export default function Header() {
  return (
    <header className="flex p-4 relative h-20">
      <div className="inline-flex items-center">
        <div className="w-[200px]">
          <dotlottie-player
            src="json/logo.json"
            autoplay
            speed="1"
            hover
            bounce
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
      <div className="inline-flex items-center absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2">
        <div className="inline-flex items-center py-4 px-6 gap-2 rounded-full hover:bg-slate-800 hover:cursor-pointer transition">
          <img
            className="w-5"
            speed="1" direction="1"
            src={process.env.PUBLIC_URL + "/svg/playground.svg"}
            alt=""
          />
          <span>Playground</span>
        </div>
      </div>
    </header>
  );
}
