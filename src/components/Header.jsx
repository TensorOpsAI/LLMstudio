export default function Header() {
  return (
    <header className="header--container">
      <div className="header--left">
        <img src="images/icon.png" alt="" className="icon" />
        <span className="title">LLM Studio</span>
        <span className="beta">beta</span>
      </div>
      <div className="header--center">
        {/* <div className="button home">
          <img src={process.env.PUBLIC_URL + "/svg/home.svg"} alt="" />
        </div> */}
        <div className="button playground selected">
          <img src={process.env.PUBLIC_URL + "/svg/playground.svg"} alt="" />
          <span>Playground</span>
        </div>
        {/* <div className="button settings">
          <img src={process.env.PUBLIC_URL + "/svg/settings.svg"} alt="" />
        </div> */}
      </div>
      <div className="header--right">
        <img src="images/claudio.jpg" alt="" className="icon" />
        <div className="user">
          <span className="name">Cláudio Lemos</span>
          <span className="email">claudio.lemos@tensorops.ai</span>
        </div>
        {/* <img src={process.env.PUBLIC_URL + "/svg/arrow.svg"} alt="" /> */}
      </div>
    </header>
  );
}
