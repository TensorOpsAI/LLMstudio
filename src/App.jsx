import React from "react";
import Header from "./components/Header";
import Playground from "./components/playground/Playground";
import { ThemeProvider, createTheme } from "@mui/material/styles";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <div className="App">
        <Header />
        <Playground />
      </div>
    </ThemeProvider>
  );
}

export default App;
