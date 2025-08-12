import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Welcome from "./pages/Welcome";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Upload from "./pages/Upload";

function App() {
  return (
    <BrowserRouter>
      <nav style={{padding:10, borderBottom:"1px solid #ddd"}}>
        <Link to="/">Welcome</Link>{" | "}
        <Link to="/login">Sign In</Link>{" | "}
        <Link to="/register">Sign Up</Link>{" | "}
        <Link to="/upload">Upload</Link>
      </nav>
      <div style={{padding:20}}>
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/upload" element={<Upload />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
