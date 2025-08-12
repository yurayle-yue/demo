import React, { useState } from "react";

export default function Upload() {
  const [kotlinFile, setKotlinFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [resp, setResp] = useState(null);

  const uploadKotlin = async () => {
    if (!kotlinFile) return alert("pilih file kotlin");
    const fd = new FormData();
    fd.append("file", kotlinFile);
    const res = await fetch("/api/upload_kotlin", {
      method: "POST",
      body: fd
    });
    const j = await res.json();
    setResp(j);
  };

  const predict = async () => {
    if (!imageFile) return alert("pilih gambar");
    const fd = new FormData();
    fd.append("image", imageFile);
    const res = await fetch("/api/predict_food", {
      method: "POST",
      body: fd
    });
    const j = await res.json();
    setResp(j);
  };

  return (
    <div>
      <h2>Upload Kotlin (contoh)</h2>
      <input type="file" accept=".kt" onChange={e=>setKotlinFile(e.target.files[0])} />
      <button onClick={uploadKotlin}>Upload Kotlin</button>

      <hr/>

      <h2>Upload Image untuk Food Detection</h2>
      <input type="file" accept="image/*" onChange={e=>setImageFile(e.target.files[0])} />
      <button onClick={predict}>Predict Food</button>

      <pre>{resp && JSON.stringify(resp, null, 2)}</pre>
    </div>
  );
}
