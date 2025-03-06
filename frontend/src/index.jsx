import React, { useState } from "react";
import ReactDOM from "react-dom/client";

function Index() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [response, setResponse] = useState("");
  const [uploaded, setUploaded] = useState(false);
  const [loading, setLoading] = useState(false); 

  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile)); 
      setUploaded(false);
      setLoading(false);
      setResponse("");
    }
  };

 
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) {
      alert("Sélectionne une image !");
      return;
    }

    console.log("Image sélectionnée :", file.name);
    setUploaded(true);
    setLoading(true); 
    setResponse(""); 

    
    setTimeout(() => {
      setLoading(false); 
      setResponse("Description générée : Ceci est une image représentant...");
    }, 2000);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}> Image en Texte </h1>

    
      <label style={styles.fileInputLabel}>
        Choisir une image
        <input type="file" accept="image/*" onChange={handleFileChange} style={styles.fileInput} />
      </label>

     
      {preview && (
        <div style={styles.imageContainer}>
          <img src={preview} alt="Preview" style={styles.imagePreview} />
        </div>
      )}

     
      {preview && !uploaded && (
        <button onClick={handleSubmit} style={styles.button}>Upload Image</button>
      )}

      
      {loading && <p style={styles.loading}>⏳ Génération en cours...</p>}

      
      {!loading && response && <p style={styles.response}>{response}</p>}
    </div>
  );
}


const styles = {
  container: {
    textAlign: "center",
    padding: "30px",
    fontFamily: "Arial, sans-serif",
  },
  title: {
    fontSize: "28px",
    fontWeight: "bold",
    marginBottom: "20px",
  },
  fileInputLabel: {
    display: "inline-block",
    backgroundColor: "#007bff",
    color: "white",
    padding: "12px 24px",
    fontSize: "16px",
    borderRadius: "8px",
    cursor: "pointer",
    marginBottom: "20px",
  },
  fileInput: {
    display: "none",
  },
  imageContainer: {
    width: "300px",
    height: "300px",
    margin: "0 auto",
    border: "2px solid #ccc",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.1)",
  },
  imagePreview: {
    maxWidth: "100%",
    maxHeight: "100%",
    borderRadius: "10px",
  },
  button: {
    backgroundColor: "#28a745",
    color: "white",
    border: "none",
    padding: "12px 24px",
    fontSize: "18px",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "0.3s",
    marginTop: "20px",
  },
  loading: {
    fontSize: "18px",
    color: "#555",
    marginTop: "15px",
    fontWeight: "bold",
    animation: "blink 1s infinite",
  },
  response: {
    fontSize: "18px",
    color: "#333",
    fontWeight: "bold",
    marginTop: "20px",
  },
};


ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Index />
  </React.StrictMode>
);
