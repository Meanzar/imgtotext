import React from 'react'

export default function index() {
    response = null;
    const handleSubmit = (e) => {
        e.preventDefault();
        console.log('submit');
    }
  return (
    <div>
        <h1>Front</h1>
        <input type='file'></input>
        <button onClick={handleSubmit}>Upload</button>
        {response ? <p>{response}</p> : null
        }
            
    </div>
  )
}
