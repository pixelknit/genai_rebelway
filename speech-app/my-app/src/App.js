import React, { useState } from 'react';
import axios from 'axios';

function App () {
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('');

  const handleRecord = async() => {
    setStatus('Recording...');
    try {
      await axios.post('http://127.0.0.1:8000/record');
      setStatus('Recording complete.');
    } catch (err){
      console.log(err);
      setStatus('Failed to record.');
    }
  };

  const handleTranscribe = async () => {
    setStatus('Transcribing...');
    try{
      const res = await axios.get('http://127.0.0.1:8000/transcribe');
      setTranscript(res.data.transcript);
      setStatus('Transcription complete.');
    } catch (err){
      console.log(err);
      setStatus('Failed to transcribe.');
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Voice Recorder & Transcriber</h1>
      <button onClick={handleRecord}>Record</button>
      <button onClick={handleTranscribe} style={{ marginLeft: 10 }}>
        Transcribe
      </button>
      <p>{status}</p>
      <h2>Transcript:</h2>
      <p>{transcript}</p>
    </div>
  );

}

export default App;

