import React, { useState, useEffect } from 'react';
import { BookOpen, Cpu, CheckCircle, XCircle } from 'lucide-react';
import UploadSection from './components/UploadSection';
import ChatSection from './components/ChatSection';
import { uploadPdf, checkHealth } from './api';

export default function App() {
  const [documentInfo, setDocumentInfo] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    async function pingBackend() {
      const res = await checkHealth();
      if (res.status === 'healthy') {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    }
    pingBackend();
    const interval = setInterval(pingBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async (file) => {
    const res = await uploadPdf(file);
    setDocumentInfo(res);
  };

  const handleReset = () => {
    setDocumentInfo(null);
    setUploadError(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-left">
          <div className="logo-icon">
            <BookOpen size={24} />
          </div>
          <div>
            <h1 className="app-title">RAG PDF AI</h1>
            <p className="app-subtitle">Factual Q&A grounded in your PDF documents with page citations</p>
          </div>
        </div>

        <div className="header-right">
          <div className={`status-pill ${backendStatus}`}>
            <span className="status-dot"></span>
            <span>Backend: {backendStatus === 'online' ? 'Connected' : 'Offline'}</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <UploadSection
          documentInfo={documentInfo}
          onUploadSuccess={handleUpload}
          onReset={handleReset}
          isUploading={isUploading}
          setIsUploading={setIsUploading}
          error={uploadError}
          setError={setUploadError}
        />

        <ChatSection documentInfo={documentInfo} />
      </main>

      <footer className="footer">
        <p>
          Powered by <strong>PyMuPDF</strong>, <strong>SentenceTransformers (all-MiniLM-L6-v2)</strong>, <strong>NumPy</strong>, and <strong>Google Gemini</strong>.
        </p>
      </footer>
    </div>
  );
}
