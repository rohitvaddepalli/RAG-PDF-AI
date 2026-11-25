import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2, RotateCcw } from 'lucide-react';

export default function UploadSection({ onUploadSuccess, documentInfo, onReset, isUploading, setIsUploading, error, setError }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (file) => {
    setError(null);

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setError('Please upload a valid PDF file (.pdf).');
      return;
    }

    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File size exceeds the 20 MB limit.');
      return;
    }

    setIsUploading(true);
    try {
      await onUploadSuccess(file);
    } catch (err) {
      setError(err.message || 'Failed to process and embed PDF.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="card upload-card">
      <div className="section-header">
        <h2 className="section-title">
          <FileText className="icon" size={20} />
          1. Upload PDF Document
        </h2>
        <span className="badge">Max 20MB</span>
      </div>

      {error && (
        <div className="alert alert-error">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {!documentInfo ? (
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''} ${isUploading ? 'loading' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isUploading && fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInput}
            accept=".pdf,application/pdf"
            style={{ display: 'none' }}
            disabled={isUploading}
          />

          {isUploading ? (
            <div className="loading-state">
              <Loader2 className="spinner" size={40} />
              <p className="loading-title">Parsing & Embedding PDF...</p>
              <p className="loading-desc">Extracting pages and computing 384-d vectors with SentenceTransformers</p>
            </div>
          ) : (
            <div className="empty-state">
              <div className="icon-wrapper">
                <UploadCloud size={32} />
              </div>
              <p className="drop-title">Drag & drop your PDF here, or click to browse</p>
              <p className="drop-desc">Supports searchable PDF files up to 20MB</p>
              <button
                type="button"
                className="btn btn-outline"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Select File
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="document-badge">
          <div className="doc-meta">
            <CheckCircle2 className="success-icon" size={24} />
            <div>
              <p className="doc-name">{documentInfo.file_name || 'document.pdf'}</p>
              <p className="doc-stats">
                {documentInfo.page_count} page{documentInfo.page_count !== 1 ? 's' : ''} • {documentInfo.chunk_count} semantic chunk{documentInfo.chunk_count !== 1 ? 's' : ''} indexed
              </p>
            </div>
          </div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onReset}>
            <RotateCcw size={14} />
            Upload Another PDF
          </button>
        </div>
      )}
    </div>
  );
}
