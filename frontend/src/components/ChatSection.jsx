import React, { useState } from 'react';
import { MessageSquare, Send, Sparkles, BookOpen, Loader2, Bookmark, Check } from 'lucide-react';
import { askQuestion } from '../api';
import FormattedAnswer from './FormattedAnswer';

const QUICK_PROMPTS = [
  'Summarize the key takeaways and main points.',
  'What are the primary findings, metrics, or conclusions?',
  'What action items or recommendations are outlined?',
];

export default function ChatSection({ documentInfo }) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);

  const isDocActive = Boolean(documentInfo?.document_id);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    const cleanQ = question.trim();
    if (!cleanQ || !isDocActive || isLoading) return;

    setError(null);
    setIsLoading(true);

    try {
      const res = await askQuestion(documentInfo.document_id, cleanQ);
      const newEntry = {
        id: Date.now(),
        question: cleanQ,
        answer: res.answer,
        sources: res.sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setChatHistory([newEntry, ...chatHistory]);
      setQuestion('');
    } catch (err) {
      setError(err.message || 'Failed to retrieve an answer from the document.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickPrompt = (promptText) => {
    setQuestion(promptText);
  };

  return (
    <div className={`card chat-card ${!isDocActive ? 'disabled-card' : ''}`}>
      <div className="section-header">
        <h2 className="section-title">
          <MessageSquare className="icon" size={20} />
          2. Ask Questions
        </h2>
        {isDocActive && (
          <span className="badge badge-success">
            <Check size={12} /> Ready for Queries
          </span>
        )}
      </div>

      {!isDocActive && (
        <div className="overlay-notice">
          <BookOpen size={24} />
          <p>Please upload a PDF document above to start asking questions.</p>
        </div>
      )}

      <div className="quick-prompts">
        <span className="prompt-label">Quick Prompts:</span>
        <div className="prompt-chips">
          {QUICK_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              className="chip"
              disabled={!isDocActive || isLoading}
              onClick={() => handleQuickPrompt(prompt)}
            >
              <Sparkles size={12} />
              {prompt}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="ask-form">
        <div className="input-group">
          <textarea
            className="text-input"
            rows={2}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={isDocActive ? "Ask anything about the uploaded PDF..." : "Upload a PDF first..."}
            disabled={!isDocActive || isLoading}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />
          <button
            type="submit"
            className="btn btn-primary send-btn"
            disabled={!isDocActive || !question.trim() || isLoading}
          >
            {isLoading ? <Loader2 className="spinner" size={18} /> : <Send size={18} />}
            <span>Ask</span>
          </button>
        </div>
      </form>

      {error && (
        <div className="alert alert-error">
          <span>{error}</span>
        </div>
      )}

      <div className="history-list">
        {chatHistory.length === 0 && isDocActive && !isLoading && (
          <div className="empty-history">
            <p>No questions asked yet. Type a question or click a quick prompt above!</p>
          </div>
        )}

        {isLoading && (
          <div className="answer-card loading-card">
            <div className="answer-header">
              <Loader2 className="spinner" size={16} />
              <span>Retrieving relevant chunks & formulating grounded answer...</span>
            </div>
          </div>
        )}

        {chatHistory.map((item) => (
          <div key={item.id} className="answer-card">
            <div className="query-row">
              <span className="query-badge">Q:</span>
              <span className="query-text">{item.question}</span>
              <span className="query-time">{item.timestamp}</span>
            </div>

            <div className="answer-body">
              <div className="answer-content">
                <FormattedAnswer text={item.answer} />
              </div>
            </div>

            {item.sources && item.sources.length > 0 && (
              <div className="sources-container">
                <div className="sources-title">
                  <Bookmark size={14} />
                  <span>Source References ({item.sources.length})</span>
                </div>
                <div className="sources-grid">
                  {item.sources.map((src, sIdx) => (
                    <div key={sIdx} className="source-card">
                      <div className="source-meta">
                        <span className="page-pill">Page {src.page}</span>
                        <span className="score-pill">
                          {Math.round(src.score * 100)}% match
                        </span>
                      </div>
                      <div className="source-snippet">"{src.snippet}"</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
