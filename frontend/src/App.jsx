import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Cpu, Sparkles, BookOpen, ChevronRight, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

const App = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your CUDA-Q Assistant. How can I help you with quantum-classical programming today?' }
  ]);
  const [loading, setLoading] = useState(false);
  const [selectedSources, setSelectedSources] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input }),
      });

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources 
      }]);
      
      if (data.sources && data.sources.length > 0) {
        setSelectedSources(data.sources);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please make sure the backend server is running.',
        isError: true 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar - Desktop */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Cpu className="logo-icon" />
            <span>CUDA-Q</span>
          </div>
        </div>
        
        <div className="sidebar-content">
          <div className="section-label">SOURCES</div>
          <div className="sources-list">
            {selectedSources.length > 0 ? (
              selectedSources.map((source, i) => (
                <motion.div 
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  key={i} 
                  className="source-item"
                >
                  <BookOpen size={14} className="source-icon" />
                  <div className="source-info">
                    <span className="source-name">{source.source}</span>
                    <span className="source-file">{source.chunk_file}</span>
                  </div>
                  <ChevronRight size={14} className="source-arrow" />
                </motion.div>
              ))
            ) : (
              <div className="sources-empty">
                Ask a question to see retrieved context sources
              </div>
            )}
          </div>
        </div>
        
        <div className="sidebar-footer">
          <div className="status-badge">
            <div className="status-dot"></div>
            <span>System Online</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="header-title">
            <h1>Assistant</h1>
            <div className="model-badge">
              <Sparkles size={12} />
              <span>Qwen 3 14B RAG</span>
            </div>
          </div>
        </header>

        <div className="messages-container">
          <AnimatePresence initial={false}>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`message-wrapper ${m.role}`}
              >
                <div className={`message-avatar ${m.role}`}>
                  {m.role === 'assistant' ? <Bot size={20} /> : <User size={20} />}
                </div>
                <div className="message-content-container">
                  <div className="message-bubble">
                    {m.content}
                  </div>
                  {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                    <div className="message-meta">
                      Found in {m.sources.length} sources
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {loading && (
            <div className="message-wrapper assistant">
              <div className="message-avatar assistant">
                <Bot size={20} />
              </div>
              <div className="message-bubble typing">
                <Loader2 size={16} className="spin" />
                <span>Processing quantum state...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <footer className="chat-input-area">
          <form onSubmit={handleSubmit} className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about CUDA-Q kernels, types, or syntax..."
              autoFocus
            />
            <button type="submit" disabled={!input.trim() || loading}>
              <Send size={18} />
            </button>
          </form>
          <div className="footer-note">
            Powered by NVIDIA CUDA-Q and LangChain
          </div>
        </footer>
      </main>
    </div>
  );
};

export default App;
