import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Send, Bot, User, Cpu, Sparkles, BookOpen, ChevronRight, Loader2, Check, Copy, Zap, MessageSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { prepare, layout } from '@chenglou/pretext';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';
import './App.css';


const CodeBlock = ({ node, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const text = String(children).replace(/\n$/, '');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!match) {
    return <code className="inline-code" {...props}>{children}</code>;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="code-block-container"
    >
      <div className="code-header">
        <div className="code-lang-wrapper">
          <div className="code-dots">
            <span className="dot red"></span>
            <span className="dot yellow"></span>
            <span className="dot green"></span>
          </div>
          <span className="code-lang">{match[1].toUpperCase()}</span>
        </div>
        <button
          className={`copy-button ${copied ? 'copied' : ''}`}
          onClick={handleCopy}
          type="button"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        customStyle={{
          margin: 0,
          background: '#0d0d0f',
          padding: '20px',
          fontSize: '0.9rem',
          lineHeight: '1.6',
          fontFamily: "'Fira Code', 'JetBrains Mono', monospace",
        }}
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    </motion.div>
  );
};

// Pretext-optimized Markdown Container
const OptimizedMarkdown = ({ content, isAssistant }) => {
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    if (containerRef.current) {
      const observer = new ResizeObserver((entries) => {
        for (let entry of entries) {
          setContainerWidth(entry.contentRect.width);
        }
      });
      observer.observe(containerRef.current);
      return () => observer.disconnect();
    }
  }, []);

  // Use Pretext to calculate stable layout metrics
  const layoutMetrics = useMemo(() => {
    if (!content || containerWidth === 0) return null;
    try {
      // Font shorthand matches the CSS: 16px Inter/Outfit
      const prepared = prepare(content, '16px Outfit, sans-serif');
      return layout(prepared, containerWidth, 24); // 24px line-height
    } catch (e) {
      return null;
    }
  }, [content, containerWidth]);

  return (
    <div ref={containerRef} className="markdown-container">
      <ReactMarkdown 
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{ code: CodeBlock }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

const App = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your **CUDA-Q AI Assistant**. I can help you with quantum circuit simulation, GPU-accelerated kernels, and more. How can I assist you today?' }
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

    const assistantMessageId = Date.now();
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      id: assistantMessageId,
      sources: []
    }]);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantAnswer = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const payload = JSON.parse(line);

            if (payload.type === 'sources') {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId ? { ...msg, sources: payload.data } : msg
              ));
              setSelectedSources(payload.data);
            } else if (payload.type === 'token') {
              assistantAnswer += payload.data;
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId ? { ...msg, content: assistantAnswer } : msg
              ));
            } else if (payload.type === 'error') {
              throw new Error(payload.data);
            }
          } catch (err) {
            console.error('JSON Parse Error:', err);
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId ? {
          ...msg,
          content: '⚠️ Error: ' + error.message,
          isError: true
        } : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-section">
            <Cpu className="logo-icon" size={24} />
            <div className="logo-text">
              <span className="brand">CUDA-Q</span>
              <span className="subbrand">RAG ENGINE</span>
            </div>
          </div>
        </div>

        <div className="sidebar-content">
          <div className="section-label">RETRIEVED CONTEXT</div>
          <div className="sources-list">
            <AnimatePresence mode="popLayout">
              {selectedSources.length > 0 ? (
                selectedSources.map((source, i) => (
                  <motion.div 
                    key={i} 
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="source-item"
                  >
                    <BookOpen size={14} className="source-icon" />
                    <div className="source-info">
                      <span className="source-name">{source.source}</span>
                      <span className="source-file">{source.chunk_file}</span>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="sources-empty">
                  <Sparkles size={16} className="empty-icon" />
                  <span>Waiting for query...</span>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="status-badge">
            <div className="status-dot-pulse"></div>
            <span>NVIDIA GPU ACCELERATED</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="header-left">
            <Zap size={20} className="header-icon" />
            <div className="header-title">
              <h1>Quantum Intelligence</h1>
              <span className="model-name">Gemma 4 Enhanced</span>
            </div>
          </div>
          <div className="header-right">
            <div className="model-badge">
              <div className="dot"></div>
              <span>4bit Quantized</span>
            </div>
          </div>
        </header>

        <div className="messages-container">
          {messages.map((m, i) => (
            <motion.div 
              key={i} 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`message-wrapper ${m.role}`}
            >
              <div className={`message-avatar ${m.role}`}>
                {m.role === 'assistant' ? <Bot size={20} /> : <User size={20} />}
              </div>
              <div className="message-content-container">
                <div className="message-bubble">
                  <OptimizedMarkdown 
                    content={m.content} 
                    isAssistant={m.role === 'assistant'} 
                  />
                </div>
              </div>
            </motion.div>
          ))}
          {loading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="message-wrapper assistant"
            >
              <div className="message-avatar assistant"><Bot size={20} /></div>
              <div className="message-bubble typing">
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span>Thinking via CUDA-Q...</span>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <footer className="chat-input-area">
          <form onSubmit={handleSubmit} className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything about CUDA-Q or Quantum Computing..."
            />
            <button type="submit" disabled={!input.trim() || loading}>
              {loading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
};

export default App;