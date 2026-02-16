import React, { useState, useRef, useEffect, useCallback } from 'react'
import { api, ChatResponse } from '../api/client'
import ReactMarkdown from 'react-markdown'
import './ChatWidget.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  youtubeLinks?: string[]
  needsEscalation?: boolean
}

export function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Здравствуйте! Чем могу помочь? 💊',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [unread, setUnread] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    if (isOpen) {
      scrollToBottom()
      setUnread(0)
    }
  }, [messages, isOpen, scrollToBottom])

  const sendMessage = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)

    try {
      const response: ChatResponse = await api.sendMessage(trimmed, sessionId || undefined)

      if (!sessionId) setSessionId(response.session_id)

      const botMsg: Message = {
        role: 'assistant',
        content: response.answer,
        youtubeLinks: response.youtube_links,
        needsEscalation: response.needs_escalation,
      }

      setMessages(prev => [...prev, botMsg])

      if (!isOpen) setUnread(prev => prev + 1)

      // Автоэскалация при низкой уверенности
      if (response.needs_escalation && response.session_id) {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content:
              '💡 Я не нашёл точного ответа. Нажмите кнопку ниже, чтобы связаться с оператором, или задайте вопрос иначе.',
          },
        ])
      }
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Ошибка соединения. Попробуйте позже.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleEscalation = async () => {
    if (!sessionId) return
    try {
      const res = await api.createEscalation(sessionId)
      setMessages(prev => [...prev, { role: 'assistant', content: `✅ ${res.message}` }])
    } catch {
      // ignore
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Скрываем виджет на странице /operator
  if (typeof window !== 'undefined' && window.location.pathname.startsWith('/operator')) {
    return null
  }

  return (
    <>
      {/* Кнопка виджета */}
      <button
        className={`widget-toggle ${isOpen ? 'widget-toggle-open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Открыть чат техподдержки"
      >
        {isOpen ? '✕' : '💬'}
        {!isOpen && unread > 0 && <span className="widget-badge">{unread}</span>}
      </button>

      {/* Окно чата */}
      {isOpen && (
        <div className="widget-window">
          <div className="widget-header">
            <span>💊 Техподдержка Фармбазис</span>
            <button onClick={() => setIsOpen(false)}>✕</button>
          </div>

          <div className="widget-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`widget-msg widget-msg-${msg.role}`}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {msg.youtubeLinks?.map((link, j) => (
                  <a key={j} href={link} target="_blank" rel="noopener noreferrer" className="widget-yt-link">
                    📹 Видео-инструкция
                  </a>
                ))}
                {msg.needsEscalation && (
                  <button className="widget-escalation-btn" onClick={handleEscalation}>
                    📞 Связаться с оператором
                  </button>
                )}
              </div>
            ))}
            {loading && (
              <div className="widget-msg widget-msg-assistant widget-typing">
                <span></span><span></span><span></span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="widget-input">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Задайте вопрос..."
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={!input.trim() || loading}>
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  )
}
