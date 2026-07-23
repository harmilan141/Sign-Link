import { Mic, Send, Volume2, MessageSquare, Video, Search, User } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
import { useI18n } from '../contexts/I18nContext.jsx';
import { api } from '../services/api.js';
import { useSpeechToText, useTextToSpeech } from '../hooks/useSpeech.js';

export function Chat() {
  const { peerId } = useParams();
  const navigate = useNavigate();
  const { socket, user } = useAuth();
  const { t } = useI18n();

  const [receiverId, setReceiverId] = useState(peerId || '');
  const [contacts, setContacts] = useState([]);
  const [searchFilter, setSearchFilter] = useState('');
  const [activeContact, setActiveContact] = useState(null);

  const [messageText, setMessageText] = useState('');
  const [messages, setMessages] = useState([]);
  const [onlineUsers, setOnlineUsers] = useState({});
  const [typing, setTyping] = useState(false);

  const bottomRef = useRef(null);
  const { speak } = useTextToSpeech();
  const speechToText = useSpeechToText({
    onResult: (text) => setMessageText(text)
  });

  // Keep receiverId in sync with route param
  useEffect(() => {
    setReceiverId(peerId || '');
  }, [peerId]);

  // Load contacts/users list
  useEffect(() => {
    async function loadContacts() {
      try {
        const { data } = await api.get('/users/search', { params: { q: searchFilter } });
        setContacts(data.users || []);
      } catch (err) {
        console.error('[Chat] Failed to load contacts:', err);
      }
    }
    const timer = setTimeout(loadContacts, 200);
    return () => clearTimeout(timer);
  }, [searchFilter]);

  // Sync active contact object
  useEffect(() => {
    if (!receiverId) {
      setActiveContact(null);
      return;
    }
    const found = contacts.find((c) => String(c.user_id) === String(receiverId));
    if (found) {
      setActiveContact(found);
    }
  }, [receiverId, contacts]);

  // Load message history when receiverId changes
  useEffect(() => {
    if (!receiverId) {
      setMessages([]);
      return;
    }
    async function loadMessages() {
      try {
        const { data } = await api.get(`/messages/${receiverId}`);
        setMessages(data.messages || []);
        await api.patch(`/messages/${receiverId}/seen`);
        socket?.emit('messages-seen', { peerId: Number(receiverId) });
      } catch (err) {
        console.error('[Chat] Failed to load message history:', err);
      }
    }
    loadMessages();
  }, [receiverId, socket]);

  // Socket listeners for real-time messages & presence
  useEffect(() => {
    if (!socket) return;

    const onMessage = (message) => {
      if (String(message.sender_id) === String(receiverId) || String(message.receiver_id) === String(receiverId)) {
        setMessages((current) => [...current, message]);
      }
      if (String(message.sender_id) === String(receiverId)) {
        api.patch(`/messages/${receiverId}/seen`);
        socket.emit('messages-seen', { peerId: Number(receiverId) });
      }
    };

    const onSeen = ({ messageIds, seenAt }) => {
      setMessages((current) =>
        current.map((message) =>
          messageIds.includes(message.message_id) ? { ...message, seen_at: message.seen_at || seenAt } : message
        )
      );
    };

    const onPresenceSnapshot = (users) => {
      setOnlineUsers(Object.fromEntries(users.map((entry) => [entry.userId, entry.online])));
    };

    const onPresence = ({ userId, online }) => {
      setOnlineUsers((current) => ({ ...current, [userId]: online }));
    };

    const onTyping = ({ senderId, isTyping }) => {
      if (String(senderId) === String(receiverId)) setTyping(isTyping);
    };

    socket.on('message-received', onMessage);
    socket.on('messages-seen', onSeen);
    socket.on('presence-snapshot', onPresenceSnapshot);
    socket.on('user-online', onPresence);
    socket.on('typing', onTyping);

    return () => {
      socket.off('message-received', onMessage);
      socket.off('messages-seen', onSeen);
      socket.off('presence-snapshot', onPresenceSnapshot);
      socket.off('user-online', onPresence);
      socket.off('typing', onTyping);
    };
  }, [receiverId, socket]);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message handler
  async function send(event) {
    event.preventDefault();
    if (!receiverId || !messageText.trim()) return;

    const textToSend = messageText.trim();
    setMessageText('');

    try {
      const payload = { receiverId: Number(receiverId), messageText: textToSend };
      const { data } = await api.post('/messages', payload);
      setMessages((current) => [...current, data.message]);
      socket?.emit('typing', { receiverId: Number(receiverId), isTyping: false });
    } catch (err) {
      console.error('[Chat] Failed to send message:', err);
    }
  }

  function handleSelectContact(userItem) {
    setReceiverId(String(userItem.user_id));
    setActiveContact(userItem);
    navigate(`/chat/${userItem.user_id}`);
  }

  return (
    <section className="chat-layout">
      {/* Sidebar: Contacts List */}
      <aside className="chat-side">
        <div className="contacts-search">
          <input
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            placeholder="Search contacts..."
            aria-label="Search contacts"
          />
        </div>

        <div className="contacts-list">
          {contacts.length > 0 ? (
            contacts.map((c) => {
              const isOnline = onlineUsers[c.user_id];
              const isSelected = String(c.user_id) === String(receiverId);
              return (
                <div
                  key={c.user_id}
                  className={`contact-card ${isSelected ? 'active' : ''}`}
                  onClick={() => handleSelectContact(c)}
                >
                  <div className="contact-avatar">
                    {c.full_name ? c.full_name[0].toUpperCase() : 'U'}
                  </div>
                  <div className="contact-info">
                    <h4>{c.full_name || `@${c.username}`}</h4>
                    <p>@{c.username}</p>
                  </div>
                  <span className={`presence-dot ${isOnline ? 'online' : ''}`} title={isOnline ? 'Online' : 'Offline'} />
                </div>
              );
            })
          ) : (
            <p style={{ fontSize: '12px', color: '#94a3b8', padding: '12px', textAlign: 'center' }}>
              No contacts found.
            </p>
          )}
        </div>
      </aside>

      {/* Main Chat Panel */}
      <div className="chat-panel">
        {receiverId ? (
          <>
            {/* Chat Panel Header */}
            <div className="chat-panel-header">
              <div className="chat-user-meta">
                <div className="contact-avatar">
                  {activeContact?.full_name ? activeContact.full_name[0].toUpperCase() : 'U'}
                </div>
                <div>
                  <h4 style={{ margin: 0, fontSize: '14px', fontWeight: 600 }}>
                    {activeContact?.full_name || `User #${receiverId}`}
                  </h4>
                  <span style={{ fontSize: '12px', color: onlineUsers[Number(receiverId)] ? '#22c55e' : '#94a3b8' }}>
                    {onlineUsers[Number(receiverId)] ? '● Online' : '○ Offline'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <Link className="icon-button" to={`/call/${receiverId}`} title="Start Video Call">
                  <Video size={18} />
                </Link>
              </div>
            </div>

            {/* Messages Feed */}
            <div className="messages">
              {messages.map((message) => {
                const isMine = message.sender_id === user?.user_id;
                return (
                  <div
                    key={message.message_id || `${message.created_at}-${message.message_text}`}
                    className={`message ${isMine ? 'mine' : ''}`}
                  >
                    <p>{message.message_text}</p>
                    <time>
                      {new Date(message.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      {isMine ? ` · ${message.seen_at ? 'Seen' : 'Sent'}` : ''}
                    </time>
                    <button
                      className="speak-button"
                      onClick={() => speak(message.message_text, 'hi')}
                      aria-label="Read message aloud"
                      title="Read aloud"
                      style={{ marginTop: '4px' }}
                    >
                      <Volume2 size={14} />
                    </button>
                  </div>
                );
              })}
              {typing ? <div className="typing">Typing...</div> : null}
              <div ref={bottomRef} />
            </div>

            {/* Composer Bar */}
            <form className="composer" onSubmit={send}>
              <input
                value={messageText}
                onChange={(e) => {
                  setMessageText(e.target.value);
                  if (receiverId) socket?.emit('typing', { receiverId: Number(receiverId), isTyping: true });
                }}
                placeholder="Type a message..."
              />
              {speechToText.supported ? (
                <button
                  className="icon-button"
                  type="button"
                  onClick={speechToText.listening ? speechToText.stop : speechToText.start}
                  aria-label="Speech to text"
                  title={speechToText.listening ? 'Listening...' : 'Voice typing'}
                  style={{ color: speechToText.listening ? '#ef4444' : undefined }}
                >
                  <Mic size={18} />
                </button>
              ) : null}
              <button className="primary-button" type="submit">
                <Send size={18} />
                {t('send') || 'Send'}
              </button>
            </form>
          </>
        ) : (
          /* Empty Chat State */
          <div className="empty-chat-state">
            <MessageSquare size={48} style={{ opacity: 0.4 }} />
            <h3 style={{ margin: 0 }}>Select a Conversation</h3>
            <p style={{ margin: 0, fontSize: '13px', maxWidth: '300px' }}>
              Choose a contact from the list on the left to start messaging in real time.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
