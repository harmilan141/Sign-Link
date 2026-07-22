import { authenticateSocket } from '../middleware/auth.js';
import { createCall, finishCall } from '../models/callModel.js';
import { createMessage, markConversationSeen } from '../models/messageModel.js';

const onlineUsers = new Map();

function onlineSnapshot() {
  return [...onlineUsers.entries()].map(([userId, socketIds]) => ({
    userId,
    online: socketIds.size > 0
  }));
}

export function registerSockets(io) {
  io.use(authenticateSocket);

  io.on('connection', (socket) => {
    const userId = socket.user.userId;
    if (!onlineUsers.has(userId)) onlineUsers.set(userId, new Set());
    onlineUsers.get(userId).add(socket.id);
    socket.join(`user:${userId}`);
    socket.emit('presence-snapshot', onlineSnapshot());
    io.emit('user-online', { userId, online: true });

    socket.on('typing', ({ receiverId, isTyping }) => {
      io.to(`user:${receiverId}`).emit('typing', { senderId: userId, isTyping: Boolean(isTyping) });
    });

    socket.on('send-message', async ({ receiverId, messageText }, ack) => {
      try {
        const message = await createMessage(userId, receiverId, messageText);
        socket.emit('message-sent', message);
        io.to(`user:${receiverId}`).emit('message-received', message);
        ack?.({ ok: true, message });
      } catch (error) {
        ack?.({ ok: false, message: error.message });
      }
    });

    socket.on('messages-seen', async ({ peerId }, ack) => {
      try {
        const seenMessages = await markConversationSeen(userId, peerId);
        io.to(`user:${peerId}`).emit('messages-seen', {
          byUserId: userId,
          messageIds: seenMessages.map((message) => message.message_id),
          seenAt: new Date().toISOString()
        });
        ack?.({ ok: true, seenMessages });
      } catch (error) {
        ack?.({ ok: false, message: error.message });
      }
    });

    socket.on('incoming-call', async ({ receiverId, offer }, ack) => {
      try {
        console.log('[Socket.IO] incoming-call received', { callerId: userId, receiverId, hasOffer: Boolean(offer) });
        const callId = await createCall(userId, receiverId);
        io.to(`user:${receiverId}`).emit('incoming-call', { callId, callerId: userId, offer });
        console.log('[Socket.IO] incoming-call forwarded', { callId, callerId: userId, receiverId });
        ack?.({ ok: true, callId });
      } catch (error) {
        ack?.({ ok: false, message: error.message });
      }
    });

    socket.on('call-accepted', ({ callerId, callId, answer }) => {
      console.log('[Socket.IO] call-accepted received and forwarded', {
        callerId,
        receiverId: userId,
        callId,
        hasAnswer: Boolean(answer)
      });
      io.to(`user:${callerId}`).emit('call-accepted', { callId, receiverId: userId, answer });
    });

    socket.on('call-rejected', async ({ callerId, callId }) => {
      if (callId) await finishCall(callId, 'rejected');
      io.to(`user:${callerId}`).emit('call-rejected', { callId, receiverId: userId });
    });

    socket.on('offer', ({ receiverId, callId, offer }) => {
      console.log('[Socket.IO] offer received and forwarded', { callerId: userId, receiverId, callId, hasOffer: Boolean(offer) });
      io.to(`user:${receiverId}`).emit('offer', { callId, callerId: userId, offer });
    });

    socket.on('answer', ({ receiverId, callId, answer }) => {
      console.log('[Socket.IO] answer received and forwarded', { senderId: userId, receiverId, callId, hasAnswer: Boolean(answer) });
      io.to(`user:${receiverId}`).emit('answer', { callId, senderId: userId, answer });
    });

    socket.on('ice-candidate', ({ receiverId, callId, candidate }) => {
      console.log('[Socket.IO] ice-candidate received and forwarded', {
        senderId: userId,
        receiverId,
        callId,
        hasCandidate: Boolean(candidate)
      });
      io.to(`user:${receiverId}`).emit('ice-candidate', { callId, senderId: userId, candidate });
    });

    socket.on('call-ended', async ({ receiverId, callId }) => {
      if (callId) await finishCall(callId, 'completed');
      io.to(`user:${receiverId}`).emit('call-ended', { callId, senderId: userId });
    });

    socket.on('translation', ({ receiverId, text, inputMethod }) => {
      console.log('[Socket.IO] translation forwarded', { senderId: userId, receiverId, text, inputMethod });
      io.to(`user:${receiverId}`).emit('translation', { senderId: userId, text, inputMethod });
    });

    socket.on('live-caption', ({ receiverId, text }) => {
      io.to(`user:${receiverId}`).emit('live-caption', { senderId: userId, text });
    });

    socket.on('peer-landmarks', ({ receiverId, landmarks }) => {
      io.to(`user:${receiverId}`).emit('peer-landmarks', { senderId: userId, landmarks });
    });

    socket.on('disconnect', () => {
      const sockets = onlineUsers.get(userId);
      sockets?.delete(socket.id);
      if (!sockets || sockets.size === 0) {
        onlineUsers.delete(userId);
        io.emit('user-online', { userId, online: false });
      }
    });
  });
}
