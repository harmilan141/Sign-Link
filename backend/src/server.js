import http from 'http';
import { Server } from 'socket.io';
import { createApp } from './app.js';
import { env } from './config/env.js';
import { pingDatabase } from './config/db.js';
import { registerSockets } from './sockets/index.js';
import { isLocalOrigin } from './utils/isLocalOrigin.js';

const app = createApp();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: (origin, callback) => {
      if (isLocalOrigin(origin)) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true
  }
});

app.set('io', io);
registerSockets(io);

server.listen(env.port, env.host, async () => {
  try {
    await pingDatabase();
    console.log(`Sign Link API running on ${env.host}:${env.port}`);
  } catch (error) {
    console.error('API started, but database connection failed:', error.message);
  }
});
