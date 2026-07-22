import axios from 'axios';
import { LANDMARK_API_URL } from '../config/network.js';

export const translatorApi = axios.create({
  baseURL: LANDMARK_API_URL
});

export async function startSession() {
  const { data } = await translatorApi.post('/api/translate/session/start');
  return data.session_id;
}

export async function sendSessionFrame(sessionId, image, config = {}) {
  const { data } = await translatorApi.post('/api/translate/session/frame', {
    session_id: sessionId,
    image
  }, config);
  return data;
}

export async function stopSession(sessionId) {
  const { data } = await translatorApi.post('/api/translate/session/stop', {
    session_id: sessionId
  });
  return data.translation;
}

export async function translateVideoFile(file, config = {}) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await translatorApi.post('/api/translate/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    ...config
  });
  return data.translation;
}
