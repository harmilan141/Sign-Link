import { Router } from 'express';
import { predictSign, speechToText, textToSpeech, translate } from '../controllers/aiController.js';
import { authenticate } from '../middleware/auth.js';

export const aiRouter = Router();

// Public translation & TTS endpoints (supporting both /translate and /ai/translate)
aiRouter.post('/translate', translate);
aiRouter.post('/ai/translate', translate);
aiRouter.post('/text-to-speech', textToSpeech);
aiRouter.post('/ai/text-to-speech', textToSpeech);
aiRouter.post('/tts', textToSpeech);

// Protected endpoints
aiRouter.use(authenticate);
aiRouter.post('/sign/predict', predictSign);
aiRouter.post('/speech-to-text', speechToText);
