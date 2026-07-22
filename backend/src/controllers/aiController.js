import { asyncHandler } from '../utils/asyncHandler.js';

export const predictSign = asyncHandler(async (req, res) => {
  res.json({ sign: 'HELP', confidence: 0.95, provider: 'placeholder' });
});

export const translate = asyncHandler(async (req, res) => {
  const { text = '', targetLanguage = 'en' } = req.body;
  if (!text.trim()) {
    return res.json({ sourceText: text, translatedText: '', targetLanguage });
  }
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${targetLanguage}&dt=t&q=${encodeURIComponent(text)}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Google Translate returned status ${response.status}`);
    }
    const data = await response.json();
    const translatedText = data[0].map((item) => item[0]).join('');
    res.json({ sourceText: text, translatedText, targetLanguage, provider: 'google-gtx' });
  } catch (error) {
    res.status(500).json({ message: 'Translation failed: ' + error.message });
  }
});

export const speechToText = asyncHandler(async (req, res) => {
  res.json({ text: '', provider: 'placeholder' });
});

export const textToSpeech = asyncHandler(async (req, res) => {
  res.json({ audioUrl: null, provider: 'placeholder' });
});
