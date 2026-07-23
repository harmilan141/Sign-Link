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
  const { text = '', targetLanguage = 'hi' } = req.body;
  if (!text.trim()) {
    return res.status(400).json({ message: 'Text is required for TTS' });
  }

  const lang = targetLanguage.split('-')[0].toLowerCase();
  console.log(`[TTS Request] Synthesizing text into voice for language '${lang}': "${text}"`);

  // Provider 1: Google Translate GTX endpoint (supports hi, pa, en)
  try {
    const gtxUrl = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodeURIComponent(text)}&tl=${lang}&client=gtx`;
    const response = await fetch(gtxUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://translate.google.com/'
      }
    });

    if (response.ok) {
      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const dataUrl = `data:audio/mp3;base64,${buffer.toString('base64')}`;
      console.log(`[TTS Success] Generated audio via Google GTX for language '${lang}' (${buffer.length} bytes)`);
      return res.json({ audioUrl: dataUrl, provider: 'google-gtx', lang });
    }
    console.warn(`[TTS Provider 1] Google GTX returned status ${response.status}`);
  } catch (err) {
    console.warn(`[TTS Provider 1 Error] ${err.message}`);
  }

  // Provider 2: SoundOfText API fallback
  try {
    const voiceCode = lang === 'hi' ? 'hi-IN' : lang === 'pa' ? 'pa-IN' : 'en-US';
    const sotRes = await fetch('https://soundoftext.com/api/s2/sounds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine: 'Google', data: { text, voice: voiceCode } })
    });

    if (sotRes.ok) {
      const sotData = await sotRes.json();
      if (sotData.success && sotData.id) {
        const soundUrl = `https://soundoftext.com/static/sounds/${sotData.id}.mp3`;
        const audioRes = await fetch(soundUrl);
        if (audioRes.ok) {
          const buffer = Buffer.from(await audioRes.arrayBuffer());
          const dataUrl = `data:audio/mp3;base64,${buffer.toString('base64')}`;
          console.log(`[TTS Success] Generated audio via SoundOfText for language '${lang}' (${buffer.length} bytes)`);
          return res.json({ audioUrl: dataUrl, provider: 'soundoftext', lang });
        }
      }
    }
  } catch (err) {
    console.warn(`[TTS Provider 2 Error] ${err.message}`);
  }

  res.status(500).json({ message: 'All TTS voice providers failed to synthesize audio.' });
});
