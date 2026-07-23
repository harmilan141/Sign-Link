import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export function useSpeechToText({ onResult, continuous = false } = {}) {
  const [listening, setListening] = useState(false);
  const [supported] = useState(() => Boolean(window.SpeechRecognition || window.webkitSpeechRecognition));
  const recognitionRef = useRef(null);
  const shouldRestartRef = useRef(false);
  const silenceTimerRef = useRef(null);
  const startRef = useRef(null);

  const start = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    shouldRestartRef.current = true;

    // Clean up any existing active recognition instance first
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.stop();
      } catch (e) {}
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = continuous;
    
    recognition.onstart = () => setListening(true);
    recognition.onend = () => {
      setListening(false);
      if (shouldRestartRef.current) {
        // Instantiate a brand new instance to prevent browser "already started" errors
        setTimeout(() => {
          if (shouldRestartRef.current && startRef.current) {
            startRef.current();
          }
        }, 100);
      }
    };
    recognition.onerror = (event) => {
      console.warn("[Speech] Recognition error:", event.error);
      if (event.error === 'no-speech') return; // ignore silence errors
      setListening(false);
    };

    recognition.onresult = (event) => {
      const resultIndex = event.resultIndex;
      const currentResult = event.results[resultIndex];
      const text = (currentResult[0]?.transcript || '').trim();
      const isFinal = currentResult.isFinal || false;
      
      if (text) {
        onResult?.(text, isFinal);

        // Auto-finalize on short silence (1.5 seconds) to reduce latency
        if (!isFinal) {
          if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = setTimeout(() => {
            console.log("[Speech] Silence timeout reached. Finalizing chunk...");
            recognition.stop(); // triggers final result and ends session
          }, 1500);
        } else {
          if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        }
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [onResult, continuous]);

  // Keep startRef updated with the latest start callback
  useEffect(() => {
    startRef.current = start;
  }, [start]);

  const stop = useCallback(() => {
    shouldRestartRef.current = false;
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      } catch (e) {}
      recognitionRef.current = null;
    }
    setListening(false);
  }, []);

  return { listening, supported, start, stop };
}

export function useTextToSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef(null);

  const stop = useCallback(() => {
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current = null;
      } catch (e) {}
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  }, []);

  const speak = useCallback(
    async (text, lang = 'hi') => {
      if (!text || !text.trim()) {
        stop();
        return;
      }
      stop();

      // Normalize lang code (e.g. 'hi-IN' -> 'hi', 'pa-IN' -> 'pa')
      const targetLang = (lang || 'hi').split('-')[0].toLowerCase();

      try {
        setSpeaking(true);
        const res = await fetch('/api/text-to-speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, targetLanguage: targetLang })
        });

        if (res.ok) {
          const data = await res.json();
          if (data.audioUrl) {
            const audio = new Audio(data.audioUrl);
            audioRef.current = audio;
            audio.onended = () => setSpeaking(false);
            audio.onerror = () => {
              setSpeaking(false);
              fallbackSpeechSynthesis(text, targetLang);
            };
            await audio.play();
            return;
          }
        }
      } catch (err) {
        console.warn('[Cloud TTS] Cloud synthesis failed, using browser fallback:', err);
      }

      // Fallback to local SpeechSynthesis if cloud TTS is unreachable
      fallbackSpeechSynthesis(text, targetLang);
    },
    [stop]
  );

  function fallbackSpeechSynthesis(text, lang) {
    if (!('speechSynthesis' in window)) {
      setSpeaking(false);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang === 'hi' ? 'hi-IN' : lang === 'pa' ? 'pa-IN' : 'en-US';
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  return { speak, speaking, stop, supported: true };
}
