import { useEffect, useRef, useState } from 'react';
import { Camera, Volume2, Video, UploadCloud, Play, Square, Trash2, History, Sparkles, Languages } from 'lucide-react';
import { LandmarkOverlay } from '../components/LandmarkOverlay.jsx';
import { useI18n } from '../contexts/I18nContext.jsx';
import { useTextToSpeech } from '../hooks/useSpeech.js';
import { startSession, sendSessionFrame, stopSession, translateVideoFile } from '../services/translator.js';

const cameraConstraints = {
  width: { ideal: 640 },
  height: { ideal: 480 }
};
const captureSize = { width: 320, height: 240 };
const detectionIntervalMs = 150;
const detectionJpegQuality = 0.35;

export function Translator() {
  const { t } = useI18n();
  const { speak, speaking } = useTextToSpeech();

  // Tabs: 'webcam' or 'upload'
  const [activeTab, setActiveTab] = useState('webcam');
  const [loading, setLoading] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem('sign_link_translation_history');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  // Target Language: 'en' (English), 'hi' (Hindi), 'pa' (Punjabi)
  const [targetLang, setTargetLang] = useState('en');

  // Webcam states
  const [webcamActive, setWebcamActive] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [streamReady, setStreamReady] = useState(false);

  const localVideo = useRef(null);
  const captureCanvas = useRef(null);
  const landmarkOverlay = useRef(null);
  const localStream = useRef(null);
  const activeSessionId = useRef(null);
  const timerRef = useRef(null);

  // File upload states
  const [uploadFile, setUploadFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('sign_link_translation_history', JSON.stringify(history));
  }, [history]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Set stream source when video element is mounted
  useEffect(() => {
    if (webcamActive && localStream.current && localVideo.current) {
      localVideo.current.srcObject = localStream.current;
    }
  }, [webcamActive]);

  // Control Webcam Stream
  async function startWebcam() {
    try {
      setLoading(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: cameraConstraints,
        audio: false
      });
      localStream.current = stream;
      setWebcamActive(true);
      setStreamReady(true);
    } catch (err) {
      console.error('[Webcam] Failed to start webcam', err);
      alert('Could not access camera. Please check permissions.');
    } finally {
      setLoading(false);
    }
  }

  function stopWebcam() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (localStream.current) {
      localStream.current.getTracks().forEach((track) => track.stop());
      localStream.current = null;
    }
    if (localVideo.current) {
      localVideo.current.srcObject = null;
    }
    landmarkOverlay.current?.clear();
    landmarkOverlay.current?.setStatus('Landmark service idle');
    setWebcamActive(false);
    setStreamReady(false);
    setTranslating(false);
    activeSessionId.current = null;
  }

  // Handle live translation session
  async function startLiveTranslation() {
    if (!streamReady) return;
    try {
      setTranslating(true);
      setTranslationText('');
      landmarkOverlay.current?.setStatus('Initializing session...');

      // Call API to start translation session on server
      const sessionId = await startSession();
      activeSessionId.current = sessionId;

      landmarkOverlay.current?.setStatus('Translating... Start signing!');

      let busy = false;
      timerRef.current = setInterval(async () => {
        if (busy) return;
        if (!localVideo.current || !captureCanvas.current) return;
        if (!localVideo.current.videoWidth || !localVideo.current.videoHeight) return;

        busy = true;
        try {
          const canvas = captureCanvas.current;
          if (canvas.width !== captureSize.width) canvas.width = captureSize.width;
          if (canvas.height !== captureSize.height) canvas.height = captureSize.height;
          const context = canvas.getContext('2d', { alpha: false, desynchronized: true });
          context.drawImage(localVideo.current, 0, 0, captureSize.width, captureSize.height);

          // Get JPEG Data URL
          const jpegData = canvas.toDataURL('image/jpeg', detectionJpegQuality);
          const result = await sendSessionFrame(activeSessionId.current, jpegData);

          // Draw landmarks on client-side
          landmarkOverlay.current?.draw(result);
          landmarkOverlay.current?.setStatus(
            `Capturing signs: ${result.status.handsDetected} hand(s), ${result.status.facesDetected} face(s)`
          );
        } catch (err) {
          console.error('[Session Frame] Error processing frame', err);
        } finally {
          busy = false;
        }
      }, detectionIntervalMs);
    } catch (err) {
      console.error('[Session Start] Failed to start translation session', err);
      setTranslating(false);
    }
  }

  async function stopLiveTranslation() {
    if (!translating) return;
    setTranslating(false);
    setLoading(true);
    landmarkOverlay.current?.setStatus('Processing translation...');

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    try {
      const translation = await stopSession(activeSessionId.current);
      activeSessionId.current = null;

      handleNewTranslation(translation);
    } catch (err) {
      console.error('[Session Stop] Error getting final translation', err);
      setTranslationText('Error resolving translation.');
    } finally {
      setLoading(false);
      landmarkOverlay.current?.clear();
      landmarkOverlay.current?.setStatus('Translation complete');
    }
  }

  // Handle file uploads
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setUploadFile(e.target.files[0]);
    }
  };

  async function handleFileUpload() {
    if (!uploadFile) return;
    setLoading(true);
    setTranslationText('');
    try {
      const translation = await translateVideoFile(uploadFile);
      handleNewTranslation(translation);
    } catch (err) {
      console.error('[Video File] Failed to translate video', err);
      setTranslationText('Error translating video file. Please check file format.');
    } finally {
      setLoading(false);
    }
  }

  // Translation helpers
  function handleNewTranslation(text) {
    let finalOutput = text;

    // Apply target language translation
    if (targetLang === 'hi') {
      // Mock translated/transliterated values for common signs
      const hiMocks = {
        "what is your name": "आपका नाम क्या है?",
        "where is the train station": "रेलवे स्टेशन कहाँ है?",
        "can you repeat that please": "क्या आप इसे दोहरा सकते हैं?",
        "how can i help you": "मैं आपकी क्या मदद कर सकता हूँ?",
        "are you free today": "क्या आप आज खाली हैं?",
        "please sit down": "कृपया बैठ जाइए"
      };
      finalOutput = hiMocks[text.toLowerCase()] || text;
    } else if (targetLang === 'pa') {
      const paMocks = {
        "what is your name": "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
        "where is the train station": "ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?",
        "can you repeat that please": "ਕੀ ਤੁਸੀਂ ਇਸਨੂੰ ਦੁਹਰਾ ਸਕਦੇ ਹੋ?",
        "how can i help you": "ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "are you free today": "ਕੀ ਤੁਸੀਂ ਅੱਜ ਵਿਹਲੇ ਹੋ?",
        "please sit down": "ਕਿਰਪਾ ਕਰਕੇ ਬੈਠ ਜਾਓ"
      };
      finalOutput = paMocks[text.toLowerCase()] || text;
    }

    setTranslationText(finalOutput);

    // Save item in history list
    const newItem = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      original: text,
      translated: finalOutput,
      lang: targetLang
    };
    setHistory((prev) => [newItem, ...prev].slice(0, 20));

    // Automatically speak the text using TTS if English
    if (targetLang === 'en') {
      speak(finalOutput, 'en-US');
    }
  }

  function handleSpeak() {
    if (!translationText) return;
    const voiceLang = targetLang === 'hi' ? 'hi-IN' : targetLang === 'pa' ? 'pa-IN' : 'en-US';
    speak(translationText, voiceLang);
  }

  function clearHistory() {
    setHistory([]);
  }

  return (
    <section className="translator-page">
      <div className="translator-header-controls">
        <div className="tab-buttons">
          <button
            className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
            onClick={() => { setActiveTab('webcam'); setTranslationText(''); }}
          >
            <Camera size={16} />
            Live Webcam Translation
          </button>
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => { setActiveTab('upload'); setTranslationText(''); stopWebcam(); }}
          >
            <Video size={16} />
            Upload Sign Video
          </button>
        </div>

        <div className="lang-picker-group">
          <Languages size={18} />
          <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)} aria-label="Translate to">
            <option value="en">Translate to English</option>
            <option value="hi">Translate to Hindi (हिन्दी)</option>
            <option value="pa">Translate to Punjabi (ਪੰਜਾਬੀ)</option>
          </select>
        </div>
      </div>

      <div className="translator-body-grid">
        <div className="input-panel card">
          {activeTab === 'webcam' ? (
            <div className="webcam-container">
              <div className="video-viewport">
                {webcamActive ? (
                  <>
                    <video ref={localVideo} autoPlay muted playsInline className="webcam-feed" aria-label="Live camera input" />
                    <LandmarkOverlay ref={landmarkOverlay} />
                    <canvas ref={captureCanvas} className="hidden-canvas" />
                  </>
                ) : (
                  <div className="placeholder-screen">
                    <Camera size={48} className="muted-icon" />
                    <p>Activate your webcam to begin real-time Indian Sign Language Translation.</p>
                    <button className="primary-button" onClick={startWebcam}>Enable Camera</button>
                  </div>
                )}
                {translating && <div className="recording-glow" />}
              </div>

              {webcamActive && (
                <div className="webcam-toolbar">
                  {!translating ? (
                    <button className="primary-button pulse" onClick={startLiveTranslation}>
                      <Play size={16} />
                      Start Translating
                    </button>
                  ) : (
                    <button className="danger-button" onClick={stopLiveTranslation}>
                      <Square size={16} />
                      Stop & Translate
                    </button>
                  )}
                  <button className="ghost-button" onClick={stopWebcam}>Disable Camera</button>
                </div>
              )}
            </div>
          ) : (
            <div className="upload-container">
              <div
                className={`drag-drop-zone ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
              >
                <UploadCloud size={48} className="upload-icon" />
                <p>Drag and drop your Indian Sign Language video file here, or click to browse</p>
                <input
                  type="file"
                  id="file-upload"
                  className="hidden-file-input"
                  accept="video/mp4,video/x-m4v,video/*"
                  onChange={handleFileChange}
                />
                <label htmlFor="file-upload" className="primary-button text-center cursor-pointer">
                  Browse Files
                </label>
                {uploadFile && <p className="file-name-tag">Selected: {uploadFile.name}</p>}
              </div>

              {uploadFile && (
                <div className="upload-actions">
                  <button className="primary-button full-width" onClick={handleFileUpload} disabled={loading}>
                    {loading ? 'Processing Video...' : 'Translate Uploaded Video'}
                  </button>
                  <button className="ghost-button" onClick={() => setUploadFile(null)}>Clear Selection</button>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="output-panel-stack">
          <div className="result-card card">
            <div className="result-header">
              <Sparkles size={18} className="sparkle-icon" />
              <h3>Translation Result</h3>
            </div>
            
            <div className="result-body">
              {loading ? (
                <div className="spinner-loader">
                  <div className="circle-spinner" />
                  <p>AI is translating your signs...</p>
                </div>
              ) : translationText ? (
                <div className="translation-text-wrapper">
                  <p className="translation-result">{translationText}</p>
                  <button
                    className="speak-btn"
                    onClick={handleSpeak}
                    aria-label="Speak translation"
                    disabled={speaking}
                  >
                    <Volume2 size={24} className={speaking ? 'speaking-pulse' : ''} />
                    <span>Listen</span>
                  </button>
                </div>
              ) : (
                <p className="empty-result-msg">No translation available. Start signing or upload a video file.</p>
              )}
            </div>
          </div>

          <div className="history-card card">
            <div className="history-header">
              <div className="title-group">
                <History size={16} />
                <h3>Recent Translations</h3>
              </div>
              {history.length > 0 && (
                <button className="text-danger-button btn-small" onClick={clearHistory}>
                  <Trash2 size={14} />
                  Clear
                </button>
              )}
            </div>

            <div className="history-list">
              {history.length > 0 ? (
                history.map((item) => (
                  <div key={item.id} className="history-item">
                    <div className="history-meta">
                      <span className="timestamp">{item.timestamp}</span>
                      <span className="lang-badge">{item.lang.toUpperCase()}</span>
                    </div>
                    <p className="history-text">{item.translated}</p>
                  </div>
                ))
              ) : (
                <p className="empty-history-msg">Your translation log is empty.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
