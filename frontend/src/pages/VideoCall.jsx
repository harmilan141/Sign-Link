import { Maximize, Mic, MicOff, PhoneOff, Video, VideoOff, Play, Square, Volume2, VolumeX, Sparkles, ArrowLeftRight } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { LandmarkOverlay } from '../components/LandmarkOverlay.jsx';
import { useAuth } from '../contexts/AuthContext.jsx';
import { useI18n } from '../contexts/I18nContext.jsx';
import { detectLandmarks } from '../services/landmarks.js';
import { useTextToSpeech, useSpeechToText } from '../hooks/useSpeech.js';
import { startSession, sendSessionFrame, stopSession } from '../services/translator.js';
import { api } from '../services/api.js';

const rtcConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const cameraConstraints = {
  width: { ideal: 640 },
  height: { ideal: 480 }
};
const captureSize = { width: 480, height: 360 };
const detectionIntervalMs = 150;
const detectionJpegQuality = 0.55;

function logPeerState(pc) {
  console.log('ICE state:', pc.iceConnectionState);
  console.log('Connection state:', pc.connectionState);
  console.log('Signaling state:', pc.signalingState);
}

export function VideoCall() {
  const { peerId } = useParams();
  const { socket, user } = useAuth();
  const { t } = useI18n();
  const { speak } = useTextToSpeech();

  const [receiverId, setReceiverId] = useState(peerId || '');
  const [callId, setCallId] = useState(null);
  const [status, setStatus] = useState('Ready');
  const [commMode, setCommMode] = useState(null); // 'Hearing' or 'DeafMute'
  const [isSwapped, setIsSwapped] = useState(false); // swap local <-> remote PiP
  const [autoTTS, setAutoTTS] = useState(true);
  const autoTTSRef = useRef(true);
  useEffect(() => {
    autoTTSRef.current = autoTTS;
  }, [autoTTS]);
  const [speakingIdx, setSpeakingIdx] = useState(null);
  const [typedMessage, setTypedMessage] = useState('');
  const [localLiveCaption, setLocalLiveCaption] = useState('');
  const [remoteLiveCaption, setRemoteLiveCaption] = useState('');
  const [audioOn, setAudioOn] = useState(true);
  const [videoOn, setVideoOn] = useState(true);
  const [speakerOn, setSpeakerOn] = useState(true);
  const [seconds, setSeconds] = useState(0);
  const [mediaReady, setMediaReady] = useState(false);

  // AI Translation States
  const [translating, setTranslating] = useState(false);
  const [loadingTranslation, setLoadingTranslation] = useState(false);
  const [liveTranslationText, setLiveTranslationText] = useState('');
  const [remoteTranslationText, setRemoteTranslationText] = useState('');
  const [autoTranslate, setAutoTranslate] = useState(true);
  const [callHistory, setCallHistory] = useState([]);

  const { listening, start: startSTT, stop: stopSTT } = useSpeechToText({
    continuous: true,
    onResult: (text, isFinal) => {
      if (!isFinal) {
        setLocalLiveCaption(text);
        if (receiverId) {
          socket?.emit('live-caption', { receiverId: Number(receiverId), text });
        }
      } else {
        setLocalLiveCaption('');
        const newEntry = {
          sender: 'me',
          text,
          inputMethod: 'voice',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setCallHistory((prev) => [...prev, newEntry]);
        if (receiverId) {
          socket?.emit('translation', {
            receiverId: Number(receiverId),
            text,
            inputMethod: 'voice'
          });
        }
      }
    }
  });

  function selectMode(mode) {
    setCommMode(mode);
    if (mode === 'Hearing') {
      startSTT();
    } else if (mode === 'DeafMute') {
      if (localStream.current) {
        localStream.current.getAudioTracks().forEach((track) => {
          track.enabled = false;
        });
      }
      setAudioOn(false);
    }
  }

  function handleSpeak(text, idx) {
    if (!('speechSynthesis' in window)) return;
    if (speakingIdx === idx && idx !== -1) {
      window.speechSynthesis.cancel();
      setSpeakingIdx(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.onstart = () => { if (idx !== -1) setSpeakingIdx(idx); };
    utterance.onend = () => setSpeakingIdx(null);
    utterance.onerror = () => setSpeakingIdx(null);
    window.speechSynthesis.speak(utterance);
  }

  async function handleTranslateMessage(idx, text, targetLang) {
    setCallHistory((prev) => prev.map((item, i) => {
      if (i === idx) {
        return { ...item, translating: true, targetLanguage: targetLang, error: null };
      }
      return item;
    }));

    try {
      const { data } = await api.post('/ai/translate', { text, targetLanguage: targetLang });
      setCallHistory((prev) => prev.map((item, i) => {
        if (i === idx) {
          return { ...item, translating: false, translation: data.translatedText };
        }
        return item;
      }));
    } catch (error) {
      const errMsg = error.response?.data?.message || 'Translation failed';
      setCallHistory((prev) => prev.map((item, i) => {
        if (i === idx) {
          return { ...item, translating: false, error: errMsg };
        }
        return item;
      }));
    }
  }

  function handleSendTypedMessage(e) {
    e.preventDefault();
    if (!typedMessage.trim()) return;
    const newEntry = {
      sender: 'me',
      text: typedMessage,
      inputMethod: 'typed',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setCallHistory((prev) => [...prev, newEntry]);
    if (receiverId) {
      socket?.emit('translation', {
        receiverId: Number(receiverId),
        text: typedMessage,
        inputMethod: 'typed'
      });
    }
    setTypedMessage('');
  }

  function getAttributionLabel(item) {
    const senderStr = item.sender === 'me' ? 'Me' : 'Remote';
    let methodStr = 'Text Message';
    if (item.inputMethod === 'voice') methodStr = 'Voice Caption';
    if (item.inputMethod === 'sign') methodStr = 'Sign Translation';
    return `${senderStr} (${methodStr})`;
  }

  const localVideo = useRef(null);
  const remoteVideo = useRef(null);
  const captureCanvas = useRef(null);
  const localLandmarkOverlay = useRef(null);
  const remoteLandmarkOverlay = useRef(null);
  const localStream = useRef(null);
  const mediaRequest = useRef(null);
  const peer = useRef(null);
  const callIdRef = useRef(null);
  const pendingIceCandidates = useRef([]);
  const detectionAbort = useRef(null);
  const historyFeedRef = useRef(null);

  // Auto Trigger Refs
  const translatingRef = useRef(false);
  const activeSessionIdRef = useRef(null);
  const idleFramesRef = useRef(0);
  const autoTranslateRef = useRef(true);
  const speakerOnRef = useRef(true);

  const detectionMetrics = useRef({
    droppedFrames: 0,
    lastLogAt: performance.now(),
    processedFrames: 0,
    totalDetectionMs: 0
  });

  useEffect(() => setReceiverId(peerId || ''), [peerId]);

  useEffect(() => {
    callIdRef.current = callId;
  }, [callId]);

  useEffect(() => {
    autoTranslateRef.current = autoTranslate;
  }, [autoTranslate]);

  useEffect(() => {
    speakerOnRef.current = speakerOn;
  }, [speakerOn]);

  useEffect(() => {
    let timer;
    if (status === 'Connected') timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [status]);

  // Scroll to bottom on new history entries
  useEffect(() => {
    if (historyFeedRef.current) {
      historyFeedRef.current.scrollTop = historyFeedRef.current.scrollHeight;
    }
  }, [callHistory]);

  // Frame Capture / Landmark Detection & Translation Loop
  useEffect(() => {
    if (commMode !== 'DeafMute') {
      localLandmarkOverlay.current?.clear();
      localLandmarkOverlay.current?.setStatus('Landmark service idle');
      return;
    }
    if (!mediaReady || !localStream.current || !videoOn) {
      localLandmarkOverlay.current?.clear();
      localLandmarkOverlay.current?.setStatus('Landmark service idle');
      return;
    }

    let cancelled = false;
    let busy = false;
    let firstDetectionLogged = false;
    const timer = setInterval(async () => {
      if (busy) {
        detectionMetrics.current.droppedFrames += 1;
        return;
      }
      if (!localVideo.current || !captureCanvas.current) return;
      if (!localVideo.current.videoWidth || !localVideo.current.videoHeight) return;

      busy = true;
      const startedAt = performance.now();
      detectionAbort.current = new AbortController();
      try {
        const canvas = captureCanvas.current;
        if (canvas.width !== captureSize.width) canvas.width = captureSize.width;
        if (canvas.height !== captureSize.height) canvas.height = captureSize.height;
        const context = canvas.getContext('2d', { alpha: false, desynchronized: true });
        context.drawImage(localVideo.current, 0, 0, captureSize.width, captureSize.height);

        const jpegData = canvas.toDataURL('image/jpeg', detectionJpegQuality);
        
        let result;
        if (translatingRef.current && activeSessionIdRef.current) {
          result = await sendSessionFrame(
            activeSessionIdRef.current,
            jpegData,
            { signal: detectionAbort.current.signal }
          );
        } else {
          result = await detectLandmarks(
            jpegData,
            { signal: detectionAbort.current.signal }
          );
        }

        if (!cancelled) {
          const elapsed = performance.now() - startedAt;
          const metrics = detectionMetrics.current;
          metrics.processedFrames += 1;
          metrics.totalDetectionMs += elapsed;
          localLandmarkOverlay.current?.draw(result);

          // Real-time sharing of landmarks with remote peer
          if (receiverId && status === 'Connected') {
            socket?.emit('peer-landmarks', {
              receiverId: Number(receiverId),
              landmarks: result
            });
          }

          const handsCount = result.status.handsDetected || 0;

          // Intelligent Auto-Translation Trigger Logic
          if (autoTranslateRef.current) {
            if (handsCount > 0) {
              idleFramesRef.current = 0;
              if (!translatingRef.current) {
                console.log('[AI Auto-Trigger] Hands detected. Starting translation session...');
                try {
                  const sessionId = await startSession();
                  activeSessionIdRef.current = sessionId;
                  translatingRef.current = true;
                  setTranslating(true);
                  setLiveTranslationText('');
                  await sendSessionFrame(sessionId, jpegData);
                } catch (err) {
                  console.error('[AI Auto-Trigger] Failed to start session', err);
                }
              }
            } else {
              if (translatingRef.current) {
                idleFramesRef.current += 1;
                if (idleFramesRef.current >= 13) {
                  console.log('[AI Auto-Trigger] No hands detected for 2 seconds. Finalizing...');
                  try {
                    const sessionId = activeSessionIdRef.current;
                    translatingRef.current = false;
                    setTranslating(false);
                    activeSessionIdRef.current = null;
                    setLoadingTranslation(true);
                    
                    const translation = await stopSession(sessionId);
                    if (translation && 
                        !translation.includes("No hands detected") && 
                        !translation.includes("Gesture too short")) {
                      setLiveTranslationText(translation);
                      
                      const newEntry = {
                        sender: 'me',
                        text: translation,
                        inputMethod: 'sign',
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                      };
                      setCallHistory((prev) => [...prev, newEntry]);

                      if (receiverId) {
                        socket?.emit('translation', {
                          receiverId: Number(receiverId),
                          text: translation,
                          inputMethod: 'sign'
                        });
                      }
                    }
                  } catch (err) {
                    console.error('[AI Auto-Trigger] Failed to stop session', err);
                  } finally {
                    setLoadingTranslation(false);
                  }
                }
              }
            }
          }

          const statusText = translatingRef.current
            ? `AI Translation Active (Hands: ${handsCount})`
            : `AI Idle (Waiting for signs...)`;
          localLandmarkOverlay.current?.setStatus(statusText);

          if (!firstDetectionLogged) {
            firstDetectionLogged = true;
            console.log('[MediaPipe] First response time:', Math.round(elapsed), 'ms');
          }

          const now = performance.now();
          if (now - metrics.lastLogAt >= 5000) {
            metrics.droppedFrames = 0;
            metrics.lastLogAt = now;
            metrics.processedFrames = 0;
            metrics.totalDetectionMs = 0;
          }
        }
      } catch (error) {
        if (!cancelled && error.name !== 'CanceledError' && error.name !== 'AbortError') {
          localLandmarkOverlay.current?.setStatus('Landmark service unavailable');
        }
      } finally {
        detectionAbort.current = null;
        busy = false;
      }
    }, detectionIntervalMs);

    return () => {
      cancelled = true;
      detectionAbort.current?.abort();
      clearInterval(timer);
      localLandmarkOverlay.current?.clear();
    };
  }, [mediaReady, videoOn, commMode]);

  // Socket Signaling & Real-time Translation listeners
  useEffect(() => {
    if (!socket) return;
    socket.on('incoming-call', handleIncomingCall);
    socket.on('call-accepted', handleCallAccepted);
    socket.on('call-rejected', () => {
      console.log('[WebRTC] Call rejected');
      setStatus('Rejected');
    });
    socket.on('answer', handleAnswer);
    socket.on('ice-candidate', handleIceCandidate);
    socket.on('call-ended', endLocalCall);

    // Socket translation listener
    socket.on('translation', ({ text, inputMethod }) => {
      console.log('[WebRTC] Received remote translation:', text, inputMethod);
      setRemoteLiveCaption('');
      const newEntry = {
        sender: 'remote',
        text: text,
        inputMethod: inputMethod || 'sign',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setCallHistory((prev) => [...prev, newEntry]);
      setRemoteTranslationText(text);

      // Auto-play TTS if enabled and input method is sign language
      if (autoTTSRef.current && (inputMethod === 'sign' || !inputMethod)) {
        handleSpeak(text, -1);
      }
    });

    socket.on('live-caption', ({ text }) => {
      setRemoteLiveCaption(text);
    });

    // Receive and display remote landmarks
    let remoteLandmarkTimeout = null;
    socket.on('peer-landmarks', ({ landmarks }) => {
      remoteLandmarkOverlay.current?.draw(landmarks);
      if (remoteLandmarkTimeout) clearTimeout(remoteLandmarkTimeout);
      remoteLandmarkTimeout = setTimeout(() => {
        remoteLandmarkOverlay.current?.clear();
      }, 400);
    });

    return () => {
      socket.off('incoming-call', handleIncomingCall);
      socket.off('call-accepted', handleCallAccepted);
      socket.off('call-rejected');
      socket.off('answer', handleAnswer);
      socket.off('ice-candidate', handleIceCandidate);
      socket.off('call-ended');
      socket.off('translation');
      socket.off('live-caption');
      socket.off('peer-landmarks');
      if (remoteLandmarkTimeout) clearTimeout(remoteLandmarkTimeout);
    };
  }, [socket]);

  // Manual AI Translation Handler (used only if autoTranslate is false)
  async function toggleTranslation() {
    if (autoTranslate) return;

    if (!translatingRef.current) {
      try {
        setLiveTranslationText('');
        localLandmarkOverlay.current?.setStatus('Initializing AI session...');
        const sessionId = await startSession();
        activeSessionIdRef.current = sessionId;
        translatingRef.current = true;
        setTranslating(true);
        localLandmarkOverlay.current?.setStatus('Translating signs... Start signing!');
      } catch (err) {
        console.error('[WebRTC Translate] Failed to start translation', err);
        localLandmarkOverlay.current?.setStatus('Failed to start AI translation');
      }
    } else {
      try {
        const sessionId = activeSessionIdRef.current;
        translatingRef.current = false;
        setTranslating(false);
        activeSessionIdRef.current = null;
        setLoadingTranslation(true);
        
        localLandmarkOverlay.current?.setStatus('Processing final translation...');
        const translation = await stopSession(sessionId);
        if (translation && 
            !translation.includes("No hands detected") && 
            !translation.includes("Gesture too short")) {
          setLiveTranslationText(translation);
          
          const newEntry = {
            sender: 'me',
            text: translation,
            inputMethod: 'sign',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          };
          setCallHistory((prev) => [...prev, newEntry]);

          if (receiverId && translation) {
            socket?.emit('translation', {
              receiverId: Number(receiverId),
              text: translation,
              inputMethod: 'sign'
            });
          }
        }
      } catch (err) {
        console.error('[WebRTC Translate] Failed to stop translation', err);
        setLiveTranslationText('Error processing translation.');
      } finally {
        setLoadingTranslation(false);
        localLandmarkOverlay.current?.clear();
        localLandmarkOverlay.current?.setStatus('Translation complete');
      }
    }
  }

  async function ensureMedia(executionPath) {
    if (localStream.current) {
      if (localVideo.current && localVideo.current.srcObject !== localStream.current) {
        localVideo.current.srcObject = localStream.current;
      }
      return localStream.current;
    }

    if (mediaRequest.current) return mediaRequest.current;

    mediaRequest.current = navigator.mediaDevices
      .getUserMedia({ video: cameraConstraints, audio: true })
      .then((stream) => {
        localStream.current = stream;
        if (localVideo.current) localVideo.current.srcObject = stream;
        setMediaReady(true);
        return stream;
      })
      .finally(() => {
        mediaRequest.current = null;
      });

    await mediaRequest.current;
    return localStream.current;
  }

  function attachPeerDiagnostics(pc) {
    pc.oniceconnectionstatechange = () => logPeerState(pc);
    pc.onconnectionstatechange = () => logPeerState(pc);
    pc.onsignalingstatechange = () => logPeerState(pc);
  }

  async function flushPendingIceCandidates(pc) {
    if (!pc.remoteDescription) return;

    while (pendingIceCandidates.current.length) {
      const candidate = pendingIceCandidates.current.shift();
      await pc.addIceCandidate(candidate);
    }
  }

  async function addIceCandidateSafely(candidate) {
    const pc = peer.current;
    if (!pc || !pc.remoteDescription) {
      pendingIceCandidates.current.push(candidate);
      return;
    }
    await pc.addIceCandidate(candidate);
  }

  async function createPeer(targetUserId, executionPath) {
    const stream = await ensureMedia(executionPath);
    const pc = new RTCPeerConnection(rtcConfig);
    attachPeerDiagnostics(pc);

    stream.getTracks().forEach((track) => pc.addTrack(track, stream));

    pc.ontrack = (event) => {
      if (remoteVideo.current && event.streams[0]) {
        remoteVideo.current.srcObject = event.streams[0];
        remoteVideo.current.play?.().catch(() => {});
      }
    };

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        socket?.emit('ice-candidate', {
          receiverId: Number(targetUserId),
          callId: callIdRef.current,
          candidate: event.candidate
        });
      }
    };
    peer.current = pc;
    return pc;
  }

  async function handleCallAccepted({ answer, callId: nextCallId }) {
    callIdRef.current = nextCallId;
    setCallId(nextCallId);

    if (!peer.current) return;
    await peer.current.setRemoteDescription(answer);
    await flushPendingIceCandidates(peer.current);
    setStatus('Connected');
  }

  async function handleAnswer({ answer }) {
    if (!peer.current) return;
    await peer.current.setRemoteDescription(answer);
    await flushPendingIceCandidates(peer.current);
    setStatus('Connected');
  }

  async function handleIceCandidate({ candidate }) {
    if (candidate) await addIceCandidateSafely(candidate);
  }

  async function startCall() {
    if (!receiverId) return;
    setSeconds(0);
    setStatus('Calling');
    pendingIceCandidates.current = [];
    const pc = await createPeer(receiverId, 'startCall');
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    socket.emit('incoming-call', { receiverId: Number(receiverId), offer }, (ack) => {
      if (ack?.callId) {
        callIdRef.current = ack.callId;
        setCallId(ack.callId);
      }
    });
  }

  async function handleIncomingCall({ callerId, callId: nextCallId, offer }) {
    setReceiverId(String(callerId));
    callIdRef.current = nextCallId;
    setCallId(nextCallId);
    setSeconds(0);
    setStatus('Incoming');
    pendingIceCandidates.current = [];
    const accepted = window.confirm('Accept incoming video call?');
    if (!accepted) {
      socket.emit('call-rejected', { callerId, callId: nextCallId });
      return setStatus('Rejected');
    }
    const pc = await createPeer(callerId, 'handleIncomingCall');
    await pc.setRemoteDescription(offer);
    await flushPendingIceCandidates(pc);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    socket.emit('call-accepted', { callerId, callId: nextCallId, answer });
    setStatus('Connected');
  }

  function toggleAudio() {
    localStream.current?.getAudioTracks().forEach((track) => (track.enabled = !audioOn));
    setAudioOn(!audioOn);
  }

  function toggleVideo() {
    localStream.current?.getVideoTracks().forEach((track) => (track.enabled = !videoOn));
    setVideoOn(!videoOn);
  }

  function toggleSpeaker() {
    setSpeakerOn((s) => !s);
  }

  function endLocalCall() {
    peer.current?.close();
    peer.current = null;
    mediaRequest.current = null;
    detectionAbort.current?.abort();
    localStream.current?.getTracks().forEach((track) => track.stop());
    localStream.current = null;
    if (localVideo.current) localVideo.current.srcObject = null;
    if (remoteVideo.current) remoteVideo.current.srcObject = null;
    landmarkOverlay.current?.clear();
    landmarkOverlay.current?.setStatus('Landmark service idle');
    
    // Clean up translation states
    if (activeSessionIdRef.current) {
      stopSession(activeSessionIdRef.current).catch(() => {});
      activeSessionIdRef.current = null;
    }
    translatingRef.current = false;
    setTranslating(false);
    setLiveTranslationText('');
    setRemoteTranslationText('');
    setCallHistory([]);
    idleFramesRef.current = 0;

    stopSTT();
    setCommMode(null);
    setLocalLiveCaption('');
    setRemoteLiveCaption('');

    setMediaReady(false);
    setStatus('Ended');
  }

  function endCall() {
    if (receiverId) socket?.emit('call-ended', { receiverId: Number(receiverId), callId });
    endLocalCall();
  }

  return (
    <section className="call-page">
      <div className="call-toolbar">
        <label className="field compact">
          <span>Receiver ID</span>
          <input value={receiverId} onChange={(e) => setReceiverId(e.target.value)} />
        </label>
        <div className="call-status">{status} · {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')}</div>
        <button className="primary-button" onClick={startCall}><Video size={18} />{t('startCall')}</button>
      </div>

      <div className={`call-workspace ${status === 'Connected' ? 'with-sidebar' : ''}`}>
        <div className={`video-grid${isSwapped ? ' swapped' : ''}`}>
          {/* Remote Participant Video: always bound to remoteVideo ref */}
          <video
            ref={remoteVideo}
            autoPlay
            playsInline
            className={isSwapped ? 'local-video' : 'remote-video'}
            onClick={isSwapped ? () => setIsSwapped(false) : undefined}
            aria-label="Remote participant video"
          />

          {/* Local Camera Preview: always bound to localVideo ref */}
          <video
            ref={localVideo}
            autoPlay
            muted={true} /* Always mute local preview element to avoid audio loopback echo */
            playsInline
            className={isSwapped ? 'remote-video' : 'local-video'}
            onClick={!isSwapped ? () => setIsSwapped(true) : undefined}
            aria-label="Your camera preview"
          />

          {/* Local camera landmarks overlay (follows localVideo container location) */}
          <LandmarkOverlay
            ref={localLandmarkOverlay}
            className={isSwapped ? 'landmark-layer-fullscreen' : 'landmark-layer'}
          />
          {/* Remote participant landmarks overlay (follows remoteVideo container location) */}
          <LandmarkOverlay
            ref={remoteLandmarkOverlay}
            className={isSwapped ? 'landmark-layer' : 'landmark-layer-fullscreen'}
          />
          <canvas ref={captureCanvas} className="hidden-canvas" />
          {/* Swap button: bottom-left of PiP overlay */}
          <button
            className="pip-swap-btn"
            onClick={() => setIsSwapped(s => !s)}
            aria-label={isSwapped ? 'Restore layout' : 'Swap video windows'}
            title={isSwapped ? 'Restore layout' : 'Swap: show your camera full-screen'}
          >
            <ArrowLeftRight size={14} />
          </button>
        </div>

        {status === 'Connected' && !commMode && (
          <div className="mode-selection-modal-overlay">
            <div className="mode-selection-modal card">
              <h2>{t('chooseMode') || 'Choose Communication Mode'}</h2>
              <p>Please select your mode for this call. This choice determines your inputs and will lock for the call's duration.</p>
              <div className="mode-options">
                <button className="mode-option-btn hearing" onClick={() => selectMode('Hearing')}>
                  <Volume2 size={24} />
                  <strong>Hearing User</strong>
                  <span>Use voice captioning and text chat. Camera sign detection is disabled.</span>
                </button>
                <button className="mode-option-btn deaf" onClick={() => selectMode('DeafMute')}>
                  <Sparkles size={24} />
                  <strong>Deaf / Mute User</strong>
                  <span>Use sign-to-text translation and text chat. Microphone is muted.</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {status === 'Connected' && commMode && (
          <div className="call-translation-sidebar card">
            <div className="sidebar-header">
              <Sparkles size={16} className="sparkle-icon" />
              <h3>Shared Conversation</h3>
            </div>
            
            {commMode === 'DeafMute' ? (
              <div className="translation-section">
                <h4>My Sign Translation</h4>
                <p className="description">Convert your sign language into text/voice for the other participant.</p>
                
                <div className="toggle-container" style={{ margin: '10px 0' }}>
                  <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={autoTranslate}
                      onChange={(e) => setAutoTranslate(e.target.checked)}
                    />
                    <span>Intelligent Interpreter Mode</span>
                  </label>
                </div>

                {!autoTranslate && (
                  <div className="control-row">
                    {!translating ? (
                      <button className="primary-button btn-small pulse" onClick={toggleTranslation}>
                        <Play size={14} /> Start Translating
                      </button>
                    ) : (
                      <button className="danger-button btn-small" onClick={toggleTranslation}>
                        <Square size={14} /> Stop & Send
                      </button>
                    )}
                  </div>
                )}
                
                {autoTranslate && (
                  <div className="interpreter-status-badge">
                    {translating ? (
                      <span className="badge active-recording">● Actively Translating...</span>
                    ) : (
                      <span className="badge idle">○ Standing by (raise hands to sign)</span>
                    )}
                  </div>
                )}

                {loadingTranslation && (
                  <div className="loading-indicator">Processing signs...</div>
                )}
              </div>
            ) : (
              <div className="translation-section">
                <h4>Voice Captioning Active</h4>
                <p className="description" style={{ margin: 0 }}>Your speech is being transcribed into captions in real time.</p>
                <div className="toggle-container" style={{ marginTop: '10px' }}>
                  <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={autoTTS}
                      onChange={(e) => setAutoTTS(e.target.checked)}
                    />
                    <span>Auto-play Voice (TTS)</span>
                  </label>
                </div>
              </div>
            )}
            
            <div className="translation-history-section" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <h4 style={{ margin: '0 0 10px 0' }}>Conversation Stream</h4>
              <div className="call-history-feed" ref={historyFeedRef}>
                {callHistory.map((item, idx) => {
                  const attribution = getAttributionLabel(item);
                  return (
                    <div key={idx} className={`history-bubble ${item.sender}`} style={{ position: 'relative' }}>
                      <div className="bubble-header">
                        <span className="sender-tag">{attribution}</span>
                        <span className="time-tag">{item.time}</span>
                      </div>
                      
                      <div className="bubble-content-block">
                        <p className="bubble-text">{item.text}</p>
                        
                        {item.translation && (
                          <div className="translation-result">
                            <strong>Translated ({item.targetLanguage === 'hi' ? 'Hindi' : item.targetLanguage === 'pa' ? 'Punjabi' : 'English'}):</strong>
                            <p>{item.translation}</p>
                          </div>
                        )}
                        
                        {item.translating && (
                          <div className="loading-mini">Translating...</div>
                        )}
                        
                        {item.error && (
                          <div className="error-mini">
                            {item.error} · <button onClick={() => handleTranslateMessage(idx, item.text, item.targetLanguage || user?.preferred_language || 'en')}>Retry</button>
                          </div>
                        )}
                      </div>

                      {commMode === 'Hearing' && (
                        <div className="message-controls-mini">
                          <button
                            className={`message-action-btn-mini ${speakingIdx === idx ? 'speaking' : ''}`}
                            onClick={() => handleSpeak(item.translation || item.text, idx)}
                            title={speakingIdx === idx ? 'Stop Speaking' : 'Read Aloud'}
                          >
                            {speakingIdx === idx ? <Square size={12} /> : <Volume2 size={12} />}
                          </button>
                          
                          <select
                            value={item.targetLanguage || user?.preferred_language || 'en'}
                            onChange={(e) => handleTranslateMessage(idx, item.text, e.target.value)}
                            className="lang-select-mini"
                            title="Translate to..."
                          >
                            <option value="en">English</option>
                            <option value="hi">हिन्दी</option>
                            <option value="pa">ਪੰਜਾਬੀ</option>
                          </select>
                        </div>
                      )}
                    </div>
                  );
                })}

                {localLiveCaption && (
                  <div className="history-bubble me live-caption-bubble">
                    <div className="bubble-header">
                      <span className="sender-tag">Me (Speaking...)</span>
                    </div>
                    <p className="bubble-text live-text">{localLiveCaption}</p>
                  </div>
                )}

                {remoteLiveCaption && (
                  <div className="history-bubble remote live-caption-bubble">
                    <div className="bubble-header">
                      <span className="sender-tag">Remote (Speaking...)</span>
                    </div>
                    <p className="bubble-text live-text">{remoteLiveCaption}</p>
                  </div>
                )}

                {callHistory.length === 0 && !localLiveCaption && !remoteLiveCaption && (
                  <p className="empty-feed-msg">No messages yet. Select mode and start communication.</p>
                )}
              </div>

              <form onSubmit={handleSendTypedMessage} className="sidebar-message-input-form">
                <input
                  type="text"
                  placeholder="Type a message..."
                  value={typedMessage}
                  onChange={(e) => setTypedMessage(e.target.value)}
                  className="message-input"
                />
                <button type="submit" className="primary-button send-btn">Send</button>
              </form>
            </div>
          </div>
        )}
      </div>

      <div className="call-controls">
        <button className="icon-button" onClick={toggleAudio} aria-label="Toggle audio">{audioOn ? <Mic /> : <MicOff />}</button>
        <button className="icon-button" onClick={toggleVideo} aria-label="Toggle video">{videoOn ? <Video /> : <VideoOff />}</button>
        {commMode === 'Hearing' && (
          <button className="icon-button" onClick={toggleSpeaker} aria-label="Toggle speaker">{speakerOn ? <Volume2 /> : <VolumeX />}</button>
        )}
        <button className="danger-button" onClick={endCall}><PhoneOff size={18} />{t('end')}</button>
        <button className="icon-button" onClick={() => remoteVideo.current?.requestFullscreen()} aria-label="Fullscreen"><Maximize /></button>
      </div>
    </section>
  );
}