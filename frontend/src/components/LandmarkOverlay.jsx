import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from 'react';

const handConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

const faceSampleStep = 8;

function resizeCanvasToDetection(canvas, detection) {
  const width = detection?.image?.width || 640;
  const height = detection?.image?.height || 480;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return { width, height };
}

function point(landmark, width, height) {
  return {
    x: landmark.x * width,
    y: landmark.y * height
  };
}

export const LandmarkOverlay = forwardRef(function LandmarkOverlay(props, ref) {
  const { className } = props;
  const canvasRef = useRef(null);
  const statusRef = useRef('Landmark service idle');
  const [status, setStatus] = useState('Landmark service idle');

  const clear = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const draw = useCallback((detection) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context || !detection) {
      clear();
      return;
    }

    const { width, height } = resizeCanvasToDetection(canvas, detection);
    context.clearRect(0, 0, width, height);
    context.lineCap = 'round';
    context.lineJoin = 'round';

    for (const hand of detection.hands || []) {
      context.strokeStyle = '#4cbf9f';
      context.lineWidth = 3;
      context.beginPath();
      for (const [from, to] of handConnections) {
        const a = hand.landmarks[from];
        const b = hand.landmarks[to];
        if (!a || !b) continue;
        const start = point(a, width, height);
        const end = point(b, width, height);
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
      }
      context.stroke();

      for (const landmark of hand.landmarks) {
        const current = point(landmark, width, height);
        context.beginPath();
        context.arc(current.x, current.y, 4, 0, Math.PI * 2);
        context.fillStyle = '#f1b84b';
        context.fill();
        context.strokeStyle = '#17202a';
        context.lineWidth = 1;
        context.stroke();
      }
    }

    // Face landmarks rendering loop has been disabled to eliminate canvas rendering lag
    // and keep visualization focused purely on hand movements for sign language.
  }, [clear]);

  const updateStatus = useCallback((nextStatus) => {
    if (statusRef.current === nextStatus) return;
    statusRef.current = nextStatus;
    setStatus(nextStatus);
  }, []);

  useImperativeHandle(ref, () => ({
    clear,
    draw,
    setStatus: updateStatus
  }), [clear, draw, updateStatus]);

  return (
    <div className={`landmark-layer ${className || ''}`}>
      <canvas ref={canvasRef} aria-hidden="true" />
      <div className="detection-badge">{status}</div>
    </div>
  );
});
