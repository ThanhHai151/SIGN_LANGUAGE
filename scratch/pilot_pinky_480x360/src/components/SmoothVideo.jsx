import React, { useEffect, useRef, useState } from 'react';
import { Loader2, VideoOff, RotateCcw } from 'lucide-react';
import { mediaUrl } from '../services/api';

/**
 * Video mẫu QIPEDC phát mượt:
 *  - phát qua bộ đệm của backend (đã chuyển chỉ mục lên đầu file, cache lâu dài) → hình đầu tiên hiện ngay;
 *  - hiện ảnh bìa + vòng tải cho tới khi đủ dữ liệu phát liền mạch, rồi hiện video mờ dần (không nháy đen);
 *  - backend không tải được → tự chuyển sang link gốc QIPEDC; vẫn lỗi → nút thử lại.
 */
export default function SmoothVideo({ url, poster, style }) {
  const videoRef = useRef(null);
  const [src, setSrc] = useState(() => mediaUrl(url));
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    setSrc(mediaUrl(url));
    setReady(false);
    setFailed(false);
  }, [url]);

  // Bộ đệm backend chưa kịp có video (lần đầu xem) mà quá 3,5 giây → phát thẳng từ QIPEDC
  useEffect(() => {
    if (ready || failed || src === url) return undefined;
    const t = setTimeout(() => setSrc(url), 3500);
    return () => clearTimeout(t);
  }, [src, ready, failed, url]);

  const onError = () => {
    if (src !== url) { setSrc(url); setReady(false); return; } // thử link gốc
    setFailed(true);
  };

  const retry = () => {
    setFailed(false);
    setReady(false);
    setSrc(`${mediaUrl(url)}${mediaUrl(url).includes('?') ? '&' : '?'}r=${Date.now()}`);
  };

  return (
    <div style={{ position: 'relative', width: '100%', aspectRatio: '16 / 9', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000', ...style }}>
      {poster && (
        <img
          src={poster}
          alt=""
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: ready ? 0 : 1, transition: 'opacity .25s ease' }}
        />
      )}
      {!failed && (
        <video
          key={src}
          ref={videoRef}
          src={src}
          poster={poster || undefined}
          preload="auto"
          autoPlay
          loop
          muted
          playsInline
          controls={ready}
          onCanPlayThrough={() => setReady(true)}
          onPlaying={() => { setReady(true); setWaiting(false); }}
          onWaiting={() => setWaiting(true)}
          onCanPlay={() => setWaiting(false)}
          onTimeUpdate={() => { if (waiting) setWaiting(false); }}
          onSeeked={() => setWaiting(false)}
          onError={onError}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: ready ? 1 : 0, transition: 'opacity .3s ease' }}
        />
      )}
      {!failed && (!ready || waiting) && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
          <div style={{ background: 'rgba(5,8,15,0.6)', borderRadius: 999, padding: 10, display: 'flex' }}>
            <Loader2 size={26} className="spin" style={{ color: '#e2e8f0', animation: 'svspin 0.9s linear infinite' }} />
          </div>
        </div>
      )}
      {failed && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', background: 'rgba(5,8,15,0.75)' }}>
          <VideoOff size={26} />
          <span style={{ fontSize: 13 }}>Không tải được video mẫu</span>
          <button className="btn btn-secondary" onClick={retry} style={{ padding: '6px 14px', fontSize: 12 }}>
            <RotateCcw size={13} /> Thử lại
          </button>
        </div>
      )}
      <style>{'@keyframes svspin { to { transform: rotate(360deg); } }'}</style>
    </div>
  );
}
