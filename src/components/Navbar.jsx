import React from 'react';
import { Camera, MessageSquare, BookOpen, Activity, Volume2, VolumeX, Sparkles } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, backendStatus, ttsEnabled, setTtsEnabled }) {
  return (
    <header style={{
      width: '100%',
      padding: '16px 24px',
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(9, 13, 22, 0.85)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '16px',
      flexWrap: 'wrap'
    }}>
      {/* Brand Identity */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '22px',
          boxShadow: '0 4px 14px var(--primary-glow)'
        }}>
          🤟
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '18px', fontWeight: '800', letterSpacing: '-0.02em', margin: 0 }}>
              VSL Connect <span style={{ color: 'var(--primary)' }}>AI</span>
            </h1>
            <span className="badge badge-cyan" style={{ fontSize: '10px', padding: '2px 8px' }}>
              Hybrid PyTorch
            </span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0 }}>
            Hệ thống Giao tiếp Hai chiều Ngôn ngữ Ký hiệu Việt Nam
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav style={{
        display: 'flex',
        alignItems: 'center',
        background: 'rgba(17, 24, 39, 0.8)',
        padding: '4px',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-subtle)',
        gap: '4px'
      }}>
        <button
          id="nav-tab-camera"
          onClick={() => setActiveTab('camera')}
          className={`btn ${activeTab === 'camera' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '8px 16px', fontSize: '13px' }}
        >
          <Camera size={16} />
          Camera Nhận Diện (Sign → Text)
        </button>

        <button
          id="nav-tab-text-to-sign"
          onClick={() => setActiveTab('text_to_sign')}
          className={`btn ${activeTab === 'text_to_sign' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '8px 16px', fontSize: '13px' }}
        >
          <MessageSquare size={16} />
          Văn Bản → Ký Hiệu 3D
        </button>

        <button
          id="nav-tab-dictionary"
          onClick={() => setActiveTab('dictionary')}
          className={`btn ${activeTab === 'dictionary' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '8px 16px', fontSize: '13px' }}
        >
          <BookOpen size={16} />
          Từ Điển & Luyện Tập VSL
        </button>
      </nav>

      {/* Action Indicators & System Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* TTS Toggle Button */}
        <button
          onClick={() => setTtsEnabled(!ttsEnabled)}
          className="btn btn-secondary"
          style={{ padding: '8px 12px', fontSize: '12px' }}
          title={ttsEnabled ? "Tự động phát âm tiếng nói (Đang Bật)" : "Tắt phát âm tiếng nói"}
        >
          {ttsEnabled ? (
            <>
              <Volume2 size={15} style={{ color: 'var(--accent-success)' }} />
              <span style={{ color: 'var(--accent-success)' }}>Giọng nói: Bật</span>
            </>
          ) : (
            <>
              <VolumeX size={15} style={{ color: 'var(--text-muted)' }} />
              <span style={{ color: 'var(--text-muted)' }}>Giọng nói: Tắt</span>
            </>
          )}
        </button>

        {/* Backend Online Status Indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid var(--border-subtle)',
          fontSize: '12px'
        }}>
          <div className="pulse-dot" style={{
            backgroundColor: backendStatus === 'online' ? 'var(--accent-success)' : 'var(--accent-warning)',
            boxShadow: backendStatus === 'online' ? '0 0 10px var(--accent-success)' : '0 0 10px var(--accent-warning)'
          }} />
          <span style={{ color: 'var(--text-secondary)' }}>
            {backendStatus === 'online' ? 'FastAPI + PyTorch Sẵn sàng' : 'Chế độ Client Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
