import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import CameraSignRecognizer from './components/CameraSignRecognizer';
import TextToSignVisualizer from './components/TextToSignVisualizer';
import VSLDictionaryExplorer from './components/VSLDictionaryExplorer';
import { checkBackendHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('camera');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [ttsEnabled, setTtsEnabled] = useState(true);

  // Kiểm tra trạng thái Backend FastAPI
  useEffect(() => {
    async function verifyBackend() {
      const res = await checkBackendHealth();
      if (res && res.status === 'online') {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    }

    verifyBackend();
    const interval = setInterval(verifyBackend, 10000); // kiểm tra mỗi 10s
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header & Thanh điều hướng */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        ttsEnabled={ttsEnabled}
        setTtsEnabled={setTtsEnabled}
      />

      {/* Nội dung theo Tab đang chọn */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'camera' && (
          <CameraSignRecognizer ttsEnabled={ttsEnabled} />
        )}

        {activeTab === 'text_to_sign' && (
          <TextToSignVisualizer ttsEnabled={ttsEnabled} />
        )}

        {activeTab === 'dictionary' && (
          <VSLDictionaryExplorer
            onSelectSignFor3D={(signItem) => {
              setActiveTab('text_to_sign');
            }}
          />
        )}
      </main>

      {/* Footer */}
      <footer style={{
        padding: '16px 24px',
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(9, 13, 22, 0.95)',
        textAlign: 'center',
        fontSize: '12px',
        color: 'var(--text-muted)'
      }}>
        <p style={{ margin: 0 }}>
          <strong>VSL Connect AI</strong> — Hệ thống Giao tiếp Hai chiều Ngôn ngữ Ký hiệu Việt Nam • MediaPipe WebAssembly + PyTorch + FastAPI • Hỗ trợ hòa nhập cộng đồng người khiếm thính Việt Nam.
        </p>
      </footer>
    </div>
  );
}
