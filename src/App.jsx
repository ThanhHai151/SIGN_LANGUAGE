import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import CameraSignRecognizer from './components/CameraSignRecognizer';
import TextToSignVisualizer from './components/TextToSignVisualizer';
import VSLDictionaryExplorer from './components/VSLDictionaryExplorer';
import { checkBackendHealth } from './services/api';
import './Workspace.css';

const LIVE_WORDS_KEY = 'vsl-live-qipedc';

function savedLiveWords() {
  try {
    const words = JSON.parse(localStorage.getItem(LIVE_WORDS_KEY) || '[]');
    return Array.isArray(words) ? words.filter((word) => typeof word?.id === 'string' && typeof word?.word === 'string') : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('camera');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [requestText, setRequestText] = useState(null);
  const [practiceEntry, setPracticeEntry] = useState(null);
  const [liveWords, setLiveWords] = useState(savedLiveWords);

  useEffect(() => {
    try {
      localStorage.setItem(LIVE_WORDS_KEY, JSON.stringify(liveWords));
    } catch {
      // The current session still has the selected words if storage is disabled.
    }
  }, [liveWords]);

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
    <div className="app-shell">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        ttsEnabled={ttsEnabled}
        setTtsEnabled={setTtsEnabled}
      />

      <main className={activeTab === 'dictionary' ? 'app-main dictionary-page' : 'app-main'}>
        {activeTab === 'camera' && (
          <CameraSignRecognizer
            key={practiceEntry?.id || 'camera'}
            ttsEnabled={ttsEnabled}
            practiceEntry={practiceEntry}
            liveQipedcWords={liveWords}
            onBrowsePractice={() => setActiveTab('dictionary')}
            onExitPractice={() => setPracticeEntry(null)}
          />
        )}

        {activeTab === 'text_to_sign' && (
          <TextToSignVisualizer requestText={requestText} />
        )}

        {activeTab === 'dictionary' && (
          <VSLDictionaryExplorer
            liveQipedcWords={liveWords}
            onAddLiveWord={(item) => setLiveWords((words) => words.some((word) => word.id === item.id)
              ? words : [...words, { id: item.id, word: item.word }])}
            onRemoveLiveWord={(id) => setLiveWords((words) => words.filter((word) => word.id !== id))}
            onPractice={(entry) => {
              setPracticeEntry(entry);
              setActiveTab('camera');
            }}
            onSelectSignFor3D={(signItem) => {
              // Mở tab Văn bản → Ký hiệu và diễn ngay từ vừa chọn
              setRequestText({ text: signItem.keys?.[0] || signItem.word || signItem.label, nonce: Date.now() });
              setActiveTab('text_to_sign');
            }}
          />
        )}
      </main>

    </div>
  );
}
