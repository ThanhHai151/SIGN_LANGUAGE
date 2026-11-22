import { BookOpen, Camera, MessageSquare, Volume2, VolumeX } from 'lucide-react';

const pages = [
  { id: 'camera', label: 'Camera', title: 'Camera nhận diện ký hiệu', Icon: Camera },
  { id: 'text_to_sign', label: 'Văn bản', title: 'Văn bản sang ký hiệu 3D', Icon: MessageSquare },
  { id: 'dictionary', label: 'Từ điển', title: 'Từ điển và luyện tập VSL', Icon: BookOpen },
];

export default function Navbar({ activeTab, setActiveTab, backendStatus, ttsEnabled, setTtsEnabled }) {
  const status = backendStatus === 'online' ? 'Máy chủ sẵn sàng'
    : backendStatus === 'checking' ? 'Đang kết nối máy chủ' : 'Máy chủ ngoại tuyến';
  const VoiceIcon = ttsEnabled ? Volume2 : VolumeX;

  return (
    <aside className="app-rail">
      <div className="app-brand" aria-label="VSL Connect AI">
        <strong>VSL</strong>
        <span>Connect</span>
      </div>
      <nav className="app-nav" aria-label="Điều hướng">
        {pages.map(({ id, label, title, Icon }) => (
          <button key={id} type="button" id={`nav-tab-${id.replaceAll('_', '-')}`}
            className="rail-button" aria-current={activeTab === id ? 'page' : undefined}
            aria-label={title} title={title} onClick={() => setActiveTab(id)}>
            <Icon size={20} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="rail-footer">
        <button type="button" className="rail-button" aria-label="Phát giọng nói"
          aria-pressed={ttsEnabled} title={ttsEnabled ? 'Tắt giọng nói' : 'Bật giọng nói'}
          onClick={() => setTtsEnabled(!ttsEnabled)}>
          <VoiceIcon size={18} aria-hidden="true" />
          <span>Giọng nói</span>
        </button>
        <span className="rail-status" role="status" aria-label={status} title={status} data-status={backendStatus}>
          <i aria-hidden="true" />
        </span>
      </div>
    </aside>
  );
}
