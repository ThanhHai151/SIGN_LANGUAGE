import React, { useState, useEffect } from 'react';
import { Search, BookOpen, Hand, Sparkles, Filter, CheckCircle, Film, Play, X, ExternalLink } from 'lucide-react';
import { fetchDictionary, fetchVSL102Classes } from '../services/api';

export default function VSLDictionaryExplorer({ onSelectSignFor3D }) {
  const [dictionaryData, setDictionaryData] = useState({ alphabet: {}, vocabulary: {} });
  const [vsl102List, setVsl102List] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [activeVideoModal, setActiveVideoModal] = useState(null);

  useEffect(() => {
    async function loadData() {
      const [dictData, vsl102] = await Promise.all([
        fetchDictionary(),
        fetchVSL102Classes()
      ]);

      if (dictData) {
        setDictionaryData(dictData);
      }
      if (vsl102 && vsl102.signs) {
        setVsl102List(vsl102.signs);
      }
    }
    loadData();
  }, []);

  // Tổng hợp tất cả các mục từ kết hợp cả 102 từ VSL chuẩn hóa
  const allItems = [
    // 102 Ký hiệu VSL chuẩn hóa (có video giáo trình)
    ...vsl102List.map(item => ({
      key: `vsl102_${item.index}`,
      label: item.label,
      description: item.description,
      category: item.category || 'Giáo trình VSL',
      video_url: item.video_url,
      image_url: item.image_url,
      is_sstcn: true,
      index: item.index
    })),
    // Bảng chữ cái hiện có nếu chưa có trong 102
    ...Object.entries(dictionaryData.alphabet || {})
      .filter(([key, val]) => !vsl102List.some(v => v.label.toUpperCase() === key.toUpperCase()))
      .map(([key, val]) => ({
        key: `alpha_${key}`,
        ...val,
        category: 'Bảng chữ cái ngón tay',
        is_sstcn: false
      })),
    // Từ vựng hiện có
    ...Object.entries(dictionaryData.vocabulary || {})
      .filter(([key, val]) => !vsl102List.some(v => v.label.toLowerCase() === val.label?.toLowerCase()))
      .map(([key, val]) => ({
        key: `voc_${key}`,
        ...val,
        category: 'Từ vựng giao tiếp',
        is_sstcn: false
      }))
  ];

  // Lọc theo tìm kiếm và danh mục
  const filteredItems = allItems.filter(item => {
    const term = searchTerm.toLowerCase();
    const matchesSearch = (item.label && item.label.toLowerCase().includes(term)) ||
                          (item.description && item.description.toLowerCase().includes(term)) ||
                          (item.category && item.category.toLowerCase().includes(term));
    
    if (activeCategory === 'all') return matchesSearch;
    if (activeCategory === 'sstcn') return matchesSearch && item.is_sstcn;
    if (activeCategory === 'alphabet') return matchesSearch && item.category.includes('chữ cái');
    if (activeCategory === 'vocabulary') return matchesSearch && !item.category.includes('chữ cái');
    return matchesSearch;
  });

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px', width: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Header & Bộ lọc */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={22} style={{ color: 'var(--primary)' }} />
              Kho Ngữ Liệu & Từ Điển Ngôn Ngữ Ký Hiệu Việt Nam (VSL)
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              102 Ký hiệu chuẩn hóa theo giáo trình cô Trần Thị Thiệp (có video thực hành) kết hợp Từ điển 3D
            </p>
          </div>

          {/* Ô tìm kiếm */}
          <div style={{ position: 'relative', width: '340px', maxWidth: '100%' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Tìm kiếm ký hiệu (vd: Chào, Cảm ơn, Ăn, A...)"
              style={{
                width: '100%',
                padding: '10px 14px 10px 36px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-input)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-main)',
                fontSize: '13px',
                outline: 'none'
              }}
            />
          </div>
        </div>

        {/* Tabs Danh mục */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: `Tất cả (${allItems.length})` },
            { id: 'sstcn', label: `102 Từ VSL Nhận diện Camera (${vsl102List.length})` },
            { id: 'alphabet', label: 'Bảng chữ cái ngón tay' },
            { id: 'vocabulary', label: 'Từ vựng sinh hoạt & giao tiếp' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveCategory(tab.id)}
              className={`btn ${activeCategory === tab.id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 14px', fontSize: '12px' }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Grid danh sách Ký hiệu VSL */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: '16px'
      }}>
        {filteredItems.map(item => (
          <div
            key={item.key}
            className="glass-panel"
            style={{
              padding: '18px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '12px',
              transition: 'transform 0.2s ease, border-color 0.2s ease',
              position: 'relative'
            }}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '8px' }}>
                <div>
                  <span className={`badge ${item.is_sstcn ? 'badge-cyan' : 'badge-indigo'}`} style={{ fontSize: '10px', marginBottom: '6px' }}>
                    {item.category}
                  </span>
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-main)', marginTop: '4px' }}>
                    {item.label}
                  </h3>
                </div>

                {item.video_url ? (
                  <button
                    onClick={() => setActiveVideoModal(item)}
                    className="btn btn-secondary"
                    style={{ padding: '6px 10px', fontSize: '11px', gap: '4px' }}
                    title="Xem video giáo trình thực tế"
                  >
                    <Film size={14} style={{ color: 'var(--primary)' }} /> Video
                  </button>
                ) : (
                  <div style={{
                    width: '34px',
                    height: '34px',
                    borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--primary)'
                  }}>
                    <Hand size={16} />
                  </div>
                )}
              </div>

              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.45', margin: 0 }}>
                {item.description}
              </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)' }}>
              {item.video_url ? (
                <button
                  onClick={() => setActiveVideoModal(item)}
                  style={{ background: 'none', border: 'none', color: 'var(--primary)', fontSize: '11px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', padding: 0 }}
                >
                  <Play size={12} /> Phát video mẫu
                </button>
              ) : (
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Cử chỉ mô phỏng</span>
              )}

              {onSelectSignFor3D && (
                <button
                  onClick={() => onSelectSignFor3D(item)}
                  style={{ background: 'none', border: 'none', color: 'var(--accent-success)', fontSize: '11px', fontWeight: '600', cursor: 'pointer', padding: 0 }}
                >
                  Diễn hoạt 3D →
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* MODAL POPUP VIDEO MẪU TỪ SUPABASE */}
      {activeVideoModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div className="glass-panel" style={{
            maxWidth: '580px',
            width: '100%',
            padding: '22px',
            position: 'relative',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.85)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <span className="badge badge-cyan" style={{ fontSize: '10px' }}>
                  {activeVideoModal.category}
                </span>
                <h3 style={{ fontSize: '18px', fontWeight: '700', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Film size={18} style={{ color: 'var(--primary)' }} />
                  Ký hiệu VSL: {activeVideoModal.label}
                </h3>
              </div>
              <button
                onClick={() => setActiveVideoModal(null)}
                style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}
              >
                <X size={22} />
              </button>
            </div>

            <video
              src={activeVideoModal.video_url}
              controls
              autoPlay
              loop
              playsInline
              style={{
                width: '100%',
                borderRadius: 'var(--radius-md)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)'
              }}
            />

            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '14px', lineHeight: '1.45' }}>
              💡 {activeVideoModal.description}
            </p>
          </div>
        </div>
      )}

    </div>
  );
}
