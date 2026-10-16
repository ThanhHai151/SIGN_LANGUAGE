import React, { useEffect, useMemo, useState } from 'react';
import { Search, BookOpen, Film, Play, X, Camera, Bot, Star, Sparkles, WifiOff } from 'lucide-react';
import { fetchLexicon, fetchLexiconEntry, prefetchVideos, warmVideo } from '../services/api';
import SmoothVideo from './SmoothVideo';
import SignDescription from './SignDescription';

/**
 * Từ điển VSL — duyệt BỘ TỪ VỰNG THỐNG NHẤT:
 *  ~3.300 từ Danh mục NNKH QIPEDC (Bộ GD&ĐT, video người Điếc 3 miền Bắc/Trung/Nam)
 *  + 102 từ giáo trình (mô hình camera nhận diện được) + từ có tư thế robot 3D.
 */
const PAGE = 60;

const FILTERS = [
  { id: 'common', label: 'Thông dụng', icon: Star },
  { id: 'all', label: 'Tất cả', icon: BookOpen },
  { id: 'camera', label: 'Camera nhận diện được', icon: Camera },
  { id: 'robot', label: 'Robot 3D diễn được', icon: Bot },
  { id: 'alphabet', label: 'Bảng chữ cái', icon: Sparkles },
];

const strip = (s) => (s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').replace(/Đ/g, 'D').toLowerCase();

export default function VSLDictionaryExplorer({ onSelectSignFor3D }) {
  const [entries, setEntries] = useState([]);
  const [meta, setMeta] = useState(null);
  const [status, setStatus] = useState('loading');
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('common');
  const [pos, setPos] = useState('');
  const [shown, setShown] = useState(PAGE);
  const [modal, setModal] = useState(null);
  const [modalDetail, setModalDetail] = useState(null);
  // Mở chi tiết từ → tải phần diễn tả ký hiệu (phân tích từ video)
  useEffect(() => {
    if (!modal?.id) return undefined;
    let alive = true;
    fetchLexiconEntry(modal.id).then((d) => { if (alive && d) setModalDetail(d); });
    return () => { alive = false; };
  }, [modal]);

  useEffect(() => {
    (async () => {
      const data = await fetchLexicon();
      if (data && data.entries) {
        setEntries(data.entries);
        setMeta(data.meta);
        setStatus('ok');
      } else {
        setStatus('offline');
      }
    })();
  }, []);

  useEffect(() => { setShown(PAGE); }, [searchTerm, filter, pos]);


  const counts = useMemo(() => ({
    common: entries.filter((e) => e.common && !e.fingerspelling).length,
    all: entries.length,
    camera: entries.filter((e) => e.sstcn_index !== null && e.sstcn_index !== undefined).length,
    robot: entries.filter((e) => e.pose_key || e.motion_id).length,
    alphabet: entries.filter((e) => e.fingerspelling).length,
  }), [entries]);

  const posOptions = useMemo(() => {
    const c = {};
    entries.forEach((e) => { const p = (e.pos || '').trim(); if (p) c[p] = (c[p] || 0) + 1; });
    return Object.entries(c).filter(([, n]) => n >= 20).sort((a, b) => b[1] - a[1]).map(([p]) => p);
  }, [entries]);

  // Chỉ dùng 1 video chính của QIPEDC (video giáo trình trên Supabase hay bị chặn/treo)
  const videoOf = (e) => ((e?.video_url || '').includes('qipedc.moet.gov.vn') ? e.video_url : null);

  const filtered = useMemo(() => {
    let list = entries;
    if (filter === 'common') list = list.filter((e) => e.common && !e.fingerspelling);
    else if (filter === 'camera') list = list.filter((e) => e.sstcn_index !== null && e.sstcn_index !== undefined).sort((a, b) => a.sstcn_index - b.sstcn_index);
    else if (filter === 'robot') list = list.filter((e) => e.pose_key || e.motion_id);
    else if (filter === 'alphabet') list = list.filter((e) => e.fingerspelling);
    if (pos) list = list.filter((e) => (e.pos || '').trim() === pos);
    const q = searchTerm.trim().toLowerCase();
    if (q) {
      const qa = strip(q);
      const scored = [];
      for (const e of list) {
        const w = e.word.toLowerCase();
        let score = -1;
        if (w === q) score = 0;
        else if (w.startsWith(q)) score = 1;
        else if (w.includes(q)) score = 2;
        else if (strip(w).includes(qa)) score = 3;
        else if ((e.definition || '').toLowerCase().includes(q)) score = 4;
        if (score >= 0) scored.push([score, e]);
      }
      list = scored.sort((a, b) => a[0] - b[0] || a[1].rank - b[1].rank).map((x) => x[1]);
    }
    return list;
  }, [entries, filter, pos, searchTerm]);

  // Nhờ backend tải trước video của các ô đầu danh sách (chờ người dùng gõ xong 0,6 giây)
  useEffect(() => {
    const t = setTimeout(() => prefetchVideos(filtered.slice(0, 24).map(videoOf).filter(Boolean)), 600);
    return () => clearTimeout(t);
  }, [filtered]); // eslint-disable-line react-hooks/exhaustive-deps

  const openModal = (e) => setModal(e);

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px', width: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header & bộ lọc */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={22} style={{ color: 'var(--primary)' }} />
              Từ điển Ngôn ngữ Ký hiệu Việt Nam
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: 0 }}>
              {meta
                ? `${meta.total.toLocaleString('vi-VN')} mục từ · ${meta.total_videos.toLocaleString('vi-VN')} video người thật — Danh mục NNKH QIPEDC (Bộ GD&ĐT) + 102 từ giáo trình camera + từ robot 3D. Dùng chung cho cả 2 chiều dịch.`
                : 'Danh mục NNKH QIPEDC (Bộ GD&ĐT) + giáo trình 102 từ + robot 3D'}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <select
              value={pos}
              onChange={(e) => setPos(e.target.value)}
              style={{
                padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-input)',
                border: '1px solid var(--border-subtle)', color: 'var(--text-main)', fontSize: '13px'
              }}
              title="Lọc theo từ loại"
            >
              <option value="">Mọi từ loại</option>
              {posOptions.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            <div style={{ position: 'relative', width: '320px', maxWidth: '100%' }}>
              <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Tìm ký hiệu (vd: gia đình, cảm ơn, bác sĩ…)"
                style={{
                  width: '100%', padding: '10px 14px 10px 36px', borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-input)', border: '1px solid var(--border-subtle)',
                  color: 'var(--text-main)', fontSize: '13px', outline: 'none'
                }}
              />
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {FILTERS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setFilter(id)}
              className={`btn ${filter === id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 14px', fontSize: '12px' }}
            >
              <Icon size={13} /> {label} {status === 'ok' && <span style={{ opacity: 0.75 }}>({counts[id].toLocaleString('vi-VN')})</span>}
            </button>
          ))}
        </div>
      </div>

      {status === 'loading' && (
        <div className="glass-panel" style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>Đang tải bộ từ vựng…</div>
      )}
      {status === 'offline' && (
        <div className="glass-panel" style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <WifiOff size={26} />
          Không kết nối được Backend (http://127.0.0.1:8000). Hãy chạy <code>python -m uvicorn backend.app:app --reload</code> rồi tải lại trang.
        </div>
      )}

      {status === 'ok' && (
        <>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {filtered.length.toLocaleString('vi-VN')} kết quả{filter === 'common' && !searchTerm ? ' · xếp theo mức độ thông dụng trong giao tiếp' : ''}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '16px' }}>
            {filtered.slice(0, shown).map((item) => (
              <div key={item.id} className="glass-panel" style={{ padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {/* Ảnh thu nhỏ / nút phát */}
                <button
                  onClick={() => videoOf(item) && openModal(item)}
                  onMouseEnter={() => videoOf(item) && warmVideo(videoOf(item))}
                  onFocus={() => videoOf(item) && warmVideo(videoOf(item))}
                  disabled={!videoOf(item)}
                  style={{
                    position: 'relative', border: 'none', padding: 0, background: '#05080f',
                    aspectRatio: '16 / 9', cursor: videoOf(item) ? 'pointer' : 'default', display: 'block', width: '100%'
                  }}
                  title={videoOf(item) ? 'Xem video mẫu' : 'Chưa có video mẫu'}
                >
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 26, fontWeight: 800, padding: 8, textAlign: 'center' }}>
                    {item.label}
                  </div>
                  {item.thumb_url && (
                    <img
                      src={item.thumb_url}
                      alt={item.word}
                      loading="lazy"
                      decoding="async"
                      onError={(e) => { e.currentTarget.style.display = 'none'; }}
                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  )}
                  {videoOf(item) && (
                    <span style={{
                      position: 'absolute', right: 8, bottom: 8, width: 30, height: 30, borderRadius: '50%',
                      background: 'rgba(8,15,28,0.85)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#22d3ee'
                    }}>
                      <Play size={14} />
                    </span>
                  )}
                </button>

                <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, flexWrap: 'wrap' }}>
                    <h3 style={{ fontSize: '17px', fontWeight: 700, margin: 0 }}>{item.label}</h3>
                    {item.pos && <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{item.pos}</span>}
                  </div>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                    {item.sstcn_index !== null && item.sstcn_index !== undefined && (
                      <span className="badge badge-emerald" style={{ fontSize: '10px' }}><Camera size={10} /> Camera</span>
                    )}
                    {(item.pose_key || item.motion_id) && <span className="badge badge-cyan" style={{ fontSize: '10px' }}><Bot size={10} /> Robot</span>}
                  </div>
                  {(item.definition || item.how_to) && (
                    <p style={{
                      fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.45, margin: 0,
                      display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden'
                    }}>
                      {item.definition || item.how_to}
                    </p>
                  )}
                  <div style={{ flex: 1 }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 8, borderTop: '1px solid var(--border-subtle)' }}>
                    {videoOf(item) ? (
                      <button onClick={() => openModal(item)} style={{ background: 'none', border: 'none', color: 'var(--primary)', fontSize: '11px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, padding: 0 }}>
                        <Film size={12} /> Xem video &amp; cách làm
                      </button>
                    ) : <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Chỉ có robot 3D</span>}
                    {onSelectSignFor3D && (
                      <button onClick={() => onSelectSignFor3D(item)} style={{ background: 'none', border: 'none', color: 'var(--accent-success)', fontSize: '11px', fontWeight: 600, cursor: 'pointer', padding: 0 }}>
                        Diễn trong câu →
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {shown < filtered.length && (
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <button className="btn btn-secondary" onClick={() => setShown((n) => n + PAGE * 2)} style={{ padding: '8px 20px' }}>
                Xem thêm ({(filtered.length - shown).toLocaleString('vi-VN')} từ nữa)
              </button>
            </div>
          )}
        </>
      )}

      {/* Popup video mẫu */}
      {modal && (
        <div
          onClick={() => setModal(null)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.85)', backdropFilter: 'blur(8px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999, padding: '20px'
          }}
        >
          <div className="glass-panel" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '720px', width: '100%', maxHeight: '92vh', overflowY: 'auto', padding: '20px', boxShadow: '0 20px 40px rgba(0,0,0,0.85)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12, gap: 12 }}>
              <div>
                <span className="badge badge-cyan" style={{ fontSize: '10px' }}>{modal.source}</span>
                <h3 style={{ fontSize: '22px', fontWeight: 700, margin: '6px 0 0' }}>
                  {modal.label} {modal.pos && <span style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 500 }}>· {modal.pos}</span>}
                </h3>
              </div>
              <button onClick={() => setModal(null)} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}><X size={22} /></button>
            </div>

            {videoOf(modal) && (
              <SmoothVideo url={videoOf(modal)} poster={modal.thumb_url} />
            )}

            {modal.definition && <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '12px 0 0', lineHeight: 1.5 }}><strong>Nghĩa: </strong>{modal.definition}</p>}
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Diễn tả ký hiệu</div>
              <SignDescription step={{ ...modal, ...(modalDetail && modalDetail.id === modal.id ? modalDetail : {}) }} />
            </div>
            {modal.image_url && <img src={modal.image_url} alt="" style={{ marginTop: 12, maxHeight: 160, borderRadius: 8 }} onError={(e) => { e.currentTarget.style.display = 'none'; }} />}
          </div>
        </div>
      )}
    </div>
  );
}
