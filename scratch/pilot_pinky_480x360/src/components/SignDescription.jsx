import React from 'react';
import { Hand, BookOpen, Info } from 'lucide-react';

/**
 * Diễn tả cụ thể cách làm một ký hiệu:
 *  - "Theo từ điển": mô tả gốc (nếu mục từ có)
 *  - "Phân tích từ video": số tay, hình dạng bàn tay, hướng lòng bàn tay, hướng đầu ngón tay,
 *    vị trí và chuyển động của từng tay — sinh tự động từ chuyển động người thật mà robot đã học
 *    (backend/training/describe_signs.py)
 */
const SIDE = { right: 'Tay phải', left: 'Tay trái' };

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

function Row({ label, children }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '112px 1fr', gap: 8, fontSize: 12.5, lineHeight: 1.45 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-main)' }}>{children}</span>
    </div>
  );
}

function HandPart({ p }) {
  const shape = p.shape_change ? `${cap(p.shape_change[0])} → ${p.shape_change[1]}` : cap(p.shape);
  const where = p.location_end ? `${cap(p.location)} → ${p.location_end}` : cap(p.location);
  const palm = p.palm_end ? `${cap(p.palm)} → ${p.palm_end}` : cap(p.palm);
  return (
    <div style={{
      padding: '10px 12px', borderRadius: 10, background: 'rgba(15, 23, 42, 0.55)',
      border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 5
    }}>
      <strong style={{ fontSize: 13, color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
        <Hand size={14} /> {SIDE[p.side] || p.side}
      </strong>
      <Row label="Hình dạng tay">{shape}</Row>
      <Row label="Lòng bàn tay">{palm}</Row>
      <Row label="Đầu ngón tay">{cap(p.point)}</Row>
      <Row label="Vị trí">{where}</Row>
      <Row label="Chuyển động">{cap(p.movement)}</Row>
    </div>
  );
}

export default function SignDescription({ step, compact = false }) {
  if (!step) return null;
  const d = step.sign_description;
  const isSpell = step.type === 'fingerspelling';
  const fallback = !d && !step.how_to ? (step.description && step.description !== step.definition ? step.description : '') : '';

  if (!d && !step.how_to && !fallback) {
    return (
      <p style={{ margin: 0, fontSize: 12.5, color: 'var(--text-muted)' }}>
        Chưa có diễn tả chi tiết cho ký hiệu này.
      </p>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {step.how_to && (
        <div style={{ fontSize: 12.5, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-main)', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <BookOpen size={13} style={{ color: 'var(--primary)' }} /> Theo từ điển:
          </strong>{' '}
          {step.how_to}
        </div>
      )}

      {fallback && (
        <div style={{ fontSize: 12.5, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-main)' }}>{isSpell ? 'Cách đánh vần: ' : 'Cách làm: '}</strong>
          {fallback}
        </div>
      )}

      {d && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-main)' }}>Phân tích từ video:</span>
            <span className="badge badge-cyan" style={{ fontSize: 10.5 }}>{d.hands}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: !compact && d.parts.length > 1 ? 'repeat(auto-fit, minmax(240px, 1fr))' : '1fr', gap: 8 }}>
            {d.parts.map((p) => <HandPart key={p.side} p={p} />)}
          </div>
          {d.relation && (
            <div style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--text-main)' }}>Phối hợp hai tay: </strong>{cap(d.relation)}.
            </div>
          )}
          <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 5, alignItems: 'flex-start' }}>
            <Info size={12} style={{ flexShrink: 0, marginTop: 1 }} />
            Phân tích tự động từ video người thật trong Từ điển NNKH (QIPEDC) — trái/phải tính theo người làm ký hiệu; hình dạng ngón tay có thể chưa chính xác tuyệt đối.
          </div>
        </>
      )}
    </div>
  );
}
