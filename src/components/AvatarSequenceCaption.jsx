import React, { useEffect, useRef } from 'react';

/** Caption chuỗi ký hiệu chỉ dành cho khung Avatar Mixamo. */
export default function AvatarSequenceCaption({ sequence = [], currentIndex = 0 }) {
  const activeRef = useRef(null);
  const activeStep = sequence[currentIndex];

  useEffect(() => {
    const active = activeRef.current;
    const container = active?.parentElement;
    if (!active || !container) return;
    const activeRight = active.offsetLeft + active.offsetWidth;
    if (active.offsetLeft < container.scrollLeft) {
      container.scrollTo({ left: active.offsetLeft, behavior: 'smooth' });
    } else if (activeRight > container.scrollLeft + container.clientWidth) {
      container.scrollTo({ left: activeRight - container.clientWidth, behavior: 'smooth' });
    }
  }, [currentIndex, sequence]);

  if (!sequence.length) return null;

  const activeText = activeStep?.avatar_symbol || (activeStep?.type === 'fingerspelling'
    ? (activeStep.token || activeStep.label)
    : (activeStep?.label || activeStep?.token || ''));

  return (
    <section
      style={{
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(8, 15, 28, 0.92)',
        borderRadius: '0 0 var(--radius-md) var(--radius-md)',
        padding: '10px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        minWidth: 0
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
        <strong aria-live="polite" style={{ color: 'var(--text-main)', fontSize: '14px' }}>
          {activeText}
        </strong>
        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
          {currentIndex + 1}/{sequence.length}
        </span>
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '6px',
          overflowX: 'auto',
          padding: '3px 6px 6px',
          scrollbarWidth: 'thin',
          minWidth: 0
        }}
      >
        {sequence.map((step, index) => {
          const active = index === currentIndex;
          const done = index < currentIndex;
          const fingerspelling = step?.type === 'fingerspelling';
          const text = step.avatar_symbol || (fingerspelling ? (step.token || step.label) : (step.label || step.token));
          return (
            <span
              key={`${step?.entry_id || step?.token || step?.gloss || 'step'}-${index}`}
              ref={active ? activeRef : null}
              aria-current={active ? 'step' : undefined}
              title={step?.label || step?.token || ''}
              style={{
                flex: '0 0 auto',
                minWidth: '34px',
                padding: '5px 10px',
                borderRadius: '8px',
                border: `1px solid ${active ? 'transparent' : fingerspelling ? 'rgba(251,191,36,0.38)' : 'rgba(148,163,184,0.2)'}`,
                background: active ? (fingerspelling ? '#fbbf24' : '#22d3ee') : 'rgba(255,255,255,0.05)',
                color: active ? '#07111f' : done ? '#64748b' : '#e2e8f0',
                boxShadow: active ? `0 0 12px ${fingerspelling ? 'rgba(251,191,36,0.45)' : 'rgba(34,211,238,0.4)'}` : 'none',
                fontSize: '13px',
                lineHeight: 1.2,
                textAlign: 'center',
                whiteSpace: 'nowrap',
                fontWeight: active ? 800 : 600,
                transition: 'background-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease'
              }}
            >
              {text}
            </span>
          );
        })}
      </div>
    </section>
  );
}
