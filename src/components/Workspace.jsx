import { useId, useRef, useState } from 'react';
import { SlidersHorizontal, X } from 'lucide-react';

export function Workspace({ controls, children }) {
  const [controlsOpen, setControlsOpen] = useState(false);
  const controlsId = useId();
  const openerRef = useRef(null);
  const closeControlRef = useRef(null);

  function toggleControls(open) {
    setControlsOpen(open);
    requestAnimationFrame(() => (open ? closeControlRef : openerRef).current?.focus());
  }

  return (
    <div className="workspace" onKeyDown={(event) => {
      if (event.key === 'Escape' && !event.defaultPrevented && controlsOpen) toggleControls(false);
    }}>
      <aside id={controlsId} aria-label="Điều khiển" className="workspace-sidebar" data-open={controlsOpen}>
        <div className="workspace-controls-heading">
          <SlidersHorizontal size={16} aria-hidden="true" />
          <h2>Điều khiển</h2>
          <button ref={closeControlRef} type="button" className="workspace-sheet-close icon-button"
            aria-label="Đóng điều khiển" onClick={() => toggleControls(false)}><X size={18} /></button>
        </div>
        {controls}
      </aside>
      {controlsOpen && <button type="button" tabIndex={-1} aria-label="Đóng ngăn điều khiển"
        className="workspace-scrim" onClick={() => toggleControls(false)} />}
      <div className="workspace-main">
        <div className="workspace-mobile-tools">
          <button type="button" className="btn btn-secondary" ref={openerRef}
            aria-controls={controlsId} aria-expanded={controlsOpen} onClick={() => toggleControls(!controlsOpen)}>
            <SlidersHorizontal size={15} /> Điều khiển
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function WorkspaceDialog({ dialogRef, title, onClose, children }) {
  const titleId = useId();

  return (
    <dialog ref={dialogRef} className="workspace-dialog" aria-labelledby={titleId} onClose={onClose}>
      <header className="workspace-dialog-heading">
        <h2 id={titleId}>{title}</h2>
        <button type="button" className="icon-button" aria-label={`Đóng ${title.toLowerCase()}`}
          onClick={() => dialogRef.current?.close()}><X size={20} /></button>
      </header>
      <div className="workspace-dialog-body">{children}</div>
    </dialog>
  );
}
