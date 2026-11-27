import { useState } from 'react';
import { ArrowLeft, ArrowRight, Check, GripVertical, Pencil, Trash2, X } from 'lucide-react';

export default function CaptionTokens({ tokens, selectedId, onSelect, onRemove, onMove, onEdit, captionRef }) {
  const [draggedId, setDraggedId] = useState(null);
  const [dropId, setDropId] = useState(null);
  const [editor, setEditor] = useState(null);
  const selectedIndex = tokens.findIndex(({ id }) => id === selectedId);
  const editing = selectedIndex >= 0 && editor?.id === selectedId;

  const endDrag = () => {
    setDraggedId(null);
    setDropId(null);
  };

  return (
    <>
      <div ref={captionRef} className="caption-text" data-empty={!tokens.length} role="log"
        aria-label="Văn bản đã nhận diện">
        {tokens.length ? (
          <ol className="caption-tokens">
            {tokens.map(({ id, text }) => (
              <li key={id}>
                <button type="button" className="caption-token" aria-pressed={selectedId === id}
                  data-drop-target={dropId === id} draggable
                  onClick={() => {
                    setEditor(null);
                    onSelect(selectedId === id ? null : id);
                  }}
                  onDoubleClick={() => onRemove(id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Delete') {
                      event.preventDefault();
                      onRemove(id);
                    }
                  }}
                  onDragStart={(event) => {
                    event.dataTransfer.effectAllowed = 'move';
                    event.dataTransfer.setData('text/plain', text);
                    setDraggedId(id);
                  }}
                  onDragEnd={endDrag}
                  onDragOver={(event) => {
                    if (draggedId !== null && draggedId !== id) {
                      event.preventDefault();
                      event.dataTransfer.dropEffect = 'move';
                      setDropId(id);
                    }
                  }}
                  onDragLeave={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setDropId((previous) => previous === id ? null : previous);
                    }
                  }}
                  onDrop={(event) => {
                    if (draggedId !== null) {
                      event.preventDefault();
                      onMove(draggedId, id);
                      endDrag();
                    }
                  }}
                  title="Kéo để đổi vị trí · Nhấp đúp để xóa">
                  <GripVertical size={13} aria-hidden="true" />
                  <span>{text}</span>
                </button>
              </li>
            ))}
          </ol>
        ) : 'Phụ đề sẽ xuất hiện tại đây.'}
      </div>
      {editing ? (
        <form className="caption-token-editor" onSubmit={(event) => {
          event.preventDefault();
          const text = editor.text.trim();
          if (!text) return;
          onEdit(editor.id, text);
          setEditor(null);
        }}>
          <input autoFocus aria-label="Nội dung cụm từ" value={editor.text}
            onChange={(event) => setEditor({ id: editor.id, text: event.target.value })}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault();
                setEditor(null);
              }
            }} />
          <button type="submit" className="icon-button" aria-label="Lưu cụm từ" disabled={!editor.text.trim()}>
            <Check size={15} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" aria-label="Hủy sửa cụm từ" onClick={() => setEditor(null)}>
            <X size={15} aria-hidden="true" />
          </button>
        </form>
      ) : selectedIndex >= 0 && (
        <div className="caption-token-actions" aria-label="Chỉnh sửa cụm từ đã chọn">
          <button type="button" className="icon-button" aria-label="Di chuyển cụm từ sang trái"
            disabled={selectedIndex === 0} onClick={() => onMove(selectedId, tokens[selectedIndex - 1].id)}>
            <ArrowLeft size={15} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" aria-label="Di chuyển cụm từ sang phải"
            disabled={selectedIndex === tokens.length - 1} onClick={() => onMove(selectedId, tokens[selectedIndex + 1].id)}>
            <ArrowRight size={15} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" aria-label="Sửa nội dung cụm từ"
            onClick={() => setEditor(tokens[selectedIndex])}>
            <Pencil size={15} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button caption-clear" aria-label="Xóa cụm từ đã chọn"
            onClick={() => onRemove(selectedId)}>
            <Trash2 size={15} aria-hidden="true" />
          </button>
        </div>
      )}
    </>
  );
}
