with open('src/components/CameraSignRecognizer.jsx', 'r', encoding='utf-8') as f:
    c = f.read()

target_jsx = """              // Chốt chặn bảo vệ UI:
              // Nếu đang ở mode 'digits' mà kết quả không phải chữ số (0-9 hoặc NEUTRAL) -> bỏ qua
              if (requestMode === 'digits' && data.sign !== 'NEUTRAL' && !/^[0-9]$/.test(data.sign)) {
                return;
              }
              // Nếu đang ở mode 'letters' mà kết quả là chữ số -> bỏ qua
              if (requestMode === 'letters' && /^[0-9]$/.test(data.sign)) {
                return;
              }

              setCurrentResult({
                sign: data.sign,
                label: data.label,
                confidence: data.confidence,
                type: data.type || 'alphabet',
                description: data.description,
                category: requestMode === 'digits' ? 'Chữ số VSL (0–9)' : (requestMode === 'letters' ? 'Bảng chữ cái VSL' : (requestMode === 'words' ? 'Từ vựng VSL (102 Từ)' : 'Ghép câu ngữ cảnh')),
                video_url: ''
              });"""

replacement_jsx = """              // Chốt chặn bảo vệ UI:
              // Nếu đang ở mode 'digits' mà kết quả không phải chữ số (0-9, số nhiều chữ số, hoặc NEUTRAL) -> bỏ qua
              if (requestMode === 'digits' && data.sign !== 'NEUTRAL' && !/^[0-9]+$/.test(data.sign)) {
                return;
              }
              // Nếu đang ở mode 'letters' mà kết quả là chữ số -> bỏ qua
              if (requestMode === 'letters' && /^[0-9]+$/.test(data.sign)) {
                return;
              }

              setCurrentResult({
                sign: data.sign,
                label: data.label,
                confidence: data.confidence,
                type: data.type || 'alphabet',
                description: data.description,
                category: requestMode === 'digits' ? 'Chữ số VSL' : (requestMode === 'letters' ? 'Bảng chữ cái VSL' : (requestMode === 'words' ? 'Từ vựng VSL (102 Từ)' : 'Ghép câu ngữ cảnh')),
                video_url: ''
              });"""

assert target_jsx in c, "target_jsx not found"
c = c.replace(target_jsx, replacement_jsx, 1)

target_header = "{conversionMode === 'letters' ? '🔤 Chữ cái VSL (A–Z)' : conversionMode === 'digits' ? '🔢 Chữ số VSL (0–9)' :"
replacement_header = "{conversionMode === 'letters' ? '🔤 Chữ cái VSL (A–Z)' : conversionMode === 'digits' ? '🔢 Chữ số VSL' :"

if target_header in c:
    c = c.replace(target_header, replacement_header, 1)

with open('src/components/CameraSignRecognizer.jsx', 'w', encoding='utf-8') as f:
    f.write(c)

print("src/components/CameraSignRecognizer.jsx updated successfully")

