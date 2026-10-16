# Nguồn dữ liệu bộ từ vựng VSL

| File | Nội dung | Nguồn |
|---|---|---|
| `qipedc_dictionary.json` | 4.362 video / ~3.300 từ: `[mã video, từ, từ loại, giải nghĩa, type, có ảnh minh họa]` | Danh mục Ngôn ngữ ký hiệu tiếng Việt – Dự án QIPEDC, Bộ GD&ĐT: https://qipedc.moet.gov.vn/dictionary (video: `/videos/{mã}.mp4`, ảnh nhỏ: `/thumbs/{mã}.png`). Mã có hậu tố `B`/`T`/`N` = biến thể miền Bắc/Trung/Nam. |
| `vi_syllable_freq.tsv` | Tần suất âm tiết tiếng Việt trong hội thoại (chỉ giữ các âm tiết có trong từ điển) | FrequencyWords – Hermit Dave, OpenSubtitles 2018 (CC BY-SA 4.0): https://github.com/hermitdave/FrequencyWords |

Dựng lại `backend/data/vsl_lexicon.json`:

```bash
python -m backend.training.build_lexicon
```
