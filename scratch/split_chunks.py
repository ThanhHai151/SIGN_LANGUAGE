with open('scratch/full_reconciliation_report.md', 'r', encoding='utf-8') as f:
    text = f.read()

split_marker = "## 9. PHỤ LỤC: BẢNG CHI TIẾT TOÀN BỘ 224 TRIALS (TRIAL ADJUDICATION TABLE)\n\n"
idx = text.find(split_marker)
part1 = text[:idx + len(split_marker)]
part2 = text[idx + len(split_marker):]

with open('scratch/report_part1.txt', 'w', encoding='utf-8') as f:
    f.write(part1)

with open('scratch/report_part2.txt', 'w', encoding='utf-8') as f:
    f.write(part2)

print('Part 1 bytes:', len(part1.encode('utf-8')))
print('Part 2 bytes:', len(part2.encode('utf-8')))

