with open('src/components/signConventions.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re
matches = re.findall(r"'([A-ZĂÂĐÊÔƠƯ])':\s*\{", text)
print('Letters in SIGN_CONVENTIONS:', matches)

