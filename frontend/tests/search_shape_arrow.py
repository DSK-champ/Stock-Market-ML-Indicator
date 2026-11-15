import pathlib, re

path = pathlib.Path('frontend/charting_library/bundles/library.6738f23786f3f834acd7.js')
text = path.read_text(encoding='utf-8')

for m in re.finditer(r'shape_arrow_up', text):
    start = max(0, m.start() - 100)
    end = min(len(text), m.end() + 200)
    print("Match:\n", text[start:end], "\n---")
