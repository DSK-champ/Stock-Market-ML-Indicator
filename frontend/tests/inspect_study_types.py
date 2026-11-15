import pathlib, re

path = pathlib.Path('frontend/charting_library/bundles/library.6738f23786f3f834acd7.js')
text = path.read_text(encoding='utf-8')

# Search for plots definitions: plots:[{...}]
plot_types = set(re.findall(r'type:"([a-zA-Z0-9_]+)"', text))
print("Plot types in TV bundle:", plot_types)

# Search for any studies that plot arrows or shapes or signals
studies_with_shapes = re.findall(r'name:"([^"]+)",metainfo:\{[^}]+type:"(shapes|chars|arrows|plot_shape|colorer)"', text)
print("Studies with shapes/arrows/chars:", studies_with_shapes[:10])

# Search for Parabolic SAR or Supertrend or similar indicator
sar_idx = text.find('name:"Parabolic SAR"')
if sar_idx != -1:
    print("Parabolic SAR metainfo:\n", text[sar_idx:sar_idx+600])

arrow_idx = text.find('type:"arrows"')
if arrow_idx != -1:
    print("Arrows sample:\n", text[arrow_idx-100:arrow_idx+200])

shape_idx = text.find('type:"shapes"')
if shape_idx != -1:
    print("Shapes sample:\n", text[shape_idx-100:shape_idx+200])

st_idx = text.find('name:"SuperTrend"')
if st_idx != -1:
    print("SuperTrend full definition:\n", text[st_idx:st_idx+2000])

