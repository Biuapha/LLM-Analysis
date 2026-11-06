"""Embed the fixed six-curve result and page assets into the site with synchronous data assets."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'glm53_prefill_decode_operator_block_diagram.html'
original = path.read_bytes().decode('utf-8')
newline = '\r\n' if '\r\n' in original else '\n'
html = original.replace('\r\n', '\n')

def asset(tag, name, body, attrs=''):
    global html
    block = f'<{tag} id="{name}"{attrs}>\n{body.strip()}\n</{tag}>'
    pattern = rf'<{tag} id="{name}"[^>]*>[\s\S]*?</{tag}>'
    if re.search(pattern, html):
        html = re.sub(pattern, lambda _: block, html, count=1)
    else:
        marker = '</head>' if tag == 'style' else '</body>'
        assert marker in html
        html = html.replace(marker, block+'\n'+marker, 1)

def external_json(name, filename, data):
    global html
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    js = "(() => {\n  const node = document.createElement('script');\n"
    js += f"  node.id = {json.dumps(name)}; node.type = 'application/json';\n"
    js += "  node.textContent = JSON.stringify(" + payload + ");\n"
    js += "  document.currentScript.before(node);\n})();\n"
    (root / filename).write_text(js, encoding='utf-8')
    block = f'<script id="{name}-loader" src="{filename}"></script>'
    pattern = rf'<script id="{name}(?:-loader)?"[^>]*>[\s\S]*?</script>'
    assert re.search(pattern, html), name
    html = re.sub(pattern, lambda _: block, html, count=1)

external_json('cache-data', 'data/glm53-cache-reference.js', json.loads((root / 'data/glm53-cache-reference.json').read_text(encoding='utf-8')))

body = (root / 'src/decode-comparison.html').read_text(encoding='utf-8').strip()
block = '<!-- decode-comparison-page:start -->\n'+body+'\n<!-- decode-comparison-page:end -->'
pattern = r'<!-- decode-comparison-page:start -->[\s\S]*?<!-- decode-comparison-page:end -->'
if re.search(pattern, html):
    html = re.sub(pattern, lambda _: block, html, count=1)
else:
    marker = '<section id="ds41-root"'
    assert html.count(marker) == 1
    html = html.replace(marker, block+'\n'+marker, 1)

if 'id="comparison-tab"' not in html:
    marker = '4 · 纯 P / 纯 D 仿真</button>'
    assert html.count(marker) == 1
    html = html.replace(marker, marker+'<button id="comparison-tab" type="button" role="tab" aria-controls="comparison-page" aria-selected="false" data-page="comparison" title="GLM-5.3-Flash 六曲线成果">5 · 850 / 850E 对比</button>', 1)

edits = [
    ("if(!['operators','cache','weights','performance'].includes(id))", "if(!['operators','cache','weights','performance','comparison'].includes(id))"),
    (" document.getElementById('operator-controls').hidden=id!=='operators';", " document.getElementById('operator-controls').hidden=id!=='operators';\n document.getElementById('comparison-page').hidden=id!=='comparison';"),
    (" document.getElementById('ds41-controls').hidden=!selected;", " document.getElementById('comparison-tab').disabled=selected;\n document.getElementById('ds41-controls').hidden=!selected;"),
    (" const selected=selectedModel==='deepseek-v4.1-flash';document.getElementById('ds41-root').hidden=!selected;", " const selected=selectedModel==='deepseek-v4.1-flash';\n if(selected&&page==='comparison'){showPage('performance');return;}\n document.getElementById('ds41-root').hidden=!selected;"),
    ("+' · 算子 / KV cache / 显存 / PD 仿真';showPage(location.hash.slice(1)||'operators',false);", "+' · 算子 / KV cache / 显存 / PD 仿真 / 六曲线对比';\n  const requestedPage=location.hash.slice(1)||'operators',page=selectedModel!=='glm53'&&requestedPage==='comparison'?'performance':requestedPage;\n  showPage(page,page!==requestedPage);"),
]
for old, new in edits:
    if new not in html:
        assert html.count(old) == 1, old
        html = html.replace(old, new, 1)
html = html.replace('GLM-5.3-Flash · 算子 / KV cache / 显存 / PD 仿真</title>', 'GLM-5.3-Flash · 算子 / KV cache / 显存 / PD 仿真 / 六曲线对比</title>')

data = json.loads((root / 'data/decode-comparison-b28.json').read_text(encoding='utf-8'))
assert len(data['series']) == 6
asset('style', 'comparison-style', (root / 'src/decode-comparison.css').read_text(encoding='utf-8'))
external_json('comparison-data', 'data/decode-comparison-b28.js', data)
asset('script', 'comparison-script', (root / 'src/decode-comparison.js').read_text(encoding='utf-8'))
path.write_bytes(html.replace('\n',newline).encode('utf-8'))
print('Embedded page 5, six B28-corrected curves and synchronous data assets.')
