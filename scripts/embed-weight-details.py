"""Refresh the weight-detail assets embedded in the standalone HTML."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'glm53_prefill_decode_operator_block_diagram.html'
original = path.read_bytes().decode('utf-8')
newline = '\r\n' if '\r\n' in original else '\n'
html = original.replace('\r\n', '\n')

def replace_block(tag, block_id, body, attrs=''):
    global html
    block = f'<{tag} id="{block_id}"{attrs}>\n{body}\n</{tag}>'
    pattern = rf'<{tag} id="{block_id}"[^>]*>[\s\S]*?</{tag}>'
    if re.search(pattern, html):
        html = re.sub(pattern, lambda _: block, html, count=1)
    elif tag == 'style':
        html = html.replace('</head>', block + '\n</head>', 1)
    else:
        marker = "<script>\nconst cacheData="
        assert marker in html, 'Main runtime marker not found'
        html = html.replace(marker, block + '\n' + marker, 1)

data = json.loads((root / 'data/operator-weight-catalog.json').read_text(encoding='utf-8'))
replace_block('style', 'operator-weight-style', (root / 'src/operator-weight-details.css').read_text(encoding='utf-8'))
replace_block('script', 'operator-weight-data', json.dumps(data, ensure_ascii=False, separators=(',', ':')), ' type="application/json"')
replace_block('script', 'operator-weight-script', (root / 'src/operator-weight-details.js').read_text(encoding='utf-8'))

edits = [
    (" if(!kind){\n  dfEl('math-title')", " if(!kind){\n  OperatorWeightDetails.render(dfEl('detail'),null);\n  dfEl('math-title')"),
    (" const node=kind==='node'?dfById(id):dfById(edges[0].src);", " const node=kind==='node'?dfById(id):dfById(edges[0].src);\n OperatorWeightDetails.render(dfEl('detail'),OperatorWeightDetails.glm(dfFamily,node.id,dfPhase),node.title);"),
    ("   set('flow-detail-title',edge||n.title);", "   OperatorWeightDetails.render(el('flow-detail-title').closest('.df-detail'),OperatorWeightDetails.deepseek(ui.family,n.id,ui.view),n.title);\n   set('flow-detail-title',edge||n.title);"),
    ("else{ui.node=null;set('flow-detail-title',graph.label);", "else{ui.node=null;OperatorWeightDetails.render(el('flow-detail-title').closest('.df-detail'),null);set('flow-detail-title',graph.label);")
]
for old, new in edits:
    if new not in html:
        assert html.count(old) == 1, old
        html = html.replace(old, new, 1)
path.write_bytes(html.replace('\n', newline).encode('utf-8'))
print('Updated embedded catalog, styles, renderer and selection hooks.')
