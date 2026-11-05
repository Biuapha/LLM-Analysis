"""Synchronize the reviewable 850E data/engine sources into the standalone page."""
import argparse
import json
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent
page = root / 'glm53_prefill_decode_operator_block_diagram.html'
html = page.read_text(encoding='utf8')
data = json.loads((root / 'data/a5-850e-calibration-20261105.json').read_text(encoding='utf8'))
source = (root / 'src/850e-runtime-calibration.js').read_text(encoding='utf8')
multistream = (root / 'src/moe-multistream.js').read_text(encoding='utf8')
a3_source = (root / 'src/a3-hardware-correction.js').read_text(encoding='utf8')

def embed_data(match):
    document = json.loads(match[2])
    document['a5Calibration'] = data
    return match[1] + json.dumps(document, ensure_ascii=False, separators=(',', ':')) + match[3]

updated, data_count = re.subn(r'(<script[^>]*id="pd-data"[^>]*>)([\s\S]*?)(</script>)', embed_data, html)
updated, code_count = re.subn(r'// Atlas 850E runtime observations[\s\S]*?(?=function pdPureStage\()', lambda _: source + '\n', updated)
assert data_count == code_count == 1, 'Expected exactly one embedded data/code section'
updated, stream_count = re.subn(r'// Shared-expert scheduling, pinned[\s\S]*?// End shared-expert scheduling\.\n', lambda _: multistream, updated)
assert stream_count == 1, 'Expected one multistream source section'
updated, a3_count = re.subn(r'// Cross-device estimates[\s\S]*?(?=// Atlas 850E runtime observations)', lambda _: a3_source + '\n', updated)
assert a3_count == 1, 'Expected one A3 correction section'
if args.check:
    # JSON serialization style is immaterial; compare parsed data and source.
    embedded = json.loads(re.search(r'<script[^>]*id="pd-data"[^>]*>([\s\S]*?)</script>', html)[1])
    assert embedded['a5Calibration'] == data and source.strip() in html, 'Run this script to synchronize the page'
    assert multistream.strip() in html, 'Run this script to synchronize multistream scheduling'
    assert a3_source.strip() in html, 'Run this script to synchronize A3 correction'
    print('850E embedded data and source are synchronized')
else:
    page.write_text(updated, encoding='utf8', newline='\n')
    print('Updated 850E embedded data and source')
