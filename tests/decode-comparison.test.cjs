const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const root = path.join(__dirname, '..');
const saved = JSON.parse(fs.readFileSync(path.join(root, 'data/decode-comparison-v13.json'), 'utf8'));
const html = fs.readFileSync(path.join(root, 'glm53_prefill_decode_operator_block_diagram.html'), 'utf8');
const embedded = JSON.parse(html.match(/<script id="comparison-data"[^>]*>([\s\S]*?)<\/script>/)[1]);
assert.deepEqual(embedded, saved, 'Published data must match the reproducible source');
const current = JSON.parse(cp.execFileSync(process.execPath, ['scripts/report-multistream-curves.cjs'], {cwd:root,encoding:'utf8',maxBuffer:4*1024*1024}));
assert.equal(saved.engine, current.engine);
assert.deepEqual(saved.sharedCase, current.sharedCase);
assert.equal(saved.series.length, 6);
for (let i=0;i<6;i++) {
  const actual=current.series[i], expected=saved.series[i];
  assert.equal(expected.id,actual.id);
  assert.deepEqual(expected.rows,actual.rows,`${expected.hardware} ${expected.id}: all B0–B90 points`);
  assert.deepEqual(expected.summary,actual.summary,'Memory-constrained optimum must agree with page-four engine');
}
assert.equal(saved.series[3].summary.firstMemoryRejected,42);
assert.equal(saved.series[3].summary.batchPerDP,41);
assert.ok(saved.series[3].summary.tpotMs<30,'EP8 850E optimum stops at memory, not the 30ms crossing');
console.log('PASS: all 540 predicted points and 6 constrained optima match the production engine; embedded snapshot identical; 850E memory boundary retained.');
