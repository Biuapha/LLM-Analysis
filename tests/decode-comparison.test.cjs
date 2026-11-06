const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const baseline = JSON.parse(fs.readFileSync(path.join(root, 'data/decode-comparison-v13.json'), 'utf8'));
const saved = JSON.parse(fs.readFileSync(path.join(root, 'data/decode-comparison-b28.json'), 'utf8'));
const html = fs.readFileSync(path.join(root, 'glm53_prefill_decode_operator_block_diagram.html'), 'utf8');
const loaded = {};
const loaderContext = vm.createContext({document:{createElement:()=>({}),currentScript:{before:node=>{loaded[node.id]=JSON.parse(node.textContent);}}}});
for (const [,src] of html.matchAll(/<script id="(?:cache-data|comparison-data)-loader" src="([^"]+)"/g)) {
  vm.runInContext(fs.readFileSync(path.join(root,src),'utf8'),loaderContext);
}
assert.deepEqual(loaded['comparison-data'], saved, 'Published data must match the corrected source');
assert.deepEqual(loaded['cache-data'],JSON.parse(fs.readFileSync(path.join(root,'data/glm53-cache-reference.json'),'utf8')));
const current = JSON.parse(cp.execFileSync(process.execPath, ['scripts/report-multistream-curves.cjs'], {cwd:root,encoding:'utf8',maxBuffer:4*1024*1024}));
assert.equal(baseline.engine, current.engine);
assert.deepEqual(saved.sharedCase, current.sharedCase);
assert.equal(saved.series.length, 6);
assert.deepEqual(saved.hardwareProfiles,current.hardwareProfiles);
assert.equal(saved.hardwareProfiles.find(h=>h.id==='a5-128').memoryGiB,112);
assert.equal(saved.hardwareProfiles.find(h=>h.id==='a5-128').memoryBandwidth,1.4e12);
assert.equal(saved.series[0].summary.budgetGiB,89.6);
assert.equal(saved.series[0].summary.firstMemoryRejected,73);
assert.equal(saved.series[0].rows[72].memoryFeasible,true);
assert.equal(saved.series[0].rows[73].memoryFeasible,false);
for (let i=0;i<6;i++) {
  const actual=current.series[i], expected=saved.series[i];
  assert.equal(expected.id,actual.id);
  assert.deepEqual(baseline.series[i].rows,actual.rows,'Page-four baseline must remain unchanged');
  const a=saved.anchors[expected.hardware], factor=a.targetITLms/current.series.find(s=>s.hardware===expected.hardware&&s.ranks===8).rows[28].itlMs;
  assert.ok(Math.abs(factor-expected.correctionFactor)<1e-12);
  for (const row of expected.rows.filter(r=>r.batchPerDP)) {
    const raw=actual.rows[row.batchPerDP];
    assert.ok(Math.abs(row.tpotMs-raw.tpotMs*factor)<1e-10);
    assert.ok(Math.abs(row.itlMs-raw.itlMs*factor)<1e-10);
    assert.ok(Math.abs(row.qpsPerCard-row.batchPerDP/(1023*row.tpotMs/1000))<1e-10);
    assert.equal(row.memoryFeasible,raw.memoryFeasible);
    assert.equal(row.peakGiB,raw.peakGiB);
  }
  const candidates=expected.rows.filter(r=>r.batchPerDP&&r.memoryFeasible&&r.tpotMs<=30&&r.itlMs<=105);
  const best=candidates.reduce((a,b)=>a.qpsPerCard>b.qpsPerCard?a:b);
  assert.equal(expected.summary.batchPerDP,best.batchPerDP);
  assert.equal(expected.summary.qpsPerCard,best.qpsPerCard);
  if(expected.ranks===8)assert.ok(Math.abs(expected.rows[28].itlMs-a.targetITLms)<1e-10);
}
assert.equal(saved.series[3].summary.firstMemoryRejected,42);
assert.equal(saved.series[3].summary.batchPerDP,41);
assert.ok(saved.series[3].summary.tpotMs<30,'EP8 850E optimum stops at memory, not the 30ms crossing');
// Exercise the chart, labels, toggles, inspector and exports as pure JS without images.
const nodes=new Map(), downloads=[];
class Element {
  constructor(id=''){this.id=id;this.value=id==='comparison-batch'?'28':'';this.innerHTML='';this.attrs={};this.listeners={};this.dataset={};}
  setAttribute(k,v){this.attrs[k]=String(v);}
  addEventListener(k,v){this.listeners[k]=v;}
  querySelectorAll(){return [...this.innerHTML.matchAll(/data-series-toggle="(\d+)"/g)].map(m=>{const e=new Element();e.dataset.seriesToggle=m[1];return e;});}
  cloneNode(){const e=new Element();e.innerHTML=this.innerHTML;return e;}
  querySelector(){return {remove(){}};}
  click(){} remove(){}
}
const document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);},createElement:()=>new Element(),body:{append(){}}};
document.getElementById('comparison-data').textContent=JSON.stringify(saved);
const context=vm.createContext({document,Blob,URL:{createObjectURL:b=>{downloads.push(b);return 'blob:test';},revokeObjectURL(){}},setTimeout:()=>0});
const renderer=html.match(/<script id="comparison-script"[^>]*>([\s\S]*?)<\/script>/)[1];
assert.equal(renderer.trim(),fs.readFileSync(path.join(root,'src/decode-comparison.js'),'utf8').trim());
vm.runInContext(renderer,context);
const element=id=>document.getElementById('comparison-'+id);
const count=(text,pattern)=>[...text.matchAll(pattern)].length;
assert.equal(count(element('chart').innerHTML,/data-qps-label=/g),6);
assert.equal(count(element('chart').innerHTML,/data-calibration-anchor=/g),2);
for(const s of saved.series)assert.ok(element('chart').innerHTML.includes(s.summary.qpsPerCard.toFixed(3)+' QPS/卡'));
const rectangles=[...element('chart').innerHTML.matchAll(/<rect data-(?:qps|anchor)-label="[^"]+" x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)].map(m=>m.slice(1).map(Number));
assert.equal(rectangles.length,8);
for(let i=0;i<rectangles.length;i++)for(let j=i+1;j<rectangles.length;j++){
  const [x,y,w,h]=rectangles[i],[a,b,c,d]=rectangles[j];
  assert.ok(x+w<=a||a+c<=x||y+h<=b||b+d<=y,'Chart labels must not overlap');
}
const button=new Element();button.dataset.seriesToggle='0';
element('legend').listeners.click({target:{closest:()=>button}});
assert.equal(count(element('chart').innerHTML,/data-qps-label=/g),5);
assert.equal(count(element('chart').innerHTML,/data-calibration-anchor=/g),1);
element('reset').listeners.click();
assert.equal(count(element('chart').innerHTML,/data-qps-label=/g),6);
for(const batch of [0,28,90]){element('batch').value=String(batch);element('batch').listeners.input();assert.ok(!element('inspection').innerHTML.includes('NaN'));}
for(const kind of ['json','csv','svg'])element(kind).listeners.click();
Promise.all(downloads.map(b=>b.text())).then(([j,c,s])=>{
  assert.deepEqual(JSON.parse(j),saved);
  assert.equal(c.replace(/^\uFEFF/,'').trim().split(/\r?\n/).length,547);
  assert.ok(c.includes('baselineTPOTms'));
  assert.equal(count(s,/data-qps-label=/g),6);
  assert.equal(count(s,/data-calibration-anchor=/g),2);
  assert.ok(s.includes('850 100 ms / 850E 62.4 ms'));
  console.log('PASS: 540 corrected points, 6 optima/QPS labels, 2 anchors, unchanged engine/capacity, loaders, toggles, inspector and CSV/JSON/SVG exports; no images read.');
}).catch(error=>{console.error(error);process.exitCode=1;});
