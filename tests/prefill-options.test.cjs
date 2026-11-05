const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');

function load(file) {
  const html = fs.readFileSync(file, 'utf8');
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
  const data = {};
  for (const [, attrs, body] of scripts) {
    if (attrs.includes('application/json')) data[attrs.match(/id="([^"]+)"/)[1]] = JSON.parse(body);
    else new vm.Script(body);
  }
  const source = scripts.find(([, , body]) => body.includes('function calculatePDSimulation('))[2];
  const context = vm.createContext({memoryData:data['weight-data'], GiB:2**30,
    document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
  const memory = source.slice(source.indexOf('function calculateRankMemory('), source.indexOf('function readMemoryInputs('));
  const engine = source.slice(source.indexOf('const pdData='), source.indexOf('/* Browser UI;'));
  const exporting = source.slice(source.indexOf('function pdExportData('), source.indexOf('/* Official msModeling'));
  const native = source.slice(source.indexOf('function pdNativePayload('), source.indexOf('function pdOfficialView('));
  vm.runInContext(memory + engine + exporting + native + '\nthis.pdWorld = pdWorld;', context);
  return context;
}
const api = load(path.join(root, 'glm53_prefill_decode_operator_block_diagram.html'));
const plain = value => JSON.parse(JSON.stringify(value));
const base = {p:{hw:'a3-752',nodes:1,tp:8,ep:16,usableGiB:0},d:{hw:'a3-752',nodes:1,tp:1,ep:16,usableGiB:0},
  input:4096,output:128,ttft:100,itl:1,tpot:1,fidelity:'analytic',decodeMode:'ratio',mtpDraft:5,
  fixedP:1,fixedD:1,maxP:4,maxD:4,util:85,reserve:4,chunk:0,pScale:1,dScale:1,
  computeScale:1,memoryScale:1,overlap:0,imbalance:1.1};
const config = (hit=0, pcp=1, extra={}) => ({...structuredClone(base),prefixHitRate:hit,...extra,
  p:{...base.p,pcp,...extra.p},d:{...base.d,...extra.d}});
const run = c => {
  const result = api.calculatePDSimulation(c);
  assert.equal(result.valid,true,JSON.stringify(result.errors));
  assert.equal(result.p.valid,true,JSON.stringify(result.p.errors));
  assert.equal(result.d.valid,true,JSON.stringify(result.d.errors));
  return result;
};
const close = (a,b) => assert.ok(Math.abs(a-b) <= Math.max(1e-12,Math.abs(b)*1e-10), `${a} != ${b}`);
const cold = run(config()), warm = run(config(50));
assert.equal(warm.p.best.plan.cachedTokens,2048);
assert.equal(warm.p.best.plan.processedTokensPerSequence,2048);
assert.equal(warm.p.best.plan.contextTokens,4096);
assert.ok(warm.p.best.seconds<cold.p.best.seconds);
assert.ok(warm.p.best.qps>cold.p.best.qps);
assert.equal(warm.p.best.kvBytes,cold.p.best.kvBytes);
assert.deepEqual(plain(warm.d),plain(cold.d));
close(warm.p.best.tokensPerSecond, warm.p.best.qps*4096);
close(warm.p.best.computedTokensPerSecond, warm.p.best.qps*2048);

// A cached prefix still participates in attention; it is not a shorter cold prompt.
const short = run(config(0,1,{input:2048}));
assert.ok(warm.p.best.breakdown.sfa>short.p.best.breakdown.sfa);
for (const input of [1,65,4096]) {
  const full = run(config(100,1,{input}));
  assert.equal(full.p.best.plan.computedTokens,1);
  assert.equal(full.p.best.plan.headCalls,1);
  assert.ok(Number.isFinite(full.p.best.qps)&&full.p.best.seconds>0);
}

// Summed chunks must match literal execution, including arbitrary warm offsets,
// sparse-cap crossings, KPool residues, remainders, and one final LM head.
for(const pcp of [1,2]) for(const [input,hit,chunk] of [[4099,33.3,65],[4096,50,128],[257,99,64],[4097,0,129]]) {
  const r = run(config(hit,pcp,{input,chunk})), c=r.config, b=r.p.best, plan=b.plan;
  let prefix=plan.cachedTokens, expected=0, workspace=0;
  const times=Object.fromEntries(Object.keys(b.breakdown).map(k=>[k,0]));
  while(prefix<input){
    const q=Math.min(chunk,input-prefix), f=api.pdForward(c,c.p,1,q,prefix,'prefill',prefix+q===input);
    expected+=f.seconds;workspace=Math.max(workspace,f.workspace);
    for(const k in times)times[k]+=f.times[k];
    prefix+=q;
  }
  close(b.seconds,expected);close(b.workspace,workspace);
  for(const k in times)close(b.breakdown[k],times[k]);
  assert.equal(plan.cachedTokens+plan.processedTokensPerSequence,input);
}

const cp = run(config(0,2));
assert.equal(cp.config.p.dp,1);
assert.equal(api.pdWorld(cp.config.p),16);
assert.equal(cp.p.memory.world,16);
assert.equal(cp.p.best.kvBytes,cold.p.best.kvBytes); // PCP does not imply DCP.
assert.equal(cp.p.memory.weight,cold.p.memory.weight);
close(cp.p.best.qps,cp.config.p.dp*cp.p.best.batch/cp.p.best.seconds);
assert.deepEqual(plain(cp.d),plain(cold.d));
const mtpCold=run(config(0,1,{decodeMode:'mtp',output:1024,itl:.09,tpot:.02}));
const mtpWarm=run(config(80,2,{decodeMode:'mtp',output:1024,itl:.09,tpot:.02}));
assert.deepEqual(plain(mtpWarm.d),plain(mtpCold.d));
assert.equal(run(config(50,2,{p:{ep:1}})).p.memory.expertTP,16);
const f1=api.pdForward(cold.config,cold.config.p,1,4096,0,'prefill');
const f2=api.pdForward(cp.config,cp.config.p,1,4096,0,'prefill');
assert.equal(f2.queryTokens,2048);
assert.equal(f2.globalQueryTokens,4096);
assert.equal(f2.kdaParallel,8); // Joint TP*PCP KDA sharding is not supported.
close(f2.times.kda,f1.times.kda);
assert.ok(f2.times.communication>0);
const tp1 = run(config(0,1,{p:{tp:1}})), tp1cp = run(config(0,2,{p:{tp:1}}));
assert.equal(tp1cp.config.p.dp,8);
assert.equal(tp1cp.p.memory.n,32);
assert.ok(tp1cp.p.memory.weight<tp1.p.memory.weight);
assert.equal(api.pdForward(tp1cp.config,tp1cp.config.p,1,4096,0,'prefill').kdaParallel,2);

for(const [pcp,tp,nodes,ep] of [[1,8,1,16],[2,8,1,16],[4,4,1,16],[8,2,1,16],[16,1,1,16],[32,1,2,32],[64,1,4,64]]) {
  const r=run(config(50,pcp,{p:{tp,nodes,ep}}));
  assert.equal(r.config.p.dp*tp*pcp,nodes*16);
}
for(const hit of [-1,101,NaN,Infinity]) {
  // JSON cloning turns NaN/Infinity into null. Null must not be accepted as a default.
  const r=api.calculatePDSimulation(config(hit));assert.equal(r.valid,false,`invalid hit ${hit}`);
}
for(const pcp of [0,3,65,NaN]) {
  const r=api.calculatePDSimulation(config(0,pcp));assert.ok(!r.valid||!r.p.valid,`invalid PCP ${pcp}`);
}
assert.equal(api.calculatePDSimulation(config(0,4)).p.valid,false); // 8*4 > 16 ranks.
assert.equal(api.calculatePDSimulation(config(0,1,{d:{pcp:2}})).valid,false);
const calibrated=run(config(0,1,{fidelity:'calibrated'}));
// The old 4K host was not SKU-identified and cannot calibrate the 752T profile.
assert.equal(calibrated.p.best.calibration.applied,false);
for(const [hit,pcp] of [[50,1],[0,2],[100,2]]) {
  const r=run(config(hit,pcp,{fidelity:'calibrated'}));
  assert.equal(r.p.best.calibration.applied,false);
  assert.equal(r.p.calibrationCap,null);
  assert.throws(()=>api.pdNativePayload(r.config),/Prefix Cache \/ PCP/);
}
assert.doesNotThrow(()=>api.pdNativePayload(cold.config));
const exported=api.pdExportData(warm);
assert.equal(exported.schemaVersion,8);
assert.equal(exported.prefillOptions.cachedTokens,2048);
assert.equal(exported.config.p.pcp,1);
assert.equal(exported.prefillOptions.calibrationEligible,false);

// Optional compatibility check against the pre-change HTML, outside the repository.
if(process.argv[2]) {
  const old=load(path.resolve(process.argv[2]));
  for(const fidelity of ['analytic','ideal','calibrated'])for(const chunk of [0,65,1024]) {
    const c=config(0,1,{fidelity,chunk});const before=old.calculatePDSimulation(c),after=run(c);
    for(const k of ['p','d']) for(const field of ['seconds','qps','workspace','kvBytes','peakBytes'])close(after[k].best[field],before[k].best[field]);
  }
}
console.log('PASS: prefix cache boundaries, chunk integration, PCP topology/KDA/memory, D isolation, calibration, export, validation, native guard and baseline compatibility.');
