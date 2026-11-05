const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.pdData=pdData;',ctx);
const presets=JSON.parse(html.match(/const pdWorkbookCases=(\[[^\n]+\]);/)[1]);
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
for(const phase of ['prefill','decode']){
 const key=phase==='prefill'?'p':'d',base={...presets.find(p=>p.hardware==='a3-560'&&p.row===(key==='p'?4:9)).config,fixedP:1,fixedD:8};
 const initial=ctx.calculatePDSimulation(base)[key].results[0];assert.ok(initial.calibration.transfer);
 for(const change of [{util:80},{util:92},{chunk:8192},{moeMultistream:false}]){
  const r=ctx.calculatePDSimulation({...base,...change})[key].results[0];assert.ok(r.calibration.transfer);
  close(r.transfer.observedSourceSeconds,initial.transfer.observedSourceSeconds);
  close(r.transfer.sourceProxySeconds,initial.transfer.sourceProxySeconds);
  assert.equal(r.transfer.targetEmpiricallyValidated,false);
 }
 const off=ctx.calculatePDSimulation({...base,moeMultistream:false})[key].results[0];
 assert.ok(off.seconds>initial.seconds,'A stream toggle changes target cost, not its fitted source');
}
const cache=[];
for(const hit of [0,.0001,89.6484,90,95,100]){
 const c={...presets[2].config,prefixHitRate:hit,fixedP:1},r=ctx.calculatePDSimulation(c).p.results[0];
 assert.ok(r.calibration.transfer);assert.equal(r.plan.cachedTokens,Math.min(c.input-1,Math.floor(c.input*hit/100)));cache.push(r.seconds);
}
assert.ok(Math.abs(cache[0]-cache[1])/cache[0]<.001,'No cold-to-warm correction cliff');
assert.ok(cache[4]<cache[3],'Requested 95% cache is not overwritten with a 90% source measurement');
const a5=presets.find(p=>p.hardware==='a5-96'&&p.row===9).config;
const native=ctx.calculatePDSimulation({...a5,fixedD:8}).d;
for(const nodes of [2,4]){
 const s=ctx.calculatePDSimulation({...a5,fixedD:8,d:{...a5.d,nodes,ep:8*nodes}}).d,b=s.results[0];
 assert.ok(b.calibration.modelBased&&b.calibration.transfer&&b.calibration.extrapolated);
 assert.equal(b.calibration.targetNative,false);close(s.fit.scale,native.fit.scale);close(s.fit.overhead,native.fit.overhead);
 assert.ok(!s.memory.observedRuntime,'EP8 allocator geometry must not become EP16/32 geometry');
 assert.ok(s.memory.weight<native.memory.weight,'Target expert sharding changes the capacity model');
}
// The warm workbook cross-check must not enter the new cold-budget fit.
const pbase=presets.find(p=>p.hardware==='a5-96'&&p.row===4).config;
const ref=ctx.pdData.a5Calibration.prefillBudgetReference;
const cold=ctx.calculatePDSimulation({...pbase,fixedP:ref.training.batch}).p.results[0];
close(cold.seconds,ref.training.seconds);
const warm=ctx.calculatePDSimulation({...pbase,ttft:100,prefixHitRate:90,fixedP:ref.crossCheck.batch}).p.results[0];
assert.ok(Math.abs(warm.seconds/ref.crossCheck.seconds-1)<.01);
const saved=ref.crossCheck.seconds;ref.crossCheck.seconds=9999;
close(ctx.calculatePDSimulation({...pbase,ttft:100,prefixHitRate:90,fixedP:16}).p.results[0].seconds,warm.seconds);ref.crossCheck.seconds=saved;
assert.equal(warm.calibration.fit.budgetCorrection.blindValidation,false);
assert.ok(src.includes(fs.readFileSync(path.join(__dirname,'../src/a3-hardware-correction.js'),'utf8').trim()));
console.log('PASS: fixed-source A3 corrections across stream/budget/cache changes; A5 topology fit continuity and memory isolation; cold-only workbook fit and disclosed APC cross-check.');
