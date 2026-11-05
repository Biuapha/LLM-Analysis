const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);else new vm.Script(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;')),ctx);
assert.ok(src.includes(fs.readFileSync(path.join(__dirname,'../src/moe-multistream.js'),'utf8').trim()));
const base=hw=>({p:{hw,nodes:1,tp:1,pcp:8,ep:hw.startsWith('a3')?16:8,usableGiB:0},d:{hw,nodes:1,tp:hw.startsWith('a3')?2:1,pcp:1,ep:hw.startsWith('a3')?16:8,usableGiB:0},input:131072,output:1024,ttft:100,itl:.105,tpot:.03,dConstraint:'both',fidelity:'analytic',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',dSharedPrefixRate:95,fixedP:1,fixedD:1,maxP:8,maxD:90,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:95,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
let forwards=0;
for(const hw of ['a3-560','a3-752','a5-128','a5-96'])for(const nodes of [1,2,4])for(const tp of [1,2])for(const epOn of [true,false]){
 const c=base(hw),ranks=nodes*(hw.startsWith('a3')?16:8);
 c.p={...c.p,nodes,tp,pcp:4,ep:epOn?ranks:1};c.d={...c.d,nodes,tp,ep:epOn?ranks:1};
 const on=ctx.calculatePDSimulation(c),off=ctx.calculatePDSimulation({...c,moeMultistream:false});
 assert.ok(on.valid&&off.valid);assert.equal(on.config.moeMultistream,true);
 for(const phase of ['prefill','decode'])for(const variant of ['target','draft'])for(const batch of [1,8,41]){
  const stage=on.config[phase==='prefill'?'p':'d'];
  if(!on[phase==='prefill'?'p':'d'].valid)continue;
  const q=phase==='prefill'?Math.floor(c.chunk/batch):variant==='draft'?1:6;
  const a=ctx.pdForward(on.config,stage,batch,q,c.input,phase,true,variant);
  const b=ctx.pdForward(off.config,stage,batch,q,c.input,phase,true,variant);
  const m=a.multistream;assert.ok(m.enabled&&m.savedSeconds>=0&&m.savedSeconds<=m.sharedSeconds+1e-12);
  close(b.seconds-a.seconds,m.savedSeconds*a.scale);
  assert.ok(m.criticalMoESeconds+1e-12>=m.computeFloorSeconds);
  assert.ok(m.criticalMoESeconds+1e-12>=m.hbmFloorSeconds);
  close(a.times.communication,b.times.communication); // No blanket comm discount.
  close(a.times.sfa,b.times.sfa);close(a.times.kda,b.times.kda);
  assert.ok(Object.values(a.times).every(v=>v>=0));assert.ok(a.workspace>=b.workspace);
  forwards++;
 }
}
for(const value of [-1,101,NaN])assert.equal(ctx.calculatePDSimulation({...base('a5-96'),dSharedPrefixRate:value}).valid,false);
assert.equal(ctx.calculatePDSimulation({...base('a5-96'),overlap:50}).valid,false);
const c={...base('a5-96'),fixedD:8};
const independent=ctx.calculatePDSimulation({...c,dPrefixLayout:'independent'}).d.results[0];
for(const rate of [0,50,95,100]){
 const r=ctx.calculatePDSimulation({...c,dSharedPrefixRate:rate}),b=r.d.results[0],m=r.d.memory;
 close(independent.kvBytes-b.kvBytes,7*Math.floor(Math.floor(c.input*rate/100)/m.block)*m.poolBytes);
 close(independent.tpotSeconds,b.tpotSeconds);
}
// A change in prefix-hit rate affects P work, never D physical sharing implicitly.
close(ctx.calculatePDSimulation({...c,prefixHitRate:0}).d.results[0].kvBytes,ctx.calculatePDSimulation({...c,prefixHitRate:95}).d.results[0].kvBytes);
for(const hw of ['a3-560','a5-128','a5-96'])for(const fidelity of ['analytic','ideal','calibrated'])for(const nodes of [1,2]){
 const c=base(hw);c.d={...c.d,nodes,ep:nodes*(hw.startsWith('a3')?16:8)};
 const r=ctx.calculatePDSimulation({...c,fidelity});
 assert.equal(r.d.memory.config.kvDtype,hw==='a5-96'?'fp8':'int8');
 if(hw==='a5-96')close(r.d.memory.usableGiB,91.75);
}
// Native training already enabled multistream. Refitting the same training
// scope must not cancel the user's stream toggle or double-apply the savings.
const native={...base('a5-96'),fidelity:'calibrated',fixedD:8,dSharedPrefixRate:100};
const on=ctx.calculatePDSimulation(native),off=ctx.calculatePDSimulation({...native,moeMultistream:false});
close(on.d.fit.scale,off.d.fit.scale);assert.ok(on.d.fit.trainingMultistream);
assert.ok(off.d.results[0].tpotSeconds>on.d.results[0].tpotSeconds);
assert.equal(ctx.pdCrossContext(on.config,on.config.d,'decode'),null);
assert.equal(ctx.calculatePDSimulation({...base('a3-560'),fidelity:'calibrated'}).d.results[0].calibration.transfer,true);
// Published graph controls: exact event timing agrees with its six-curve model.
for(const [hw,nodes,batch,ms]of [['a5-96',1,41,19.5627389882712],['a5-96',2,58,29.920788486660236],['a5-96',4,50,29.984344024707116],['a5-128',1,43,29.798859882608703],['a5-128',2,40,29.796109743983475],['a5-128',4,38,29.670576920322908]]){
 const c=base(hw);c.d={...c.d,nodes,ep:8*nodes};
 const r=ctx.calculatePDSimulation({...c,fixedD:0}),b=r.d.best;
 assert.equal(b.batch,batch);close(b.tpotSeconds*1000,ms);
 close(b.qps/(nodes*8),batch/(1023*b.tpotSeconds));
 assert.ok(b.memoryFeasible&&b.itlSeconds<=.105&&b.tpotSeconds<=.03);
}
console.log(`PASS: ${forwards} P/D target/draft forwards, shared resource floors, event dependencies, TP/PCP/EP, stream toggle/refit isolation, physical-prefix/precision invariants and six graph optima.`);
