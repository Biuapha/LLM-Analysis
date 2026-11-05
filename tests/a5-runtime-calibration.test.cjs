const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);else new vm.Script(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.pdData=pdData;',ctx);
const base=()=>({p:{hw:'a5-96',nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw:'a5-96',nodes:1,tp:1,ep:8,usableGiB:0},input:131072,output:1024,ttft:15,itl:.07,tpot:.02,dConstraint:'both',fidelity:'calibrated',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:0,fixedD:0,maxP:32,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
for(const input of [16384,131072]){
 const c={...base(),input},r=ctx.calculatePDSimulation(c),s=r.d,b=s.best;
 assert.ok(r.valid&&b.calibration.modelBased&&b.calibration.targetNative&&b.calibration.extrapolated);
 assert.ok(b.batch>5);assert.equal(s.calibrationCap,null);assert.equal(s.memory.config.kvDtype,'fp8');
 assert.ok(s.memory.blocks>s.memory.observedRuntime.numBlocks,'Do not reuse the old allocated pool as capacity');
 assert.equal(b.projection.acceptanceMeasured,false);close(b.itlTpotRatio,3.5);close(b.qps,8*b.batch/(1023*b.tpotSeconds));
 assert.ok(b.peakBytes<=s.memory.limit&&b.tpotSeconds<=c.tpot&&b.itlSeconds<=c.itl);
 assert.equal(ctx.pdStageBoundary(s).kind,'modeled_memory');assert.ok(s.firstMemoryRejected.peakBytes>s.memory.limit);
 for(const row of s.results){assert.ok(row.itlSeconds>=row.plan.hardwareModelFloorSeconds);assert.ok(!row.feasible||row.qps<=b.qps+1e-10);}
 assert.ok(ctx.calculatePDSimulation({...c,fixedD:16}).d.best,'Untested fixed batch must be modeled');
 const one=ctx.calculatePDSimulation({...c,fixedD:8,itlTpotRatio:1}).d.results[0];
 const folded=ctx.calculatePDSimulation({...c,fixedD:8,itlTpotRatio:3.5}).d.results[0];
 close(one.itlSeconds,folded.itlSeconds);close(folded.qps/one.qps,3.5);
 const reference=ctx.calculatePDSimulation({...c,fixedD:8,ratioSource:'reference'}).d.results[0];
 assert.equal(reference.projection.acceptanceMeasured,false,'Reference acceptance does not make a prediction measured');
 assert.ok(ctx.calculatePDSimulation({...c,tpot:.01,itl:.035}).d.best.batch<b.batch);
 assert.ok(ctx.calculatePDSimulation({...c,dPrefixLayout:'independent'}).d.best.batch<b.batch);
 assert.ok(ctx.calculatePDSimulation({...c,util:90}).d.best.qps>b.qps,'More memory must permit further optimization');
 assert.equal(ctx.pdStageBoundary(ctx.calculatePDSimulation({...c,maxD:8}).d).kind,'scan_limit');
 const slower=ctx.calculatePDSimulation({...c,fixedD:16,computeScale:.5,memoryScale:.5}).d.results[0];
 assert.ok(slower.tpotSeconds>ctx.calculatePDSimulation({...c,fixedD:16}).d.best.tpotSeconds);
}
// Held-out validation is independent of model fitting; assert a real error bound
// and verify that changing validation labels cannot move the predictions.
const validation=[];
for(const family of ctx.pdData.a5Calibration.families){
 const key=family.phase==='prefill'?'p':'d';
 for(const v of family.validation){
  const c={...base(),input:family.input,output:key==='p'?1:1024,fixedP:v.batch,fixedD:v.batch,ttft:1000,itl:3.5,tpot:1};
  const row=ctx.calculatePDSimulation(c)[key].best;
  const observed=key==='p'?v.observedSeconds:v.cycle_ms/1000,predicted=key==='p'?row.seconds:row.itlSeconds;
  const error=100*(predicted/observed-1);assert.ok(Math.abs(error)<10,`${family.id} B${v.batch}: ${error}%`);
  validation.push({phase:family.phase,input:family.input,batch:v.batch,observedSeconds:observed,predictedSeconds:predicted,errorPercent:error});
  assert.ok(!row.calibration.fit.training.some(x=>x.input===family.input&&x.batch===v.batch));
  const saved={...v};v.observedSeconds=999999;v.cycle_ms=999999;
  close(ctx.calculatePDSimulation(c)[key].best.seconds,row.seconds);for(const k of Object.keys(v))delete v[k];Object.assign(v,saved);
 }
}
// Context/output/resource changes retain the calibrated hardware model.
for(const change of [{input:32768},{input:262144},{util:81},{chunk:8192},{constraintStatistic:'worst'},{output:512}]){
 const r=ctx.calculatePDSimulation({...base(),...change,fixedD:8});assert.ok(r.valid&&r.d.best.calibration.modelBased);
}
for(const change of [{mtpDraft:3},{kvDtype:'int8'},{fidelity:'analytic'}]){
 assert.ok(!ctx.calculatePDSimulation({...base(),...change}).d.results.some(x=>x.calibration.targetNative));
}
for(const input of [16384,131072]){
 const c={...base(),input,output:1,ttft:1000,fixedP:12},s=ctx.calculatePDSimulation(c).p,b=s.best;
 assert.ok(b&&b.calibration.extrapolated);assert.equal(s.calibrationCap,null);
 assert.ok(b.plan.chunkPerSequence*12<=c.chunk);assert.equal(b.plan.processedTokensPerSequence,input);
 assert.equal(b.plan.chunkBudgetMode,'per_dp');assert.ok(b.seconds>=b.plan.hardwareModelFloorSeconds);
 const cold=ctx.calculatePDSimulation({...c,fixedP:1}).p.best;
 const cached=ctx.calculatePDSimulation({...c,fixedP:1,prefixHitRate:90}).p.best;
 assert.ok(cached.seconds<cold.seconds&&cached.calibration.extrapolated);assert.ok(cached.plan.cachedTokens>0);
}
assert.equal(ctx.pdData.a5Calibration.prefillAPCCheck.accepted,false);
assert.ok(!ctx.pdData.a5Calibration.families.some(f=>f.phase==='prefill'&&f.prefixHitRate===90));
const original=JSON.parse(fs.readFileSync(path.join(__dirname,'../data/a5-850e-calibration-20261105.json'),'utf8'));
assert.deepEqual(data['pd-data'].a5Calibration,original);
assert.ok(src.includes(fs.readFileSync(path.join(__dirname,'../src/850e-runtime-calibration.js'),'utf8').trim()));
// Actual renderer text, without browser images or an external server.
const elements=new Map();
const element=()=>({textContent:'',children:[],style:{},classList:{toggle(){},remove(){}},append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},querySelector(){return element()}});
ctx.pdEl=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
ctx.document={createElement:element,getElementById:id=>ctx.pdEl(id.replace(/^pd-/,''))};ctx.pdNum=(v,n=3)=>Number(v).toFixed(n);ctx.fmtGiB=v=>(v/2**30).toFixed(3)+' GiB';ctx.pdDrawPurePlot=()=>{};
ctx.pdLabels=new Proxy({},{get:(_,key)=>String(key)});
for(const name of ['pdDisplayLatency','pdLatencyNumber','pdRenderStage']){const a=src.indexOf('function '+name+'('),b=src.indexOf('\nfunction ',a+1);vm.runInContext(src.slice(a,b),ctx);}
const rendered=ctx.calculatePDSimulation({...base(),input:16384});ctx.pdRenderStage('d',rendered.d,rendered.config);
assert.match(elements.get('d-evidence').textContent,/未测区域模型预测/);assert.match(elements.get('d-latency-label').textContent,/模型预测/);
assert.match(elements.get('d-detail').textContent,/FP8/);assert.equal(elements.get('d-breakdown').children.length,6);
ctx.pdRenderStage('p',rendered.p,rendered.config);assert.match(elements.get('p-batch-label').textContent,/batch/);assert.match(elements.get('p-foot').textContent,/总 token 预算/);
const apc=ctx.calculatePDSimulation({...base(),prefixHitRate:90});ctx.pdRenderStage('p',apc.p,apc.config);
assert.match(elements.get('p-evidence').textContent,/重复冷热输出检查不一致/);assert.match(elements.get('p-evidence').textContent,/仅为估算/);
if(process.argv.includes('--report'))console.log(JSON.stringify({fit:rendered.d.fit,validation},null,2));
console.log('PASS: unrestricted modeled batch search, hardware floor, physical-memory/SLO limits, explicit r, input/resource sensitivity, held-out isolation/error bounds, APC prediction, renderer and source consistency.');
