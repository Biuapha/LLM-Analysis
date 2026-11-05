const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.pdData=pdData;',ctx);
const base=()=>({p:{hw:'a5-96',nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw:'a5-96',nodes:1,tp:1,ep:8,usableGiB:0},input:131072,output:1024,ttft:15,itl:.07,tpot:.02,dConstraint:'both',fidelity:'calibrated',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:0,fixedD:0,maxP:32,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
for(const [input,batch,qps] of [[131072,4,3.520529532688968],[16384,5,3.9625647851516868]]){
 const r=ctx.calculatePDSimulation({...base(),input}),b=r.d.best;
 assert.ok(r.valid&&b.calibration.targetNative);assert.equal(b.batch,batch);close(b.qps,qps);
 assert.equal(r.d.calibrationCap,batch);assert.equal(r.d.memory.config.kvDtype,'fp8');
 assert.equal(b.projection.acceptanceMeasured,false);close(b.itlTpotRatio,3.5);
 close(b.qps,8*batch/(1023*b.tpotSeconds));assert.equal(b.projection.cycleMeasuredDirectly,false);
 const fixed=ctx.calculatePDSimulation({...base(),input,fixedD:batch+1}).d;
 assert.equal(fixed.best,null);assert.equal(fixed.results.length,0);assert.match(fixed.failure,/超出覆盖/);
 const one=ctx.calculatePDSimulation({...base(),input,fixedD:1,itlTpotRatio:1}).d.results[0];
 const folded=ctx.calculatePDSimulation({...base(),input,fixedD:1,itlTpotRatio:3.5}).d.best;
 // r must never be silently overwritten by measured acceptance.
 close(one.itlSeconds,folded.itlSeconds);close(folded.qps/one.qps,3.5);
 const measured=ctx.calculatePDSimulation({...base(),input,fixedD:1,ratioSource:'reference'}).d.best;
 const family=ctx.pdData.a5Calibration.families.find(f=>f.phase==='decode'&&f.input===input);
 close(measured.tpotSeconds,family.training[0].tpotSeconds);assert.equal(measured.projection.acceptanceMeasured,true);
 // Validation points cannot change the interpolation, even if their values change.
 const before=ctx.calculatePDSimulation({...base(),input,fixedD:3}).d.results[0].itlSeconds;
 family.validation[0].cycle_ms=99999;
 close(ctx.calculatePDSimulation({...base(),input,fixedD:3}).d.results[0].itlSeconds,before);
 assert.ok(!family.training.some(x=>x.batch===3));
}
for(const change of [{input:32768},{mtpDraft:3},{util:81},{chunk:8192},{constraintStatistic:'worst'},{kvDtype:'int8'},{dPrefixLayout:'independent'},{fidelity:'analytic'},{memoryScale:.9},{output:512}]){
 const r=ctx.calculatePDSimulation({...base(),...change});assert.ok(r.valid);
 assert.ok(!r.d.results.some(x=>x.calibration.targetNative),JSON.stringify(change));
}
for(const [input,ttft,seconds,cap] of [[16384,2,1.6104911360362657,8],[131072,15,10.200063539444171,4]]){
 const c={...base(),input,ttft},r=ctx.calculatePDSimulation(c),b=r.p.best;
 assert.ok(b.calibration.targetNative);assert.equal(b.batch,2);close(b.seconds,seconds);close(b.qps,2/seconds);
 assert.equal(r.p.calibrationCap,cap);assert.equal(b.plan.cachedTokens,0);assert.equal(b.plan.pcp,8);
 assert.equal(r.p.memory.block,1152);assert.equal(r.p.memory.blocks,1966);
 assert.equal(ctx.calculatePDSimulation({...c,fixedP:3}).p.best,null,'B3 must fail this TTFT limit');
 assert.match(ctx.calculatePDSimulation({...c,fixedP:cap+1}).p.failure,/超出覆盖/);
 const family=ctx.pdData.a5Calibration.families.find(f=>f.phase==='prefill'&&f.input===input);
 const before=ctx.calculatePDSimulation({...c,fixedP:3}).p.results[0].seconds;
 assert.ok(!family.training.some(x=>x.batch===3));family.validation[0].observedSeconds=99999;
 close(ctx.calculatePDSimulation({...c,fixedP:3}).p.results[0].seconds,before);
 for(const change of [{prefixHitRate:90},{mtpDraft:3},{util:81},{chunk:8192},{p:{...c.p,pcp:1}}]){
  assert.ok(!ctx.calculatePDSimulation({...c,...change}).p.results.some(x=>x.calibration.targetNative));
 }
}
assert.equal(ctx.pdData.a5Calibration.prefillAPCCheck.accepted,false);
assert.ok(!ctx.pdData.a5Calibration.families.some(f=>f.phase==='prefill'&&f.prefixHitRate===90));
const original=JSON.parse(fs.readFileSync(path.join(__dirname,'../data/a5-850e-calibration-20261105.json'),'utf8'));
assert.deepEqual(data['pd-data'].a5Calibration,original);
assert.ok(src.includes(fs.readFileSync(path.join(__dirname,'../src/850e-runtime-calibration.js'),'utf8').trim()));
assert.match(src,/b\?\.calibration\.targetNative\?`850E/);
// Exercise actual renderer text without browser images or an external server.
const elements=new Map();
const element=()=>({textContent:'',children:[],style:{},classList:{toggle(){},remove(){}},append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},querySelector(){return element()}});
ctx.pdEl=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
ctx.document={createElement:element,getElementById:id=>ctx.pdEl(id.replace(/^pd-/,''))};ctx.pdNum=(v,n=3)=>Number(v).toFixed(n);ctx.fmtGiB=v=>(v/2**30).toFixed(3)+' GiB';ctx.pdDrawPurePlot=()=>{};
ctx.pdLabels=new Proxy({},{get:(_,key)=>String(key)});
for(const name of ['pdDisplayLatency','pdLatencyNumber','pdRenderStage']){
 const a=src.indexOf('function '+name+'('),b=src.indexOf('\nfunction ',a+1);vm.runInContext(src.slice(a,b),ctx);
}
const rendered=ctx.calculatePDSimulation({...base(),input:16384});ctx.pdRenderStage('d',rendered.d,rendered.config);
assert.match(elements.get('d-evidence').textContent,/850E/);assert.doesNotMatch(elements.get('d-evidence').textContent,/82\.03%/);
assert.match(elements.get('d-latency-label').textContent,/归一化/);assert.match(elements.get('d-detail').textContent,/FP8/);
assert.equal(elements.get('d-breakdown').children.length,0);
ctx.pdRenderStage('p',rendered.p,rendered.config);
assert.match(elements.get('p-batch-label').textContent,/在途并发/);assert.match(elements.get('p-detail').textContent,/FP8/);
assert.match(elements.get('p-formula').textContent,/在途并发 ÷ 平均 TTFT/);assert.match(elements.get('p-foot').textContent,/调度等待/);
const apc=ctx.calculatePDSimulation({...base(),prefixHitRate:90});ctx.pdRenderStage('p',apc.p,apc.config);
assert.match(elements.get('p-evidence').textContent,/重复冷热输出检查不一致/);assert.match(elements.get('p-evidence').textContent,/仅为估算/);
console.log('PASS: 850E P/D observed bounds and SLOs, explicit r normalization, held-out isolation, rejected APC evidence, software scope and source consistency.');
