const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.pdData=pdData;',ctx);
const base=hw=>({p:{hw,nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw,nodes:1,tp:1,ep:8,usableGiB:0},input:131072,output:1024,ttft:7,itl:.07,tpot:.02,dConstraint:'both',fidelity:'calibrated',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:4,fixedD:4,maxP:256,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:95,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const plain=x=>JSON.parse(JSON.stringify(x));
const run=(hw,changes={})=>{const r=ctx.calculatePDSimulation({...base(hw),...changes});assert.ok(r.valid,JSON.stringify(r.errors));return r;};
const report=[];
for(const chunk of [16384,24576,32768])for(const prefixHitRate of [0,95,100]){
 const opts={chunk,prefixHitRate,ttft:1000},a=run('a5-128',opts),e=run('a5-96',opts),b=a.p.best;
 assert.ok(b.calibration.transfer&&b.calibration.extrapolated&&!b.calibration.targetNative);
 assert.equal(b.calibration.targetEmpiricallyValidated,false);
 assert.equal(b.calibration.sourceHardware,'a5-96');assert.equal(b.calibration.targetHardware,'a5-128');
 assert.equal(b.calibration.measuredCoverage.input,null);
 assert.equal(b.calibration.sourceMeasuredCoverage.input,131072);
 close(a.p.fit.scale,e.p.fit.scale);close(a.p.fit.overhead,e.p.fit.overhead);
 const raw=run('a5-128',{...opts,fidelity:'analytic'}).p.best;
 close(b.plan.rawModelSeconds,raw.seconds);close(b.seconds,a.p.fit.scale*raw.seconds+a.p.fit.overhead);
 close(b.qps,b.batch/b.seconds);assert.ok(b.seconds>=b.plan.hardwareModelFloorSeconds);
 assert.equal(b.plan.cachedTokens,Math.min(131071,Math.floor(131072*prefixHitRate/100)));
 assert.equal(a.p.memory.config.kvDtype,'int8');assert.equal(a.p.memory.usableGiB,112);
 assert.ok(!a.p.memory.observedRuntime&&!a.p.memory.hardwareProfile,'Do not copy 850E runtime memory into 850');
 close(a.p.memory.weight,run('a5-128',{...opts,fidelity:'analytic'}).p.memory.weight);
 assert.ok(!a.d.results[0].calibration.applied,'P transfer must not enable 850 D calibration');
 report.push({chunk,prefixHitRate,batch:b.batch,qps850:b.qps,qps850E:e.p.best.qps,seconds850:b.seconds,seconds850E:e.p.best.seconds,scale:a.p.fit.scale,overhead:a.p.fit.overhead});
}
// Same compute/bandwidth/precision and batch must give identical service times.
// Changing only target bandwidth must never refit the source observations.
const hw=ctx.pdData.devices.find(x=>x.id==='a5-128'),originalBandwidth=hw.memoryBandwidth;
const original=run('a5-128',{kvDtype:'fp8'}).p;
hw.memoryBandwidth=ctx.pdData.devices.find(x=>x.id==='a5-96').memoryBandwidth;
const equal=run('a5-128',{kvDtype:'fp8'}).p,reference=run('a5-96',{kvDtype:'fp8'}).p;
close(equal.best.seconds,reference.best.seconds);close(equal.best.qps,reference.best.qps);
close(equal.fit.scale,original.fit.scale);close(equal.fit.overhead,original.fit.overhead);
assert.ok(equal.best.seconds<original.best.seconds);
hw.memoryBandwidth=originalBandwidth;
for(const change of [{p:{...base('a5-128').p,nodes:2,ep:16}},{p:{...base('a5-128').p,tp:2,pcp:4}},{computeScale:.5,memoryScale:.5},{moeMultistream:false},{util:90},{kvDtype:'int8'}]){
 const r=run('a5-128',{...change,ttft:1000}).p;
 assert.ok(r.best.calibration.transfer);close(r.fit.scale,original.fit.scale);close(r.fit.overhead,original.fit.overhead);
}
for(const fidelity of ['analytic','ideal'])assert.ok(!run('a5-128',{fidelity}).p.best.calibration.applied);
for(const change of [{mtpDraft:3},{kvDtype:'bf16'}])assert.ok(!run('a5-128',change).p.results[0].calibration.transfer);
for(const chunk of [16384,32768])for(const ttft of [1,2,7]){
 const r=run('a5-128',{chunk,ttft,fixedP:0}).p,b=r.best;
 assert.ok(b&&b.seconds<=ttft&&b.peakBytes<=r.memory.limit);
 assert.ok(r.results.every(x=>!x.feasible||x.qps<=b.qps));
 report.push({kind:'optimal',chunk,ttft,batch:b.batch,qps850:b.qps,seconds850:b.seconds});
}
// Keep actual user-visible evidence explicit without reading rendered images.
const elements=new Map(),element=()=>({textContent:'',children:[],style:{},classList:{toggle(){},remove(){}},append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},querySelector(){return element()}});
ctx.pdEl=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
ctx.document={createElement:element,getElementById:id=>ctx.pdEl(id.replace(/^pd-/,''))};ctx.pdNum=(v,n=3)=>Number(v).toFixed(n);ctx.fmtGiB=v=>(v/2**30).toFixed(3)+' GiB';ctx.pdDrawPurePlot=()=>{};ctx.pdLabels=new Proxy({},{get:(_,k)=>String(k)});
for(const name of ['pdDisplayLatency','pdLatencyNumber','pdRenderStage','pdExportData']){const a=src.indexOf('function '+name+'('),b=src.indexOf('\nfunction ',a+1);vm.runInContext(src.slice(a,b),ctx);}
const r=run('a5-128');ctx.pdRenderStage('p',r.p,r.config);
assert.match(elements.get('p-evidence').textContent,/850E 实测效率迁移/);
assert.match(elements.get('p-evidence').textContent,/尚无850本机纯P实测验证/);
assert.match(elements.get('p-evidence').textContent,/缓存命中场景.*外推/);
assert.match(elements.get('p-foot').textContent,/总 token 预算/);
assert.match(elements.get('p-breakdown').title,/不是目标机型实测算子时间/);
assert.equal(ctx.pdExportData(r).prefill.best.calibration.targetEmpiricallyValidated,false);
assert.equal(data['pd-data'].prefillEfficiencyTransfer.decodeChanged,false);
if(process.argv.includes('--report'))console.log(JSON.stringify(report,null,2));
console.log('PASS: P-only A5 efficiency transfer, equal-hardware parity, fixed-source fit, target precision/capacity, SLO optimization, APC extrapolation, renderer and export provenance.');
