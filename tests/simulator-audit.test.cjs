// Cross-engine invariants and actual UI function regression checks. No screenshots.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);else new vm.Script(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;')),ctx);
const dsSource=scripts.find(([,a,b])=>b.includes('const DS41=(()=>{'))[2];
const ds=dsSource.slice(dsSource.indexOf('const DS41=(()=>{'),dsSource.indexOf(' const read=()=>',dsSource.indexOf('const DS41=(()=>{')));
vm.runInContext(ds+'return {defaults:cfgDefault,calculate,validate,stage,memory,candidate,forward};})();this.ds=DS41;',ctx);
const close=(a,b)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<1e-9*Math.max(1e-6,Math.abs(b)),`${a} != ${b}`);
const base=(extra={})=>({p:{hw:'a5-96',nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw:'a5-96',nodes:1,tp:1,ep:8,usableGiB:0},input:4096,output:128,ttft:100,itl:10,tpot:10,dConstraint:'both',fidelity:'analytic',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:1,fixedD:1,maxP:4,maxD:4,util:80,reserve:4,chunk:1024,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean',...extra});
const run=c=>{const r=ctx.calculatePDSimulation(c);assert.equal(r.valid,true,JSON.stringify(r.errors));return r;};
let cases=0;
// Reference membership: complete four-token pools plus the causal tail.
for(const prefix of [0,1,2,3,4,2046,2047,2048,2049,131071])for(const q of [1,2,4,6,64]){
 const r=run(base({input:Math.max(1,prefix),output:8}));
 const f=ctx.pdForward(r.config,r.config.d,2,q,prefix,'decode',true);
 let pools=0,selected=0;for(let t=prefix+1;t<=prefix+q;t++){pools+=Math.floor(t/4);selected+=4*Math.min(Math.floor(t/4),512)+t%4;}
 close(f.pooledPairs,2*pools);close(f.selectedPairs,2*selected);
}
for(const decodeMode of ['ratio','mtp'])for(const output of [1,2,3,6,7,8,128,1024])for(const input of [1,3,64,2044,2048,2050,4096]){
 const r=run(base({decodeMode,output,input,chunk:0}));cases++;
 if(output===1){assert.equal(r.d.notApplicable,true);assert.equal(r.d.best,null);assert.equal(r.d.results.length,0);continue;}
 const b=r.d.results[0];assert.ok(b);close(b.duration,(output-1)*b.tpotSeconds);close(b.qps,r.config.d.dp*b.batch/b.duration);
 assert.ok(b.itlTpotRatio<=Math.min(output-1,decodeMode==='mtp'?6:1));
 if(decodeMode==='mtp'){assert.ok(b.plan.verificationTokens<=output-1);assert.equal(b.plan.draftTokens,b.plan.verificationTokens-1);if(output===2)close(b.plan.rawDraftSeconds,0);}
 else{
  assert.equal(b.plan.steps,output-1);assert.equal(b.plan.lastContext,input+output-1);
  // Independently enumerate every token; catch integration/cutoff mistakes.
  let seconds=0;for(let i=0;i<output-1;i++)seconds+=ctx.pdForward(r.config,r.config.d,1,1,input+i,'decode',true).seconds;
  close(b.duration,seconds);
 }
}
// Shared prefix saves exactly (B-1) copies of complete prefix blocks, with identical compute.
for(const decodeMode of ['ratio','mtp']){
 const a=run(base({decodeMode,fixedD:4,input:16384})),b=run(base({decodeMode,fixedD:4,input:16384,dPrefixLayout:'independent'}));
 const shared=a.d.results[0],ind=b.d.results[0],m=a.d.memory;
 close(ind.kvBytes-shared.kvBytes,3*Math.floor(a.config.input/m.block)*m.poolBytes);close(shared.duration,ind.duration);
 assert.match(shared.plan.cacheAllocation.basis,/shared_prefix/);
}
// Dtype cannot differ between the memory model and fallback operator costs.
const int8=run(base({fidelity:'calibrated',input:16384,output:1024,fixedD:33,kvDtype:'int8'}));
assert.equal(int8.d.memory.config.kvDtype,'int8');assert.equal(Boolean(int8.d.results[0]?.calibration.transfer),false);
const normalized=int8.config,m=int8.d.memory;
const inferred=ctx.pdBatch(normalized,normalized.d,33,'decode',m),explicit=ctx.pdBatch({...normalized,kvDtype:'int8'},normalized.d,33,'decode',m);
close(inferred.duration,explicit.duration);
const fp8=run(base({fidelity:'calibrated',input:16384,output:1024,fixedD:33}));
assert.equal(fp8.d.memory.config.kvDtype,'fp8');assert.equal(fp8.d.best.calibration.targetNative,true);
const fp8Ideal=run({...fp8.config,fidelity:'ideal'});
assert.equal(fp8Ideal.d.memory.hardwareProfile,fp8.d.memory.hardwareProfile);
assert.ok(fp8.d.best.itlSeconds>=fp8Ideal.d.best.itlSeconds);
const reference=run(base({ratioSource:'reference'}));assert.equal(reference.d.results[0].itlTpotRatio,1);
assert.equal(run(base({decodeMode:undefined})).d.results[0].itlTpotRatio,1);
for(const [k,v]of [['fixedD',NaN],['fixedP',null],['input',Infinity],['itlTpotRatio',NaN],['itlTpotRatio',0],['itlTpotRatio',8],['constraintStatistic','p99'],['dPrefixLayout','unknown'],['kvDtype','fp4']])assert.equal(ctx.calculatePDSimulation(base({[k]:v})).valid,false,k+' '+v);
const source=base({fidelity:'calibrated',input:16384,output:1024,dConstraint:'tpot',chunk:16384,ratioSource:'custom'});source.p={...source.p,hw:'a3-800',ep:16};source.d={...source.d,hw:'a3-800',tp:4,ep:16};
const rejected=run(source);assert.equal(rejected.d.valid,false);assert.match(rejected.d.failure,/接受产出/);
assert.equal(run({...source,ratioSource:'reference'}).d.valid,true);
assert.equal(run({...source,ratioSource:'reference',kvDtype:'bf16'}).d.valid,false);
// Multiple nodes, all hardware profiles, TP/PCP and expert-TP/EP paths.
for(const hardware of data['pd-data'].devices.filter(h=>!h.empiricalOnly))for(const nodes of [1,2])for(const tp of [1,2,4,8])for(const epOn of [true,false]){
 const ranks=hardware.ranksPerNode*nodes,ep=epOn?ranks:1;
 const c=base({input:4096,output:32,fixedP:0,fixedD:0,maxP:3,maxD:3,p:{hw:hardware.id,nodes,tp,ep,pcp:tp===1?2:1,usableGiB:0},d:{hw:hardware.id,nodes,tp,ep,usableGiB:0}});
 const r=run(c);cases++;
 for(const side of ['p','d']){
  const stage=r[side];assert.ok(stage.valid,stage.failure);assert.equal(stage.config.dp*tp*(side==='p'?c.p.pcp:1),ranks);
  const rows=[...stage.results,...(stage.firstMemoryRejected?[stage.firstMemoryRejected]:[])];let previous=0;
  for(const row of rows){assert.ok(Number.isFinite(row.duration)&&row.duration>0);assert.ok(row.peakBytes>=previous);previous=row.peakBytes;close(row.peakBytes,stage.memory.weight+stage.memory.reserve+row.kvBytes+row.workspace);}
  assert.equal(stage.best?.qps??null,stage.feasible.length?Math.max(...stage.feasible.map(x=>x.qps)):null);
 }
 if(!epOn){const f=ctx.pdForward(r.config,r.config.d,2,1,4096,'decode',true);close(f.scratch.ffn,Math.max(2*12288/tp*6,f.route.localExperts*f.route.tokensPerExpert*2048/f.route.expertTP*6*c.imbalance+2*2048/tp*6));}
}
// Verify selected statistics preserve the same raw candidate costs and only change eligibility.
const mean=run(base({fixedD:0,maxD:8})),worst=run(base({fixedD:0,maxD:8,constraintStatistic:'worst'}));
mean.d.results.forEach((r,i)=>close(r.duration,worst.d.results[i].duration));
// DeepSeek shares the first-token contract and truncates short DSpark cycles.
for(const decode of ['ar','dspark'])for(const output of [1,2,3,6,7,128]){
 const r=ctx.ds.calculate({...ctx.ds.defaults(),decode,output,maxP:1,maxD:2,itl:10,tpot:10});cases++;
 assert.equal(r.valid,true,JSON.stringify(r.errors));
 if(output===1){assert.equal(r.d.notApplicable,true);assert.equal(r.d.best,null);continue;}
 const b=r.d.rows[0]||r.d.rejected;close(b.seconds,(output-1)*b.tpot);close(b.qps,r.config.d.dp*b.batch/b.seconds);
 assert.ok(b.details.targetTokens<=output-1);assert.ok(b.details.acceptedTokens<=output-1);
 if(output===2)assert.equal(b.details.draftPositions,0);
}
for(const field of ['p','d']){
 assert.equal(ctx.ds.calculate({...ctx.ds.defaults(),[field]:null}).valid,false);
 assert.equal(ctx.ds.calculate({...ctx.ds.defaults(),[field]:{hw:'a3-800',nodes:1,tp:1,ep:16}}).valid,false);
}
// UI functions are executed against a minimal DOM, including plot tooltips.
class Element{
 constructor(){this.textContent='';this.children=[];this.style={};this.classList={toggle(){},remove(){},add(){}};this.selectors=new Map();}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}setAttribute(){}
 querySelector(s){if(!this.selectors.has(s))this.selectors.set(s,new Element());return this.selectors.get(s);}
}
const elements=new Map(),el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
ctx.document={getElementById:el,createElement:()=>new Element()};ctx.svgNode=(tag,attrs,text)=>{const e=new Element();e.tag=tag;e.attrs=attrs;e.textContent=text;return e;};ctx.fmtGiB=n=>(n/2**30).toFixed(2);
vm.runInContext('let pdResult=null;'+src.slice(src.indexOf('function pdDisplayLatency('),src.indexOf('function pdUpdateDP(')),ctx);
ctx.testResult=mean;vm.runInContext('pdResult=testResult;pdRenderStage("d",pdResult.d,pdResult.config)',ctx);
const best=mean.d.best;
assert.equal(el('pd-d-candidates').children[0].children[2].textContent,(1000*best.itlSeconds).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}));
assert.match(el('pd-d-itl-label').textContent,/平均/);
assert.ok(el('pd-d-plot').children.filter(x=>x.tag==='circle').every(x=>x.children[0].textContent.includes('平均 TPOT')));
ctx.testResult=run(base({output:1}));vm.runInContext('pdResult=testResult;pdRenderStage("d",pdResult.d,pdResult.config)',ctx);
assert.equal(el('pd-d-qps').textContent,'—');assert.match(el('pd-d-status').textContent,/无需 Decode/);assert.equal(el('pd-d-candidates').children.length,0);
vm.runInContext(src.slice(src.indexOf('function pdNativePayload('),src.indexOf('function pdOfficialView(')),ctx);
assert.throws(()=>ctx.pdNativePayload(source),/800T/);
const native=base({fidelity:'msmodeling',decodeMode:'ratio',itl:.07,tpot:.02});native.p.pcp=1;
const payload=ctx.pdNativePayload(native);assert.equal(payload.dConstraint,'both');assert.equal(payload.itlTpotRatio,1);assert.equal(payload.tpot,.02);
// The legacy 850 key must use the user-corrected 112 GiB in both stages and
// explicitly forward that ceiling to an adapter with an older reference profile.
const atlas850={...native,p:{...native.p,hw:'a5-128',pcp:1,ep:8,usableGiB:0},d:{...native.d,hw:'a5-128',tp:1,ep:8,usableGiB:0}};
const atlasPayload=ctx.pdNativePayload(atlas850);
assert.equal(atlasPayload.p.usableGiB,112);assert.equal(atlasPayload.d.usableGiB,112);
assert.equal(atlas850.p.usableGiB,0,'Preparing a native request must not mutate form input');
assert.equal(ctx.pdNativePayload({...atlas850,d:{...atlas850.d,usableGiB:100}}).d.usableGiB,100);
const atlasCase=run(base({p:{hw:'a5-128',nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw:'a5-128',nodes:1,tp:1,ep:8,usableGiB:0},input:131072,output:1024,dSharedPrefixRate:95}));
const atlasHardware=vm.runInContext('pdData.devices',ctx).find(h=>h.id==='a5-128');
assert.equal(atlasHardware.memoryBandwidth,1.4e12,'1.4 TB/s must be decimal bytes/s, not TiB/s');
assert.equal(atlasHardware.memoryEfficiency,.8);
let earlierBandwidth;
try{atlasHardware.memoryBandwidth=1.6*2**40;earlierBandwidth=run(atlasCase.config);}finally{atlasHardware.memoryBandwidth=1.4e12;}
assert.ok(atlasCase.d.results[0].tpotSeconds>earlierBandwidth.d.results[0].tpotSeconds,'Lower HBM bandwidth must affect decode timing');
assert.ok(atlasCase.p.results[0].seconds>=earlierBandwidth.p.results[0].seconds,'Prefill must also consume the corrected profile');
close(atlasCase.d.results[0].peakBytes,earlierBandwidth.d.results[0].peakBytes);
for(const stage of ['p','d']){close(atlasCase[stage].memory.usableGiB,112);close(atlasCase[stage].memory.limit/2**30,89.6);}
const at72=run({...atlasCase.config,fixedD:72}).d.results[0];
const at73=run({...atlasCase.config,fixedD:73}).d.firstMemoryRejected;
assert.ok(at72.memoryFeasible&&at72.peakBytes<=89.6*2**30);
assert.equal(at73.batch,73);assert.ok(!at73.memoryFeasible&&at73.peakBytes>89.6*2**30);
assert.equal(ctx.calculatePDSimulation({...atlasCase.config,d:{...atlasCase.config.d,usableGiB:128}}).d.valid,false);
vm.runInContext('let pdNativeSettings=null;const pdNativeCopy=[];'+src.slice(src.indexOf('function pdNativeControls('),src.indexOf('function pdNativeReset(')),ctx);
for(const [k,v]of Object.entries({'ratio':'3.5','decode-mode':'mtp','ratio-source':'custom',constraint:'tpot',chunk:'1024'}))el('pd-'+k).value=v;
ctx.pdNativeControls(true);assert.equal(el('pd-constraint').value,'both');assert.equal(el('pd-constraint').disabled,true);assert.equal(el('pd-ratio').disabled,true);
ctx.pdNativeControls(false);assert.equal(el('pd-constraint').value,'tpot');assert.equal(el('pd-ratio').value,'3.5');assert.equal(el('pd-ratio').disabled,false);
// Restore a current saved task using the real async handler and a local mock service.
vm.runInContext(src.slice(src.indexOf('async function pdRestoreNative('),src.indexOf('/* Top-level dialog',src.indexOf('async function pdRestoreNative('))),ctx);
ctx.clearTimeout=()=>{};ctx.pdTimer=null;ctx.pdNativeRevision=0;ctx.AbortSignal=AbortSignal;
ctx.pdNativeReset=()=>{ctx.pdNativeRevision++;};ctx.pdNativeConnect=async()=>'http://127.0.0.1:18753';
const restored={...payload,latencyInputMode:'independent_acceptance'},job={id:'audit',config:restored};
ctx.pdNativeSaved=()=>({id:job.id,config:restored});ctx.fetch=async()=>({ok:true,json:async()=>job});
let applied=0,polled=0;ctx.pdApplyNativeConfig=()=>{applied++;};ctx.pdNativeSave=()=>{};ctx.pdNativeInputs=()=>({});ctx.readPDInputs=()=>restored;ctx.pdPollNative=async()=>{polled++;};ctx.pdNativeFailure=e=>{throw e;};
(async()=>{await ctx.pdRestoreNative();assert.equal(applied,1);assert.equal(polled,1);
 console.log(`PASS: ${cases} cross-engine/topology cases, exact token integration, KPool membership, shared KV, dtype consistency, invalid input, calibration isolation, statistics UI, native preflight/controls/restore.`);
})().catch(e=>{console.error(e);process.exitCode=1;});
