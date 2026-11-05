const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.pdData=pdData;',ctx);
const base=hw=>({p:{hw,nodes:1,tp:1,pcp:8,ep:hw.startsWith('a3')?16:8,usableGiB:0},d:{hw,nodes:1,tp:hw.startsWith('a3')?2:1,ep:hw.startsWith('a3')?16:8,usableGiB:0},input:131072,output:1024,ttft:2,itl:.07,tpot:.02,dConstraint:'both',fidelity:'calibrated',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:0,fixedD:0,maxP:32,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const fit=[],validation=[],cases=[];
for(const phase of ['prefill','decode']){
 const c={...base('a5-96'),ttft:1000,tpot:1,itl:3.5,output:phase==='prefill'?1:1024};
 const r=ctx.calculatePDSimulation(c),s=r[phase==='prefill'?'p':'d'];fit.push({phase,...s.fit});
 for(const f of ctx.pdData.a5Calibration.families.filter(x=>x.phase===phase))for(const p of [...f.training,...f.validation]){
  const sim=ctx.calculatePDSimulation({...c,input:f.input,fixedP:p.batch,fixedD:p.batch}),stage=sim[phase==='prefill'?'p':'d'];
  const row=stage.results[0]||stage.firstMemoryRejected,pred=phase==='prefill'?row.seconds:row.itlSeconds,actual=phase==='prefill'?(p.seconds??p.observedSeconds):(p.cycleSeconds??p.cycle_ms/1000);
  validation.push({phase,input:f.input,batch:p.batch,role:f.training.includes(p)?'train':'validation',observed:actual,predicted:pred,errorPercent:100*(pred/actual-1)});
 }
}
for(const hw of ['a3-560','a5-96'])for(const input of [16384,131072])for(const tpot of [.02,.05]){
 const c={...base(hw),input,tpot,itl:tpot*3.5,ttft:input===16384?2:15},r=ctx.calculatePDSimulation(c);
 for(const key of ['p','d']){const s=r[key],b=s.best;cases.push({hw,input,tpot,phase:key,batch:b?.batch,qps:b?.qps,latencyMs:b?.seconds*1000,peakGiB:b?.peakBytes/2**30,budgetGiB:s.memory.limit/2**30,scan:s.scannedBatches,boundary:ctx.pdStageBoundary(s)});}
}
const sourceValidation=[];
for(const family of ctx.pdData.runtimeCalibration.families)for(const v of family.validation||[]){
 const phase=family.phase,key=phase==='prefill'?'p':'d',c={...base('a3-560'),input:v.input,prefixHitRate:family.prefixHitRate||0};
 const context=ctx.pdCrossContext(c,c[key],phase),a=ctx.pdCrossAnchor(c,context,family,v.batch,phase);
 const actual=phase==='prefill'?v.measuredSeconds:v.itlMeanSeconds;
 sourceValidation.push({family:family.id,input:v.input,batch:v.batch,observed:actual,predicted:a.observed,errorPercent:100*(a.observed/actual-1)});
}
const ref=ctx.pdData.a5Calibration.prefillBudgetReference,p=ref.crossCheck;
const row=ctx.calculatePDSimulation({...base('a5-96'),output:1,input:p.input,ttft:100,prefixHitRate:p.prefixHitRate,chunk:ref.chunk,util:ref.util,fixedP:p.batch}).p.best;
const workbookCrossCheck={reference:ref.id,role:'cross-check-visible-during-revision-not-blind-validation',observed:p.seconds,predicted:row.seconds,errorPercent:100*(row.seconds/p.seconds-1)};
const report={fit,validation,a3SourceInterpolationValidation:sourceValidation,a3TargetEmpiricallyValidated:false,workbookCrossCheck,cases};console.log(JSON.stringify(report,null,2));
