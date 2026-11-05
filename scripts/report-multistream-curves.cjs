// Reproduce all six comparison curves directly from the standalone web engine.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),cp=require('node:child_process'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;')),ctx);
const series=[];
for(const hw of ['a5-128','a5-96'])for(const nodes of [1,2,4]){
 const ranks=nodes*8,input={p:{hw,nodes:1,tp:1,pcp:8,ep:8,usableGiB:0},d:{hw,nodes,tp:1,pcp:1,ep:ranks,usableGiB:0},input:131072,output:1024,ttft:10,itl:.105,tpot:.03,dConstraint:'both',fidelity:'analytic',moeMultistream:true,decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',dSharedPrefixRate:95,fixedP:1,fixedD:0,maxP:1,maxD:90,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:95,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'};
 const result=ctx.calculatePDSimulation(input),c=result.config,s=c.d,memory=result.d.memory;
 assert.ok(result.valid&&result.d.valid&&result.d.best);
 const rows=[{batchPerDP:0,totalInFlight:0,tpotMs:null,itlMs:null,qpsPerCard:null,memoryFeasible:null}];
 for(let batch=1;batch<=90;batch++){
  const row=ctx.pdBatch(c,s,batch,'decode',memory);
  assert.ok(!row.calibration.applied&&!row.calibration.transfer);
  assert.ok(Math.abs(row.itlSeconds/row.tpotSeconds-3.5)<1e-10);
  rows.push({batchPerDP:batch,totalInFlight:batch*ranks,tpotMs:row.tpotSeconds*1000,itlMs:row.itlSeconds*1000,qpsPerCard:row.qps/ranks,peakGiB:row.peakBytes/2**30,memoryBudgetGiB:memory.limit/2**30,memoryFeasible:row.memoryFeasible,meetsPrevious30msLimit:row.tpotSeconds<=.03});
 }
 const b=result.d.best,last=rows.filter(r=>r.batchPerDP&&r.tpotMs<=30).at(-1);
 series.push({hardware:hw==='a5-128'?'850':'850E',hardwareId:hw,id:`DP${ranks}TP1EP${ranks}`,nodes,ranks,kvDtype:memory.config.kvDtype,config:c,modelBasis:'Published v10 operator model; event-based shared-expert multistream and common theoretical timing.',rows,
  summary:{batchPerDP:b.batch,tpotMs:b.tpotSeconds*1000,qpsPerCard:b.qps/ranks,peakGiB:b.peakBytes/2**30,budgetGiB:memory.limit/2**30,limit:ctx.pdStageBoundary(result.d).kind==='modeled_memory'?'memory':'latency',firstMemoryRejected:result.d.firstMemoryRejected?.batch??null,latencyOnlyBatch:last.batchPerDP,latencyOnlyQpsPerCard:last.qpsPerCard}});
}
const report={generatedAt:new Date().toISOString(),engine:data['pd-data'].engine,commit:cp.execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),productionEngine:true,model:data['pd-data'].multistreamModel,
 sharedCase:{input:131072,output:1024,prefixHitRate:95,dSharedPrefixRate:95,mtpDraft:5,itlTpotRatio:3.5,statistic:'mean',util:80,reserveGiB:4},
 hardwareProfiles:data['pd-data'].devices.filter(d=>['a5-128','a5-96'].includes(d.id)),series};
if(process.argv[2]){fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2));console.log(JSON.stringify(series.map(s=>({hardware:s.hardware,id:s.id,summary:s.summary})),null,2));}
else console.log(JSON.stringify(report,null,2));
