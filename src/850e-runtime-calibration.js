// Atlas 850E runtime observations calibrate a hardware model, never its search range.
// The fitted phase-wide scale/overhead use training points only; unseen batch and
// context are evaluated by the operator model and bounded by the ideal roofline.
function pdA5Family(c,s,phase){
 // P-only efficiency transfer: use the same measured service-time correction
 // for both A5 devices, while evaluating each target's own operator costs.
 // This does not transfer 850E allocator geometry or its Decode calibration.
 const prefillTransfer=phase==='prefill'&&s.hw==='a5-128';
 if(c.fidelity!=='calibrated'||!pdData.a5Calibration||!(s.hw==='a5-96'||prefillTransfer)||
  c.decodeMode!=='mtp'||c.mtpDraft!==5||
  (c.kvDtype!==undefined&&c.kvDtype!=='fp8'&&!(prefillTransfer&&c.kvDtype==='int8')))return null;
 return pdData.a5Calibration.families.filter(f=>f.phase===phase&&!f.prefixHitRate)
  .sort((a,b)=>Math.abs(Math.log(c.input/a.input))-Math.abs(Math.log(c.input/b.input)))[0]||null;
}
function pdA5Profile(c,s,phase){
 if(!pdData.a5Calibration||s.hw!=='a5-96'||s.nodes!==1||s.tp!==1||s.ep!==8||
  c.decodeMode!=='mtp'||c.mtpDraft!==5||(c.kvDtype!==undefined&&c.kvDtype!=='fp8'))return null;
 const families=pdData.a5Calibration.families.filter(f=>f.phase===phase&&f.pcp===pdPCP(s)&&(!f.prefixHitRate));
 return families.sort((a,b)=>Math.abs(Math.log(c.input/a.input))-Math.abs(Math.log(c.input/b.input)))[0]||null;
}
// Interpolation is retained only for the reference acceptance assumption and
// observed comparisons. It never supplies the modeled latency or batch ceiling.
function pdA5Reference(f,batch){
 const rows=f.training.slice().sort((a,b)=>a.batch-b.batch),x=Math.max(rows[0].batch,Math.min(rows.at(-1).batch,batch));
 const a=rows.filter(r=>r.batch<=x).at(-1),b=rows.find(r=>r.batch>=x);
 const at=k=>a.batch===b.batch?a[k]:a[k]+(b[k]-a[k])*(x-a.batch)/(b.batch-a.batch);
 return f.phase==='decode'?{cycleSeconds:at('cycleSeconds'),tpotSeconds:at('tpotSeconds')}:{seconds:at('seconds')};
}
function pdA5Memory(c,s,phase,f){
 const m=f.runtimeMemory,total=s.usableGiB||m.visibleBytes/GiB;
 const memory=calculateRankMemory({kvDtype:'fp8',dp:s.dp,tp:s.tp,ep:s.ep,total,util:c.util,reserve:c.reserve,
  q:phase==='decode'?c.mtpDraft:0,sp:'off',vision:'off',casts:true,quant:'checkpoint',pcp:pdPCP(s)});
 // Same-profile loaded weights and bytes per physical block are observations.
 // Recompute the available pool from the requested budget; do not copy the
 // allocator's old numBlocks/max-running/peak allocation as a capacity limit.
 const weight=m.loadedWeightBytes,poolBytes=m.kvAllocatedBytes/m.numBlocks;
 const rawBudget=memory.limit-weight-memory.reserve,kvBudget=Math.max(0,rawBudget),blocks=Math.floor(kvBudget/poolBytes);
 return {valid:memory.valid,errors:memory.errors,config:memory.config,total:memory.total,limit:memory.limit,reserve:memory.reserve,
  weight,weightMin:weight,analyticWeightBytes:memory.weight,block:m.blockSize,poolBytes,rawBudget,kvBudget,blocks,kv:blocks*poolBytes,
  alignment:kvBudget-blocks*poolBytes,deficit:Math.max(0,-rawBudget),hardwareProfile:'a5-850e-r71-fp8',
  usableGiB:total,capacityBasis:'850E 可见容量、同画像权重/物理页成本 + 动态 KV/MTP 状态/workspace 预算；面向优化配置的容量模型',
  observedRuntime:{profile:pdData.a5Calibration.profile,allocatedPeakBytes:m.peakAllocatedBytes,kvAllocatedBytes:m.kvAllocatedBytes,numBlocks:m.numBlocks},
  allocationAssumption:'统一 block-ID 成本；共享完整前缀 + 每请求私有尾部和 5 个状态保护块；非 r71 分配器逐组重放'};
}
function pdA5Raw(c,s,batch,phase,memory,ideal=false){
 const analytic={...c,fidelity:ideal?'ideal':'analytic',kvDtype:memory.config.kvDtype,chunkBudgetMode:'per_dp',...(ideal?{pScale:1,dScale:1,overlap:0,imbalance:1}:{})};
 return pdBatch(analytic,s,batch,phase,memory);
}
function pdA5Fit(c,s,phase){
 const points=[];
 for(const f of pdData.a5Calibration.families.filter(f=>f.phase===phase&&!f.prefixHitRate)){
  const anchor={...c,input:f.input,output:phase==='decode'?f.output:1,prefixHitRate:0,chunk:16384,
   fidelity:'analytic',moeMultistream:true,dSharedPrefixRate:100,util:80,reserve:4,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,
   ratioSource:'custom',itlTpotRatio:1,dPrefixLayout:'shared',constraintStatistic:'mean'};
  // Fit against the observed topology, regardless of the requested target EP/TP.
  // Target changes must neither refit the source nor silently drop calibration.
  const stage={hw:'a5-96',nodes:1,dp:f.dp,tp:f.tp,ep:f.ep,pcp:f.pcp,usableGiB:0},memory=pdA5Memory(anchor,stage,phase,f);
  for(const r of f.training){
   const row=pdA5Raw(anchor,stage,r.batch,phase,memory),x=phase==='decode'?row.itlSeconds:row.seconds,y=phase==='decode'?r.cycleSeconds:r.seconds;
   points.push({input:f.input,batch:r.batch,x,y,weight:(r.repetitions||1)/(y*y)});
  }
 }
 // Weighted least squares of y = scale * roofline_cost + nonnegative overhead.
 // Relative errors prevent long P requests from dominating short requests.
 let xx=0,xw=0,ww=0,xy=0,wy=0;
 for(const p of points){xx+=p.weight*p.x*p.x;xw+=p.weight*p.x;ww+=p.weight;xy+=p.weight*p.x*p.y;wy+=p.weight*p.y;}
 const candidates=[{scale:xy/xx,overhead:0},{scale:0,overhead:wy/ww}],det=xx*ww-xw*xw;
 if(det>1e-15){const scale=(xy*ww-wy*xw)/det,overhead=(wy*xx-xy*xw)/det;if(scale>=0&&overhead>=0)candidates.push({scale,overhead});}
 const loss=f=>points.reduce((sum,p)=>sum+p.weight*(f.scale*p.x+f.overhead-p.y)**2,0);
 const best=candidates.sort((a,b)=>loss(a)-loss(b))[0];
 // The newer workbook supplies a matching 32K-budget P observation. P batch
 // here is nominal in-flight requests (including scheduling), never D active
 // batch. Fit only its cold row; keep the APC row as a disclosed cross-check.
 const budgetRef=pdData.a5Calibration.prefillBudgetReference;
 let budgetCorrection=null;
 if(phase==='prefill'&&budgetRef){
  const p=budgetRef.training,anchor={...c,input:p.input,output:1,prefixHitRate:0,chunk:budgetRef.chunk,
   util:budgetRef.util,reserve:4,moeMultistream:true,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,
   mtpDraft:5,decodeMode:'mtp',kvDtype:'fp8',constraintStatistic:'mean'};
  const stage={hw:'a5-96',nodes:1,dp:1,tp:1,pcp:8,ep:8,usableGiB:0};
  const family=pdA5Profile(anchor,stage,phase),raw=pdA5Raw(anchor,stage,p.batch,phase,pdA5Memory(anchor,stage,phase,family));
  const scale=p.seconds/raw.seconds,weight=Math.max(0,Math.min(1,(c.chunk-16384)/(budgetRef.chunk-16384)));
  budgetCorrection={reference:budgetRef.id,weight,scale,sourceBatch:p.batch,sourceTTFT:p.seconds,sourceRawTTFT:raw.seconds,
   evidence:'workbook_summary_nominal_inflight',blindValidation:false};
  best.scale=best.scale*(1-weight)+scale*weight;best.overhead*=1-weight;
 }
 return {...best,method:'nonnegative_relative_weighted_least_squares',trainingMultistream:true,trainingCount:points.length,training:points.map(({input,batch})=>({input,batch})),
  budgetCorrection,
  bounds:{minInput:Math.min(...points.map(p=>p.input)),maxInput:Math.max(...points.map(p=>p.input)),maxBatch:Math.max(...points.map(p=>p.batch))}};
}
function pdA5ModelRow(c,s,phase,f,batch,memory,fit){
 const decode=phase==='decode',reference=pdA5Reference(f,batch),measuredRatio=decode?reference.cycleSeconds/reference.tpotSeconds:null;
 const ratio=decode?Math.min(c.output-1,c.mtpDraft+1,c.ratioSource==='reference'?measuredRatio:c.itlTpotRatio):1;
 const scenario={...c,ratioSource:'custom',itlTpotRatio:ratio},row=pdA5Raw(scenario,s,batch,phase,memory);
 const floor=pdA5Raw({...scenario,pScale:1,dScale:1},s,batch,phase,memory,true);
 const raw=decode?row.itlSeconds:row.seconds,ideal=decode?floor.itlSeconds:floor.seconds,manual=decode?c.dScale:c.pScale;
 const corrected=fit.scale*raw+fit.overhead*manual,cycle=Math.max(ideal,corrected);
 const worstRaw=decode?row.worstITL:row.worst,worstFloor=decode?floor.worstITL:floor.worst;
 const worstCycle=Math.max(cycle,worstFloor,fit.scale*worstRaw+fit.overhead*manual);
 const seconds=cycle/ratio,worst=worstCycle/ratio,duration=seconds*(decode?c.output-1:1),qps=s.dp*batch/duration;
 const metric=decode?pdDecodeSLO(c,c.constraintStatistic==='mean'?cycle:worstCycle,c.constraintStatistic==='mean'?seconds:worst):{seconds:worst,feasible:worst<=c.ttft};
 const sameInput=pdData.a5Calibration.families.find(x=>x.phase===phase&&x.input===c.input),maxMeasuredBatch=sameInput?Math.max(...sameInput.training.map(x=>x.batch)):null;
 const targetNative=Boolean(pdA5Profile(c,s,phase)),hardwareTransfer=s.hw!=='a5-96';
 const extrapolated=!targetNative||!sameInput||batch>maxMeasuredBatch||(!decode&&c.prefixHitRate!==0)||c.chunk!==16384||
  (decode&&(c.output!==1024||c.dPrefixLayout!=='shared'))||c.computeScale!==1||c.memoryScale!==1||c.overlap!==0||c.imbalance!==1.1||
  c.pScale!==1||c.dScale!==1||c.constraintStatistic!=='mean'||c.util!==80||c.reserve!==4||Boolean(s.usableGiB)||c.moeMultistream!==true||c.dSharedPrefixRate!==100;
 return {...row,seconds,worst,duration,qps,tokensPerSecond:qps*(decode?c.output:c.input),
  ...(decode?{itlSeconds:cycle,worstITL:worstCycle,tpotSeconds:seconds,worstTPOT:worst,itlTpotRatio:ratio,
   projection:{...row.projection,kind:'calibrated_hardware_prediction',acceptanceMeasured:false,acceptanceBasis:c.ratioSource==='reference'?'邻近训练点的接受产出作为场景假设，未测 batch 不代表已验证接受率':'用户显式 r 假设',measuredAcceptedTokensPerCycle:measuredRatio,
    cycleMeasuredDirectly:false,memoryBasis:'FP8 物理页成本；按可用容量重建缓存预算，含共享前缀/私有状态/workspace，非原运行时池上限'}}:
   {computedTokensPerSecond:qps*row.plan.computedTokens}),
  breakdown:Object.fromEntries(pdGroups.map(k=>[k,row.breakdown[k]*cycle/raw])),
  sloSeconds:metric.seconds,feasible:row.memoryFeasible&&metric.feasible,
  calibration:{applied:true,runtime:false,empiricalOnly:false,targetNative,transfer:!targetNative,modelBased:true,extrapolated,
   ...(hardwareTransfer?{sourceHardware:'a5-96',targetHardware:s.hw,targetEmpiricallyValidated:false,
    transferKind:'prefill_service_efficiency',sourceKVDtype:'fp8',targetKVDtype:memory.config.kvDtype}:{}),
   factor:cycle/raw,fit,idealFloorApplied:ideal>corrected,family:f.id,profile:pdData.a5Calibration.profile,reference:pdData.a5Calibration.source,
   measuredCoverage:{input:hardwareTransfer?null:sameInput?.input??null,maxBatch:hardwareTransfer?null:maxMeasuredBatch??null},
   ...(hardwareTransfer?{sourceMeasuredCoverage:{hardware:'a5-96',input:sameInput?.input??null,maxBatch:maxMeasuredBatch??null}}:{}),
   reason:(hardwareTransfer?'850 纯 P 硬件算子模型 + 850E 实测效率迁移；保留850自身带宽、KV精度与容量，尚无850本机纯P实测验证；':'850E 硬件算子模型 + 少量实测校正；')+(fit.budgetCorrection?.weight>0?'P token预算校正含新版表128K冷缓存汇总测点（名义在途并发），APC未参与拟合；':'')+(!targetNative&&!hardwareTransfer?'EP8 实测效率迁移到目标拓扑，尚无目标拓扑实测验证；':'')+(hardwareTransfer&&c.prefixHitRate>0?'缓存命中场景沿用冷缓存效率校正，属于外推；':'')+(extrapolated?'当前为未测区域模型预测':'当前为训练覆盖内模型预测')+'。batch 由显存、TTFT/TPOT/ITL 和扫描上限决定，实测覆盖不设限。'},
  capacityBasis:memory.capacityBasis,plan:{...row.plan,...(!decode?{chunkBudgetMode:'per_dp',totalTokenBudget:c.chunk}:{}),hardwareModelFloorSeconds:ideal,rawModelSeconds:raw}};
}
function pdA5Stage(c,s,phase,f,errors){
 const decode=phase==='decode',fixed=decode?c.fixedD:c.fixedP,maxBatch=fixed||(decode?c.maxD:c.maxP);
 const limit=decode?(c.dConstraint==='itl'?c.itl:c.dConstraint==='both'?1:c.tpot):c.ttft;
 const base={config:s,phase,limit,maxBatch,valid:!errors.length,errors,results:[],feasible:[],best:null,fixedBatch:fixed,calibrationCap:null};
 if(errors.length)return {...base,failure:errors.join('；')};
 // Observed EP8 memory geometry is valid only for the matching topology.
 const memory=pdA5Profile(c,s,phase)?pdA5Memory(c,s,phase,f):pdMemory(c,s,phase),fit=pdA5Fit(c,s,phase),results=[];let firstMemoryRejected=null;
 for(let batch=fixed||1;batch<=maxBatch;batch++){
  const row=pdA5ModelRow(c,s,phase,f,batch,memory,fit);
  if(!row.memoryFeasible){firstMemoryRejected=row;break;}results.push(row);
 }
 const feasible=results.filter(r=>r.feasible).sort((a,b)=>b.qps-a.qps||a.worst-b.worst||a.batch-b.batch);
 return {...base,memory,results,feasible,best:feasible[0]||null,firstMemoryRejected,fit,
  minLatency:results.length?Math.min(...results.map(r=>r.sloSeconds)):null,scannedBatches:results.length+(firstMemoryRejected?1:0),
  failure:!results.length?'权重、KV/MTP 状态、workspace 与预留之和超出显存预算。':!feasible.length?'模型预测没有满足时延约束的候选。':''};
}
