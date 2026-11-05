// Atlas 850E runtime observations are bounded to the measured software profile.
// No source-hardware transfer or analytic memory admission enters this path.
function pdA5Family(c,s,phase){
 if(!pdData.a5Calibration||c.fidelity!=='calibrated'||s.hw!=='a5-96'||s.nodes!==1||s.tp!==1||s.ep!==8||s.usableGiB||
  c.constraintStatistic!=='mean'||c.util!==80||c.reserve!==4||c.chunk!==16384||c.decodeMode!=='mtp'||c.mtpDraft!==5||c.kvDtype!==undefined||
  c.pScale!==1||c.dScale!==1||c.computeScale!==1||c.memoryScale!==1||c.overlap!==0||c.imbalance!==1.1)return null;
 if(phase==='decode'&&(c.decodeMode!=='mtp'||c.output!==1024||c.dPrefixLayout!=='shared'))return null;
 return pdData.a5Calibration.families.find(f=>f.phase===phase&&f.input===c.input&&f.pcp===pdPCP(s)&&
  (phase!=='prefill'||f.prefixHitRate===c.prefixHitRate))||null;
}
function pdA5Reference(f,batch){
 const rows=f.training.slice().sort((a,b)=>a.batch-b.batch);
 if(batch<rows[0].batch||batch>rows.at(-1).batch)return null;
 const a=rows.filter(r=>r.batch<=batch).at(-1),b=rows.find(r=>r.batch>=batch);
 const at=k=>a.batch===b.batch?a[k]:a[k]+(b[k]-a[k])*(batch-a.batch)/(b.batch-a.batch);
 return f.phase==='decode'?{cycleSeconds:at('cycleSeconds'),tpotSeconds:at('tpotSeconds')}:{seconds:at('seconds')};
}
function pdA5Stage(c,s,phase,f,errors){
 const decode=phase==='decode',fixed=decode?c.fixedD:c.fixedP,requestedMax=decode?c.maxD:c.maxP;
 const minBatch=fixed||f.training[0].batch,cap=f.training.at(-1).batch,maxBatch=fixed||Math.min(requestedMax,cap);
 const limit=decode?(c.dConstraint==='itl'?c.itl:c.dConstraint==='both'?1:c.tpot):c.ttft;
 const m=f.runtimeMemory,capacityBasis='850E 同配置成功测点及调度器实际并发；搜索止于已验证 batch，非芯片理论上限';
 const memory={valid:true,errors:[],weight:m.loadedWeightBytes,reserve:c.reserve*GiB,limit:m.visibleBytes*c.util/100,
  block:m.blockSize,blocks:m.numBlocks,poolBytes:m.kvAllocatedBytes/m.numBlocks,kv:m.kvAllocatedBytes,kvBudget:m.kvAllocatedBytes,
  usableGiB:m.visibleBytes/GiB,config:{kvDtype:'fp8',pcp:pdPCP(s),dp:s.dp,tp:s.tp,ep:s.ep},capacityBasis,
  observedRuntime:{profile:pdData.a5Calibration.profile,allocatedPeakBytes:m.peakAllocatedBytes,kvAllocatedBytes:m.kvAllocatedBytes}};
 const base={config:s,phase,limit,maxBatch,valid:!errors.length,errors,memory,results:[],feasible:[],best:null,
  fixedBatch:fixed,calibrationCap:cap,scannedBatches:0,firstMemoryRejected:null};
 if(errors.length)return {...base,failure:errors.join('；')};
 if(fixed&&!pdA5Reference(f,fixed))return {...base,failure:`850E 此配置实测覆盖 batch ${f.training[0].batch}–${cap}；固定 batch ${fixed} 超出覆盖，未作外推。`};
 const results=[];
 for(let batch=minBatch;batch<=maxBatch;batch++){
  const ref=pdA5Reference(f,batch),measuredRatio=decode?ref.cycleSeconds/ref.tpotSeconds:null;
  const ratio=decode?(c.ratioSource==='reference'?measuredRatio:c.itlTpotRatio):null;
  const seconds=decode?ref.cycleSeconds/ratio:ref.seconds,duration=decode?(c.output-1)*seconds:seconds;
  const itl=decode?ref.cycleSeconds:null,qps=s.dp*batch/duration;
  const metric=decode?pdDecodeSLO(c,itl,seconds):{seconds,feasible:seconds<=c.ttft};
  const cached=f.cachedTokens??0;
  const reason='850E · '+pdData.a5Calibration.profile+' 同配置实测；'+(decode?(c.ratioSource==='reference'?'使用实测接受产出':'按用户 r 归一化，非该 r 的实测吞吐'):'P batch 表示每 DP 闭环在途并发，TTFT 包含排队；只对训练测点插值');
  results.push({batch,qps,seconds,worst:seconds,duration,sloSeconds:metric.seconds,feasible:metric.feasible,memoryFeasible:true,
   tokensPerSecond:qps*(decode?c.output:c.input),...(decode?{}:{computedTokensPerSecond:qps*(c.input-cached)}),
   peakBytes:m.peakAllocatedBytes,workspace:null,kvBytes:m.kvAllocatedBytes,perSequence:null,ids:null,
   breakdown:Object.fromEntries(pdGroups.map(k=>[k,0])),evals:0,capacityBasis,
   calibration:{applied:true,runtime:true,empiricalOnly:true,targetNative:true,factor:null,family:f.id,profile:pdData.a5Calibration.profile,
    reference:pdData.a5Calibration.source,reason,validation:f.validation},
   plan:decode?{decodeTokens:c.output-1,draftTokens:5,acceptedTokensPerCycle:ratio,expectedCycles:(c.output-1)/ratio,
    measuredAcceptedTokensPerCycle:measuredRatio,contextBasis:'measured full-concurrency decode window; output includes first P token'}:
    {pcp:8,inputTokens:c.input,computedTokens:c.input-cached,cachedTokens:cached,requestedHitRate:c.prefixHitRate,cacheBasis:'runtime_observed_full_blocks'},
   ...(decode?{itlSeconds:itl,worstITL:itl,tpotSeconds:seconds,worstTPOT:seconds,itlTpotRatio:ratio,
    projection:{kind:c.ratioSource==='reference'?'same_hardware_observation':'same_hardware_normalized_scenario',
     acceptanceMeasured:c.ratioSource==='reference',acceptanceBasis:c.ratioSource==='reference'?'同一稳态窗口的 accepted / draft 计数':'用户显式 r；实测周期保持不变',
     measuredAcceptedTokensPerCycle:measuredRatio,cycleMeasuredDirectly:false,
     cycleBasis:'steady TPOT × contemporaneous MTP counters; derived, not a hardware trace',
     memoryBasis:'FP8 KV；记录实际预分配缓存池与实测并发边界；非逐请求解析分配'}}:{})});
 }
 const feasible=results.filter(r=>r.feasible).sort((a,b)=>b.qps-a.qps||a.seconds-b.seconds||a.batch-b.batch);
 const minLatency=results.length?Math.min(...results.map(r=>r.sloSeconds)):null;
 return {...base,results,feasible,best:feasible[0]||null,minLatency,scannedBatches:results.length,
  failure:!results.length?'扫描上限低于此配置实测覆盖下限。':!feasible.length?'已测范围内没有满足时延约束的候选。':''};
}
