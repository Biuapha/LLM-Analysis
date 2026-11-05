// Cross-device estimates keep observations, workload assumptions, and hardware
// models separate. Source execution settings are frozen, not copied from the
// candidate: changing a target budget, chunk, EP or stream must not erase a fit.
function pdCrossContext(c,stage,phase){
 if(!pdData.crossHardwareEstimation?.enabled||c.fidelity!=='calibrated'||!['a3-560','a3-752'].includes(stage.hw))return null;
 if(c.kvDtype&&c.kvDtype!=='int8')return null;
 if(pdRuntimeFamily(c,stage,phase))return null;
 const families=(pdData.runtimeCalibration?.families||[]).filter(f=>f.hardware==='a3-800'&&f.phase===phase&&f.predictionModel==='absolute_observation');
 if(!families.length)return null;
 const side=phase==='prefill'?'p':'d',family=families[0];
 const source=pdCrossSource(c,family,c.input),bounds=pdRuntimeBounds(source,source[side],phase);
 return {source,side,family,families,bounds,preserveTargetCache:true};
}
function pdCrossSource(c,f,input){
 const lengths=f.training.map(r=>r.input),sourceInput=Math.max(Math.min(...lengths),Math.min(Math.max(...lengths),input));
 return {...c,fidelity:'calibrated',input:sourceInput,output:f.phase==='decode'?f.output:1,
  p:{hw:'a3-800',nodes:1,tp:1,pcp:8,ep:16,dp:2,usableGiB:0},
  d:{hw:'a3-800',nodes:1,tp:4,pcp:1,ep:16,dp:4,usableGiB:0},
  mtpDraft:f.mtpDraft,decodeMode:'mtp',util:f.util??80,reserve:4,chunk:f.chunk??16384,
  prefixHitRate:f.prefixHitRate??0,dPrefixLayout:'shared',dSharedPrefixRate:100,
  pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,
  ratioSource:'custom',itlTpotRatio:1,constraintStatistic:'mean',kvDtype:'int8',
  // The old source did not log stream state. Keep one explicit ON baseline
  // for fitting and report this transfer assumption; never refit on a toggle.
  moeMultistream:true,chunkBudgetMode:'per_dp',fixedP:0,fixedD:0};
}
function pdCrossAnchor(c,context,f,batch,phase){
 const source=pdCrossSource(c,f,c.input),side=context.side,bounds=pdRuntimeBounds(source,source[side],phase);
 if(!bounds)return null;
 const anchorBatch=Math.max(bounds.minBatch,Math.min(bounds.maxBatch,batch));
 const ref=pdRuntimeReference(source,source[side],phase,anchorBatch);if(!ref)return null;
 const hit=f.cacheObservation?100*f.cacheObservation.cachedTokens/f.cacheObservation.input:source.prefixHitRate;
 const proxy={...source,fidelity:'analytic',prefixHitRate:hit,
  p:{...source.p,hw:'a3-800-model-proxy'},d:{...source.d,hw:'a3-800-model-proxy'}};
 const proxyStage=proxy[side],proxyMemory=pdMemory(proxy,proxyStage);
 const row=pdBatch(proxy,proxyStage,anchorBatch,phase,proxyMemory),decode=phase==='decode';
 const observed=decode?ref.itlMeanSeconds:ref.measuredSeconds,raw=decode?row.itlSeconds:row.seconds;
 if(!(observed>0&&raw>0))return null;
 return {factor:observed/raw,observed,raw,input:source.input,anchorBatch,sourceFamily:f.id,
  sourceTopology:source[side],hit,ratio:decode?ref.itlMeanSeconds/ref.tpotSeconds:null,
  extrapolated:batch!==anchorBatch||c.input!==source.input};
}
function pdCrossBatch(c,stage,batch,phase,memory){
 const context=pdCrossContext(c,stage,phase);if(!context)return null;
 const decode=phase==='decode',anchors=context.families.map(f=>pdCrossAnchor(c,context,f,batch,phase)).filter(Boolean).sort((a,b)=>a.hit-b.hit);
 if(!anchors.length)return null;
 // Interpolate the efficiency correction across observed cache regimes. The
 // target retains its requested cached-token count and modeled context cost.
 const hit=decode?0:c.prefixHitRate,lo=anchors.filter(a=>a.hit<=hit).at(-1)||anchors[0],hi=anchors.find(a=>a.hit>=hit)||anchors.at(-1);
 const w=hi.hit===lo.hit?0:(hit-lo.hit)/(hi.hit-lo.hit),factorRaw=lo.factor+(hi.factor-lo.factor)*w;
 const ratio=decode?(c.ratioSource==='reference'?Math.min(c.mtpDraft+1,c.output-1,lo.ratio):c.itlTpotRatio):1;
 const analytic={...c,fidelity:'analytic',kvDtype:'int8',ratioSource:'custom',itlTpotRatio:ratio};
 const row=pdBatch(analytic,stage,batch,phase,memory);
 const floor=pdBatch({...analytic,fidelity:'ideal',pScale:1,dScale:1,overlap:0,imbalance:1},stage,batch,phase,memory);
 const raw=decode?row.itlSeconds:row.seconds,ideal=decode?floor.itlSeconds:floor.seconds;
 const factor=Math.max(factorRaw,ideal/raw),seconds=row.seconds*factor,duration=row.duration*factor;
 const worstITL=row.worstITL*factor,worstTPOT=row.worstTPOT*factor;
 const metric=decode?pdDecodeSLO(c,c.constraintStatistic==='mean'?row.itlSeconds*factor:worstITL,c.constraintStatistic==='mean'?seconds:worstTPOT):
  {seconds:row.worst*factor,feasible:row.worst*factor<=c.ttft};
 const extrapolated=lo.extrapolated||hi.extrapolated||hit!==lo.hit||c.mtpDraft!==5||stage.tp!==lo.sourceTopology.tp||pdPCP(stage)!==lo.sourceTopology.pcp||stage.ep!==lo.sourceTopology.ep;
 return {...row,seconds,worst:row.worst*factor,duration,qps:stage.dp*batch/duration,tokensPerSecond:stage.dp*batch/duration*(decode?c.output:c.input),
  ...(decode?{itlSeconds:row.itlSeconds*factor,worstITL,tpotSeconds:seconds,worstTPOT,itlTpotRatio:ratio,
   projection:{...row.projection,kind:'cross_hardware_estimate',acceptanceMeasured:false,acceptanceTransferred:c.ratioSource==='reference',
    acceptanceBasis:c.ratioSource==='reference'?'源测点接受产出迁移假设；非目标实测':'用户显式接受产出假设',memoryBasis:'目标硬件容量与缓存模型；未复制源运行时池上限'}}:
   {computedTokensPerSecond:stage.dp*batch/duration*row.plan.computedTokens}),
  breakdown:Object.fromEntries(pdGroups.map(k=>[k,row.breakdown[k]*factor])),sloSeconds:metric.seconds,feasible:row.memoryFeasible&&metric.feasible,
  calibration:{applied:false,runtime:false,transfer:true,modelBased:true,extrapolated,anchorBatch:lo.anchorBatch,factor,idealFloorApplied:ideal>factorRaw*raw,
   sourceFamily:lo.sourceFamily,sourceHardware:'a3-800',targetHardware:stage.hw,sourceStreamState:'unverified_assumed_on_for_fixed_fit',
   reason:'800T测点校正的A3硬件模型；源配置固定，多流/EP/显存改变不取消校正。560T本机尚无稳态batch校准，源多流状态按开启假设；未测batch继续由模型搜索。'},
  transfer:{method:'fixed_source_observation_over_current_source_model_times_target_model',observedSourceSeconds:lo.observed,sourceProxySeconds:lo.raw,
   targetRawSeconds:raw,anchorBatch:lo.anchorBatch,requestedBatch:batch,extrapolated,sourceFamily:lo.sourceFamily,sourceHardware:'a3-800',targetHardware:stage.hw,
   sourceProxy:pdData.crossHardwareEstimation.sourceProxy,sourceTopology:lo.sourceTopology,targetTopology:stage,acceptanceRatio:ratio,
   targetEmpiricallyValidated:false,sourceStreamState:'unverified_assumed_on_for_fixed_fit',cacheCorrectionAnchors:[lo.hit,hi.hit]},
  capacityBasis:'目标硬件的解析容量；独立于源实测batch和分配器上限'};
}
