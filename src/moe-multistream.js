// Shared-expert scheduling, pinned to vLLM-Ascend r71 e2e591b166b7.
// This is a critical-path estimate, not a measured overlap percentage.
function pdMoEMultistream(c,stage,kernels,moeLayers,epDispatchSeconds){
 const enabled=c.moeMultistream===true;
 if(!enabled)return {enabled:false,savedSeconds:0,reason:'共享专家串行执行'};
 const duration=k=>Math.max(k.compute,k.memory)+k.launch;
 const families=pdData.linearFamilies,shared=[],routedUp=[],routedDown=[],router=[];
 const find=name=>kernels.find(k=>k.name===name);
 for(const k of kernels){
  const match=/^(linear|quant)-(\d+)$/.exec(k.name);if(!match)continue;
  const f=families[Number(match[2])];
  if(f.examples.some(n=>n.includes('.shared_experts.')))shared.push(k);
  else if(f.mode==='expert')(f.axis===0?routedUp:routedDown).push(k);
  else if(f.examples.some(n=>n.includes('.mlp.gate.weight')))router.push(k);
 }
 const activation=find('shared-silu');if(activation)shared.push(activation);
 const sum=items=>items.reduce((n,k)=>n+duration(k),0)/moeLayers;
 const select=(axis,op)=>shared.filter(k=>{const m=/^(linear|quant)-(\d+)$/.exec(k.name);return m&&m[1]===op&&families[Number(m[2])].axis===axis;});
 const parts={router:sum(router),topk:sum([find('routing')].filter(Boolean)),
  sharedPrepare:sum(select(0,'quant')),sharedUp:sum(select(0,'linear')),
  sharedActivation:sum([activation,...select(1,'quant')].filter(Boolean)),sharedDown:sum(select(1,'linear')),
  routedUp:sum(routedUp),routedActivation:sum([find('expert-silu')].filter(Boolean)),routedDown:sum(routedDown),
  dispatch:epDispatchSeconds/moeLayers,combine:epDispatchSeconds/moeLayers};
 const dispatchEnd=parts.router+parts.topk+parts.dispatch;
 const gmm2Start=dispatchEnd+parts.routedUp+parts.routedActivation;
 const combineStart=gmm2Start+parts.routedDown,routedEnd=combineStart+parts.combine;
 const sharedUpEnd=Math.max(parts.router,parts.sharedPrepare)+parts.sharedUp;
 // AllGather exposes GMM2; non-fused MC2/All2All expose the end of dispatch.
 const activationRelease=stage.ep>1?dispatchEnd:gmm2Start;
 const sharedActivationEnd=Math.max(sharedUpEnd,activationRelease)+parts.sharedActivation;
 const sharedEnd=Math.max(sharedActivationEnd,combineStart)+parts.sharedDown;
 const serial=Object.values(parts).reduce((a,b)=>a+b,0),sharedTotal=sum(shared);
 const resources=[...shared,...routedUp,...routedDown,...router,find('routing'),find('expert-silu')].filter(Boolean);
 // Streams share Cube/Vector and HBM. Do not grant each stream a second copy
 // of the hardware bandwidth/compute budget. Collective network time is not
 // counted as HBM traffic here. Detailed contention and event overhead remain
 // unmeasured, so the schedule is still an optimistic estimate.
 const computeFloor=resources.reduce((n,k)=>n+k.compute,0)/moeLayers;
 const hbmFloor=resources.reduce((n,k)=>n+k.memory,0)/moeLayers;
 const critical=Math.max(routedEnd,sharedEnd,computeFloor,hbmFloor);
 const saving=Math.max(0,Math.min(sharedTotal,serial-critical)),keep=sharedTotal?1-saving/sharedTotal:1;
 // Attribute hidden work to shared-expert operator families. Their sizes and
 // schedule are independent of prefix length, preserving the affine context
 // summation used for long decode and chunked prefill. No negative cost bars.
 for(const k of shared){k.serialSeconds=duration(k);k.compute*=keep;k.memory*=keep;k.launch*=keep;}
 return {enabled:true,backend:stage.ep>1?'non_fused_mc2_or_alltoall':'allgather',
  savedSeconds:saving*moeLayers,sharedSeconds:sharedTotal*moeLayers,
  serialMoESeconds:serial*moeLayers,criticalMoESeconds:(serial-saving)*moeLayers,
  computeFloorSeconds:computeFloor*moeLayers,hbmFloorSeconds:hbmFloor*moeLayers,
  partsPerLayer:parts,resourceFloorApplied:critical>Math.max(routedEnd,sharedEnd),
  scope:'共享专家与路由分支按事件依赖重叠；TP/PCP 外围集合通信保留；非 DBO / 融合 MC2',
  limitation:'共享算力/HBM总预算下界；额外资源争用和同步开销未实测，属于乐观调度估计'};
}
// End shared-expert scheduling.
