// Physical cost invariants and the reported 128K/1K regression. No images or NPU access.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;'))+'\nthis.hardware=pdData.devices;',ctx);
const base=hw=>({p:{hw,nodes:1,tp:1,pcp:8,ep:hw.startsWith('a3')?16:8,usableGiB:0},d:{hw,nodes:1,tp:hw.startsWith('a3')?2:1,ep:hw.startsWith('a3')?16:8,usableGiB:0},input:131072,output:1024,ttft:2,itl:.07,tpot:.02,dConstraint:'both',fidelity:'calibrated',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:0,fixedD:0,maxP:32,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const result=hw=>ctx.calculatePDSimulation(base(hw));
for(const [hw,batch,qps]of [['a3-560',11,4.336700622018795],['a5-96',24,9.53874786175455]]){
 const r=result(hw),s=r.d,b=s.best;
 assert.ok(r.valid&&b);assert.equal(b.batch,batch);close(b.qps,qps);
 close(b.qps,8*batch/((1024-1)*b.tpotSeconds));close(b.itlSeconds/b.tpotSeconds,3.5);
 assert.ok(b.itlSeconds<=.07&&b.tpotSeconds<=.02);
 const next=s.results.find(x=>x.batch===batch+1);
 assert.ok(next.memoryFeasible&&!next.feasible);
 assert.equal(ctx.pdStageBoundary(s).kind,'latency');
 assert.equal(b.calibration.transfer,true);assert.equal(b.projection.acceptanceMeasured,false);
 // At identical batch a longer context must not produce the old A5 128K speedup.
 for(const fixedD of [4,8,16,24,32]){
  let last=0;
  for(const input of [16384,32768,65536,131072]){
   const row=ctx.calculatePDSimulation({...base(hw),fixedD,input}).d.results[0];
   assert.ok(row.tpotSeconds>=last,`${hw} B${fixedD} got faster at ${input}`);last=row.tpotSeconds;
  }
 }
 const c={...r.config,fidelity:'analytic'},device=ctx.hardware.find(x=>x.id===hw);
 const forward=(q=6,prefix=131072)=>ctx.pdForward(c,c.d,8,q,prefix,'decode',true,'verify');
 const kernel=(f,name)=>f.kernels.find(x=>x.name===name);
 const a=forward(),score=kernel(a,'index-score'),saved=device.fp32;
 device.fp32*=.01;
 close(kernel(forward(),'index-score').compute,score.compute); // BF16 inputs, FP32 accumulator does not mean FP32 GEMM.
 device.fp32=saved;
 const bf16=device.bf16;device.bf16*=.5;
 assert.ok(kernel(forward(),'index-score').compute>score.compute);
 device.bf16=bf16;
 // Append work has bounded tail state, independent of full history length.
 const shorter=forward(6,16384);
 close(kernel(a,'index-pool').memory,kernel(shorter,'index-pool').memory);
 close(kernel(a,'index-pool').compute,kernel(shorter,'index-pool').compute);
 assert.ok(score.memory>kernel(shorter,'index-score').memory);
 // Matrix row occupancy includes replicated heads. At 32 heads, Q=1 occupies
 // a quarter tile; Q=4 occupies a full tile. Their GEMM cost is equal, apart
 // from causal boundary growth (use one pool for all query positions).
 const gp32=device.gp32;device.gp32=Infinity;
 close(kernel(forward(1,131072),'index-score').compute,kernel(forward(4,131072),'index-score').compute);
 device.gp32=gp32;
}
// Same-host observations remain exactly the measured anchors, never refitted
// to make the A3/A5 comparison look closer.
const family=data['pd-data'].runtimeCalibration.families.find(f=>f.phase==='decode');
for(const row of family.training){
 const c={...base('a3-800'),input:row.input,fixedD:row.batch,dConstraint:'tpot',ratioSource:'reference'};c.d.tp=4;
 const actual=ctx.calculatePDSimulation(c).d.results[0];
 close(actual.tpotSeconds,row.tpotSeconds);close(actual.itlSeconds,row.itlMeanSeconds);
 assert.equal(actual.calibration.empiricalOnly,true);
}
const capped=ctx.calculatePDSimulation({...base('a5-96'),itl:1,tpot:1}).d;
assert.equal(ctx.pdStageBoundary(capped).kind,'transfer_coverage');
const limited=ctx.calculatePDSimulation({...base('a5-96'),maxD:4}).d;
assert.equal(ctx.pdStageBoundary(limited).kind,'scan_limit');
console.log('PASS: 128K A3/A5 constraints, context scaling, BF16 matrix precision, query-head occupancy, bounded KeyPool updates, measured-anchor isolation and scan boundaries.');
