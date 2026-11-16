const fs=require('fs'),path=require('path'),assert=require('assert/strict'),vm=require('vm');
const root=path.resolve(__dirname,'..'),d=JSON.parse(fs.readFileSync(path.join(root,'data/glm53-replay-cache-basis.json'),'utf8'));
const c=d.calculation,P=c.pool_bytes_per_id,B=c.logical_block_tokens;
assert.equal(P,12*(5001216+340992));assert.equal(P/B,6768);
assert.equal(12*(528+(128+4)/4),6732);
assert.equal((528*B),5001216);
assert.equal(Math.ceil((B/4*132)/37888)*37888,340992);
assert.equal(8*24576*2+64*128*128*4+64*6*128*4+64*6*256*2,4980736);
assert.equal(c.slope_GiB_per_Mi_tokens,P/B*1048576/2**30);
for(const [hw,e]of Object.entries(d.evidence)){
 const v=d.profiles['replay'+hw];assert.equal(e.tensors.reduce((s,t)=>s+t.size,0),v.raw_tensor_total_bytes);assert.equal(v.raw_tensor_total_bytes,e.num_blocks*P);
 assert.equal(e.groups.length,5);assert.equal(e.groups[0].layers.length,24);assert.equal(e.groups[1].layers.length,12);
 assert.equal(e.groups.slice(2).reduce((s,g)=>s+g.layers.length,0),34);
 for(const g of e.groups.slice(2)){assert.equal(g.spec.num_speculative_blocks,0);assert.equal(g.spec.num_prefill_checkpoint_blocks,0);assert.equal(g.spec.mamba_cache_mode,'align');}
 assert.equal(e.flags.tp,1);assert.equal(e.flags.speculative_tokens,5);assert(e.flags.use_replayssm);
 for(const t of v.tensors){assert.equal(t.raw_bytes,e.num_blocks*(t.page_class==='main'?5001216:340992));for(const m of t.members){assert.equal(m.storage.bytes+m.storage.padding,m.storage.page_bytes);assert(e.groups[m.group].layers.includes(m.name));}}
}
const cases=[[[1048576],'one_Mi_request'],[Array(8).fill(131072),'eight_128Ki_requests'],[[1000000],'one_decimal_M_request']];
for(const [lengths,key]of cases){const bytes=P*(lengths.reduce((s,L)=>s+Math.ceil(L/B),0)+lengths.length*(3*2+1));assert.equal(bytes,c.cases[key].bytes);assert.equal(bytes/2**30,c.cases[key].GiB);}
assert.equal(c.cases.one_Mi_request.active_ids,118);assert.equal(c.cases.eight_128Ki_requests.active_ids,168);
// The independent global null block is a pool-wide constant, not per request.
assert.equal(P/2**30,0.059703826904296875);
let node;const context=vm.createContext({document:{createElement:()=>({}),currentScript:{before:n=>node=n}}});vm.runInContext(fs.readFileSync(path.join(root,'data/glm53-replay-cache-reference.js'),'utf8'),context);
const embedded=JSON.parse(node.textContent);assert.deepEqual(embedded.profiles,d.profiles);assert.deepEqual(embedded.calculation,d.calculation);
console.log('PASS: both captured full-model cache layouts, byte accounting, Replay SSM records, shared-pool alignment, 1M workload examples, published data consistency.');
