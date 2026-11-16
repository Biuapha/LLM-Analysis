const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const jsonScripts=Object.fromEntries([...html.matchAll(/<script\b[^>]*id="([^"]+)"[^>]*type="application\/json"[^>]*>([\s\S]*?)<\/script>/g)].map(m=>[m[1],m[2]]));
const ctx=vm.createContext({document:{getElementById:id=>({textContent:jsonScripts[id]})}});
const engine=html.slice(html.indexOf('const memoryData='),html.indexOf('function readMemoryInputs'));
vm.runInContext(engine+'\nthis.api={calculateRankMemory,curveMemoryData};',ctx);
const {calculateRankMemory:calc,curveMemoryData:data}=ctx.api;
const catalog=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(root,'data/glm53-curve-checkpoint-headers.json.gz'))));
assert.equal(require('node:crypto').createHash('sha256').update(fs.readFileSync(path.join(root,'data/glm53-curve-checkpoint-headers.json.gz'))).digest('hex'),data.source.headers_sha256);
assert.deepEqual(JSON.parse(JSON.stringify(data)),JSON.parse(fs.readFileSync(path.join(root,'data/glm53-curve-weight-memory.json'),'utf8')));
const base={dp:8,tp:1,ep:8,total:96,util:90,reserve:4,q:5,sp:'auto',vision:'off',casts:true,quant:'curve_mxfp8'};
const part=(r,id)=>r.components.find(x=>x.id===id).bytes;
const full=calc({...base,dp:1,ep:1,vision:'tp',casts:false,total:1024});
assert(full.valid);assert.equal(full.weight,catalog.payload_bytes);
assert.equal(full.weight,catalog.records.reduce((s,r)=>s+r.bytes*r.count,0));
assert.equal(catalog.records.reduce((s,r)=>s+r.count,0),76108);
assert.equal(part(full,'other'),0);
assert.equal(part(calc(base),'global'),2*154880*4096*2+4096*2);
assert.equal(calc(base).weight,calc({...base,q:1}).weight);
assert.equal(calc(base).weight,calc({...base,q:7}).weight);
assert.equal(calc({...base,q:0}).weight,calc(base).weight-part(calc(base),'mtp'));
assert.equal(part(calc(base),'routed'),42*36*3*(2048*4096+2048*4096/32));
assert.equal(part(calc({...base,dp:4,tp:2}),'routed'),part(calc(base),'routed'));
const mxfp8=(n,k)=>n*k+n*k/32;
for(const tp of [1,2,4,8,16,32,64]){
 const r=calc({...base,dp:1,tp,ep:1,total:1024,sp:'off'});assert(r.valid);
 // Down block scales must shard in K, unlike INT8 per-output-channel scales.
 assert.equal(part(r,'routed'),42*288*3*mxfp8(2048,4096)/tp);
 assert.equal(part(r,'dense'),3*3*mxfp8(12288,4096)/tp);
 assert.equal(part(r,'shared'),42*3*mxfp8(2048,4096)/tp);
 assert.equal(part(r,'sfa'),11*(mxfp8(1536,4096)+mxfp8(512,4096)+4096+(mxfp8(16384,1536)+mxfp8(4096,16384)+32768*512*2)/tp));
}
// Independent row-by-row reference from raw headers; infer tensor slices,
// avoiding the generated per-module coefficients used by the page.
function rawReference(c,local){
 let total=0;
 for(const t of catalog.records){
  const n=t.name,l=Number(n.match(/\.layers\.(\d+)\./)?.[1]??-1);
  if(l===45&&!c.q||n.includes('.visual.')&&c.vision==='off')continue;
  let count=t.count,div=1;
  if(n.includes('.experts.')){count=local;div=c.ep>1?1:c.dp*c.tp;}
  else if(n.includes('.visual.')){
   if(c.vision!=='data'&&(/\.blocks\..*\.(qkv|gate_proj|up_proj)\./.test(n)||/\.blocks\..*\.(proj|down_proj)\.weight$/.test(n)||/\.merger\.(proj|gate_proj|up_proj)\./.test(n)||/\.merger\.down_proj\.weight$/.test(n)))div=c.tp;
  }else if(n.includes('embed_tokens.weight')||n==='lm_head.weight')div=c.tp;
  else if(n.includes('.self_attn.')&&!n.includes('.indexer.')){
   if(l<45&&catalog.model_config.text_config.layer_types[l]==='linear_attention'){
    if(!/\.(f_a_proj|g_a_proj|o_norm)\./.test(n))div=c.tp;
   }else if(/\.(q_b_proj|kv_b_proj|o_proj|q_proj)\./.test(n))div=c.tp;
  }else if(n.includes('.mlp.')&&!n.includes('.gate.')&&!(c.sp==='on'||c.sp==='auto'&&c.tp>1))div=c.tp;
  let bytes=t.bytes;
  if(c.casts){
   if(/_conv1d\.weight|\.A_log|\.dt_bias|\.hc_(attn|ffn)_|\.index_kpool_compress_ape/.test(n))bytes=Math.max(bytes,t.numel*4);
   if(/\.indexer\.(weights_proj\.weight|k_norm\.)/.test(n))bytes+=t.numel*4;
  }
  total+=count*bytes/div;
 }
 return total;
}
let cases=0;
for(const tp of [1,2,4,8,16,32,64])for(const dp of [1,2,3,4,8])for(const ep of new Set([1,dp*tp]))for(const q of [0,5])for(const casts of [false,true])for(const sp of ['auto','off']){
 const c={...base,tp,dp,ep,q,casts,sp,total:1024};const r=calc(c);if(!r.valid)continue;
 assert.equal(r.weight,rawReference(c,r.localMax));assert.equal(r.weightMin,rawReference(c,r.localMin));
 assert.equal(r.weight+r.reserve+r.kv+r.alignment+r.margin,r.total);
 cases++;
}
for(const vision of ['tp','data']){const c={...base,dp:2,tp:8,ep:16,vision};const r=calc(c);assert.equal(r.weight,rawReference(c,r.localMax));}
assert(!calc({...base,dp:128,ep:1}).valid);
const results=Object.fromEntries([8,16,32].map(ep=>{const r=calc({...base,dp:ep,ep});return [ep,{bytes:r.weight,GiB:r.weight/2**30,MTP_GiB:part(r,'mtp')/2**30}];}));
console.log(JSON.stringify({passed:true,raw_header_rank_cases:cases,full_checkpoint_reconciles:true,results}));
