const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'glm53_prefill_decode_operator_block_diagram.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const data = {};
for (const [, attributes, body] of scripts) {
  if (attributes.includes('application/json')) data[attributes.match(/id="([^"]+)"/)[1]] = body;
  else new vm.Script(body); // Every embedded executable script must still parse.
}
const context = vm.createContext({document:{getElementById:id=>({textContent:data[id]})}});
vm.runInContext(fs.readFileSync(path.join(root, 'src/operator-weight-details.js'), 'utf8')+'\nthis.api=OperatorWeightDetails;', context);
const api = context.api;
const native = value => JSON.parse(JSON.stringify(value));
const shape = (result, name, field='shape') => native(result.rows.find(row=>row.name.includes(name))?.[field]);

// Dimension consistency: the merged KDA GEMM must produce all six branches.
const pre = api.glm('kda','proj','prefill'), dec = api.glm('kda','proj','decode');
assert.deepEqual(shape(pre,'in_proj_qkvbfg_a','local'), [3336,4096]);
assert.deepEqual(shape(dec,'in_proj_qkvbfg_a','local'), [24896,4096]);
assert.equal(pre.rows[0].local[0], pre.rows.slice(1).reduce((n,row)=>n+row.local[0],0));
assert.deepEqual(shape(pre,'f_a_proj','local'), [128,4096]);
assert.deepEqual(shape(api.glm('kda','conv','prefill'),'_merged_conv_weight','local'), [4,3072]);
assert.deepEqual(shape(api.glm('kda','core','prefill'),'A_log','local'), [8]);
assert.match(api.glm('kda','core','prefill').notes.join(' '), /\[1, 1, 8, 1\]/);
assert.deepEqual(shape(api.glm('kda','oproj','prefill'),'o_proj','local'), [4096,1024]);
assert.deepEqual(shape(api.glm('sfa','proj','prefill'),'fused_qkv_a_proj'), [2048,4096]);
assert.deepEqual(shape(api.glm('sfa','q','prefill'),'W_UK_T'), [64,256,512]);
assert.deepEqual(shape(api.glm('sfa','vup','prefill'),'W_UV'), [64,512,256]);
// SP duplicates dense/shared MLPs; EP partitions expert IDs, not each matrix.
assert.deepEqual(shape(api.glm('moe','dense','prefill'),'mlp.gate_proj.weight','local'), [12288,4096]);
assert.deepEqual(shape(api.glm('moe','expert','prefill'),'gate_proj.weight','local'), [2048,4096]);
assert.equal(api.glm('kda','qknorm','prefill').rows.length, 0);
assert.equal(api.glm('moe','dispatch','decode').rows.length, 0);

const ds = api.deepseek('moe','experts');
assert.deepEqual(shape(ds,'w1.weight'), [2304,5120]);
assert.equal(ds.rows.find(row=>row.name.endsWith('w1.weight')).storage, '[2304, 2560]');
assert.deepEqual(shape(ds,'w2.weight'), [5120,2304]);
assert.deepEqual(shape(ds,'w1.scale'), [2304,160]);
assert.deepEqual(shape(api.deepseek('csa','out'),'wo_a.weight'), [8192,4096]);
assert.deepEqual(shape(api.deepseek('draft','confidence'),'confidence_head.proj.weight'), [1,5376]);
assert.equal(api.deepseek('engram','table').rows.filter(row=>row.name.endsWith('.weight')).length, 2);
assert.equal(api.deepseek('moe','topk').rows.length, 2);

// Cover every selectable graph node; none may silently return an empty mapping.
const flow = JSON.parse(data['flow-data']);
let cases = 0;
for(const graph of flow.cases) for(const node of graph.nodes){
  const result=api.glm(graph.family,node.id,graph.phase);
  assert.ok(result.rows.length || result.empty !== '此算子没有独立可学习权重。', `${graph.family}/${node.id} lacks a parameter classification`);
  for(const row of result.rows){assert.ok(row.shape.every(x=>Number.isInteger(x)&&x>0));assert.ok(row.local.every(x=>Number.isInteger(x)&&x>0));}
  cases++;
}
const engine = scripts.find(([,a,b])=>b.includes('const DS41=')).at(2);
const definitions = engine.slice(engine.indexOf(' const flows='),engine.indexOf(' function overview()'));
const graphContext=vm.createContext({N:(id,...rest)=>({id})});
vm.runInContext(definitions+'\nthis.graphs=flows;',graphContext);
for(const [family,graph] of Object.entries(graphContext.graphs)) for(const node of graph.nodes){
  const result=api.deepseek(family,node.id);
  assert.ok(result.rows.length || result.empty !== '此算子没有独立可学习权重。', `DS ${family}/${node.id} lacks a parameter classification`);
  cases++;
}
for(const phase of ['p','d']) for(let i=0;i<5;i++){
  const result=api.deepseek('csa',phase+i,'overview');
  assert.ok(result.rows.length || result.empty !== '此算子没有独立可学习权重。');
  cases++;
}
console.log(`PASS: ${cases} node/context classifications; matrix dimensions, TP/SP/EP rules, FP4 storage, embedded JavaScript syntax.`);
