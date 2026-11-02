/* Parameter shapes are independent of token/activation shapes. The embedded
 * catalog records checkpoint dimensions, not measured packed NPU allocations. */
const OperatorWeightDetails = (() => {
  'use strict';
  const catalog = JSON.parse(document.getElementById('operator-weight-data').textContent);
  const format = shape => shape ? '[' + shape.join(', ') + ']' : '—';
  const dtypeName = type => ({I8:'INT8', F32:'FP32', F8_E4M3:'FP8 E4M3', F8_E8M0:'E8M0 scale'}[type] || type);
  const glmSelections = {
    kda: {
      proj:['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.b_proj.weight','self_attn.forget_gate.f_a_proj.weight','self_attn.g_a_proj.weight'],
      conv:['self_attn.q_conv1d.weight','self_attn.k_conv1d.weight','self_attn.v_conv1d.weight'],
      fb:['self_attn.forget_gate.f_b_proj.weight'], gb:['self_attn.g_b_proj.weight'],
      core:['self_attn.forget_gate.A_log','self_attn.forget_gate.dt_bias'],
      norm:['self_attn.o_norm.weight'], oproj:['self_attn.o_proj.weight']
    },
    sfa: {
      proj:['self_attn.q_a_proj.weight','self_attn.kv_a_proj_with_mqa.weight'],
      qnorm:['self_attn.q_a_layernorm.weight'],
      q:['self_attn.q_b_proj.weight','self_attn.kv_b_proj.weight'],
      kv:['self_attn.kv_a_layernorm.weight'],
      iq:['self_attn.indexer.wq_b.weight','self_attn.indexer.weights_proj.weight'],
      pool:['self_attn.indexer.wk.weight','self_attn.indexer.index_kpool_compress_gate','self_attn.indexer.index_kpool_compress_ape','self_attn.indexer.k_norm.weight','self_attn.indexer.k_norm.bias'],
      vup:['self_attn.kv_b_proj.weight','self_attn.o_proj.weight']
    },
    moe: {
      hc:['ffn_hc.fn','ffn_hc.base','ffn_hc.scale','post_attention_layernorm.weight'],
      router:['mlp.gate.weight','mlp.gate.e_score_correction_bias'],
      expert:['mlp.experts.E.'], shared:['mlp.shared_experts.'], dense:['mlp.gate_proj.','mlp.up_proj.','mlp.down_proj.']
    }
  };
  const noWeight = {
    kda:{x:'该框从已归一化的 hidden 开始，仅表示输入 / AllGather；上游 Norm 的参数不属于本框。',qknorm:'L2Norm 不带可学习的 γ / β；该框无独立权重。',beta:'Cast / Sigmoid 无可学习参数；beta 是上游投影产生的激活。',initial:'按 slot 读取 recurrent state，无独立权重；state 是缓存。',meta:'请求与 slot metadata 不是模型权重。',cache:'这是历史 conv / recurrent 缓存，不是模型权重。',next:'Cache 写回无独立模型权重。'},
    sfa:{x:'输入边界 / token 分片无独立权重；归一化已在上游完成。',maincache:'KV cache 是运行状态，不是模型权重。',kgather:'AllGather 仅通信，不含可学习参数。',topk:'TopK 消费上游生成的 index Q、head weights 和 cache；这里无独立模型权重。',pack:'索引打包与 causal mask 无独立权重。',attn:'本框执行 QKᵀ / softmax / AV；投影权重分别列在 Main Q 和 V-up + OProj。',history:'历史 cache / metadata 无独立权重。',reuse:'复用前层索引，无独立模型权重。'},
    moe:{attention:'这是 KDA / SFA 输出边界；请点对应 Attention 子图查看投影权重。',residual:'这里是已生成的 residual / post / comb 激活；mHC 参数列在 mixing 节点。',dispatch:'Dispatch 只重排 / 传输 token；路由权重是动态激活。',combine:'Combine 消费动态 routing weights，没有独立模型参数。',add:'逐元素相加无独立模型权重。',out:'层输出 / deferred state 边界无新增权重。'}
  };
  const matches = (name, selectors) => selectors.some(key => key.endsWith('.') ? name.startsWith(key) : name === key);
  const derived = (name, shape, local, rule, dtype='BF16') => ({name, shape, local, dtype, rule, storage:'逻辑派生视图；不是新增 checkpoint 张量', derived:true});

  function glm(family, id, phase) {
    const selectors = glmSelections[family]?.[id] || [];
    const group = family === 'moe' && id === 'dense' ? 'dense' : family;
    const selected = selectors.flatMap(key => (catalog.glm[group] || []).filter(row => matches(row.name, [key])));
    const rows = selected.map(row => ({name:row.name, checkpointName:row.checkpointName, shape:row.shape,
      local:row.phases[phase].shape, dtype:dtypeName(row.dtype), rule:row.phases[phase].rule,
      storage:'Checkpoint ' + format(row.shape), axes:row.shape.length === 2 && row.name.endsWith('.weight') ? '线性权重通常按 [out_features, in_features] 阅读。' : ''}));
    const n = phase === 'prefill' ? 8 : 64;
    const notes = ['L 表示对应类型的某一层；E 表示单个专家。下列 shape 均不包含 token / batch 轴。',
      '完整 shape 与 checkpoint 精度来自 safetensors header；“初始分片”按图中 P: TP8 / EP16 / SP，D: TP1 / EP16。图中配置不随第四页仿真参数改变。',
      '转置、NZ 格式、量化打包、临时聚合不等于新权重；除显式派生视图外，不把这些 shape 声称为最终内核物理布局。'];
    if (family === 'kda' && id === 'proj') {
      rows.unshift(derived('in_proj_qkvbfg_a.weight · Q/K/V/b/f_a/g_a 合并', [24896,4096], [3*n*128+n+256,4096],
        '3×n×128 + n + 128 + 128；f_a / g_a 各 128 行在 TP 内复制，所以不能把 24896 整体除以 TP。'));
    }
    if (family === 'kda' && id === 'conv') {
      rows.unshift(derived('_merged_conv_weight · kernel view', [4,24576], [4,3*n*128],
        '分别移除 Q/K/V conv 的 singleton 轴，再按通道拼接并转置为 [kernel_size, 3×local_heads×head_dim]。'));
      notes.push('单个 conv checkpoint 为 [channels, 1, kernel_size]。源码参数先提升 FP32，实际合并 kernel view 转成 BF16。');
    }
    if (family === 'kda' && id === 'core') {
      notes.push('运行时 A_log broadcast view = [1, 1, '+n+', 1]，dt_bias = ['+(n*128)+']，均提升 FP32；checkpoint shape 分别为 [64] 和 [8192]。lower_bound = −5 是标量配置，不是可学习张量。');
    }
    if (family === 'sfa' && id === 'proj') {
      rows.unshift(derived('fused_qkv_a_proj.weight · 合并逻辑矩阵', [2048,4096], [2048,4096], 'q_a 1536 行 + kv_a 512 行；在 TP 内复制。某些后端随后转置。'));
    }
    if (family === 'sfa' && ['q','vup'].includes(id)) {
      const isQ = id === 'q';
      rows.unshift(derived(isQ ? 'W_UK_T · 从 kv_b 的 K 部分派生' : 'W_UV · 从 kv_b 的 V 部分派生',
        isQ ? [64,256,512] : [64,512,256], isQ ? [64,256,512] : [64,512,256],
        '图示 P 的 DSA-CP 使用全 64 heads；D 为 TP1。标准 TP 的初始每 rank 分片另列在 checkpoint 行中。'));
      notes.push('kv_b 的每 head 输出包含 K=256 与 V=256，两条支路共享同一 checkpoint 张量；表中不会把它们当成两套独立参数。');
      notes.push('SFA 的 DSA-CP 运行路径使用全 heads 的 Q / latent BMM；初始 TP 切片不是该路径最终矩阵。OProj 启用 full-weight gather 时为 [4096, 16384]，普通 TP8 初始切片为 [4096, 2048]。');
    }
    if (family === 'moe' && ['dense','shared','expert'].includes(id)) {
      const width = id === 'dense' ? 12288 : 2048;
      if (id !== 'expert') rows.unshift(derived('gate_up_proj.weight · Gate / Up 合并逻辑矩阵', [2*width,4096], [2*width,4096], '图示 Prefill 的 SP 使 Dense / Shared 权重复制；Decode 为 TP1。', 'INT8'));
      notes.push(id === 'expert' ? '每层 288 个 routed experts，EP16 每 rank 18 个完整专家；表中是单个专家的矩阵。Gate/Up 合并后每专家为 [4096, 4096]。' : 'SwiGLU 的 Gate / Up 是不同权重；scale / offset 是量化参数，并非激活 shape。');
    }
    if (family === 'moe' && id === 'hc') notes.push('mHC 的 fn/base/scale 在源码中提升 FP32；这里的 BF16 指 checkpoint 保存精度。');
    return {rows, notes, empty:noWeight[family]?.[id] || '此算子没有独立可学习权重。',
      context:'GLM-5.3-Flash · '+(phase==='prefill'?'Prefill · TP8 / EP16 / SP':'Decode · TP1 / EP16')+' · 完整权重与初始分片',
      localHeading:'图示 rank / 初始分片'};
  }

  const dsSelectors = {
    csa:{input:['layers.*.hc_attn_','layers.*.attn_norm.'],q:['layers.*.attn.wq_a.','layers.*.attn.q_norm.','layers.*.attn.wq_b.'],local:['layers.*.attn.wkv.','layers.*.attn.kv_norm.'],global:['layers.*.attn.compressor.'],idx:['layers.*.attn.indexer.'],sparse:['layers.*.attn.attn_sink'],out:['layers.*.attn.wo_a.','layers.*.attn.wo_b.']},
    moe:{input:['layers.*.hc_ffn_','layers.*.ffn_norm.'],router:['layers.*.ffn.gate.weight'],topk:['layers.*.ffn.gate.bias','layers.*.ffn.gate.bias_vl'],shared:['layers.*.ffn.shared_experts.'],experts:['layers.*.ffn.experts.']},
    engram:{table:['layers.*.engram.embed.'],project:['layers.*.engram.wkv.'],gate:['layers.*.engram.q_weight','layers.*.engram.k_weight']},
    draft:{project:['mtp.*.main_proj.','mtp.*.main_norm.'],noise:['embed.weight'],blocks:['mtp.*.attn.','mtp.*.attn_norm.','mtp.*.ffn.','mtp.*.ffn_norm.','mtp.*.hc_'],markov:['head.weight','mtp.*.norm.','mtp.*.markov_head.'],confidence:['mtp.*.confidence_head.'],verify:['layers.','embed.weight','head.weight','norm.weight']},
    overview:{p0:['embed.weight'],p1:['layers.'],p2:['layers.*.attn.compressor.'],p3:['layers.'],p4:['norm.weight','head.weight'],d1:['layers.'],d2:['norm.weight','head.weight'],d3:['mtp.','embed.weight','head.weight']}
  };
  const dsNoWeight = {
    csa:{swa:'QKᵀ / score 计算无独立投影权重；softmax 的 attn_sink 列在 CSA2 Sparse Attention。',post:'mHC Post 使用 Pre 生成的动态 mixing 系数，没有另一套 Post 参数。'},
    moe:{dispatch:'EP Dispatch 重排 / 通信，无独立权重。',combine:'消费动态路由权重进行加权 / 通信，无独立模型参数。',sum:'加法不带可学习参数。',post:'使用 mHC Pre 生成的动态系数，无独立 Post 权重。'},
    engram:{input:'Token IDs / history 是输入状态。',hash:'Hash 和词表映射不属于可学习的 dense 权重。',shard:'权重表的取回 / 通信不另建权重；表 shape 见 FP8 Lookup。',history:'History 更新不包含模型权重。',mix:'乘加消费动态 gate / value，无新增权重。',post:'这是 Transformer 输入边界；参数见对应 Attention / MoE 子图。'},
    draft:{input:'目标层特征是激活，不是另一组模型参数。',accept:'采样 / 接受逻辑无独立可学习参数。',post:'提交状态 / KV 无新增权重。'},
    overview:{d0:'这是已有 prompt / KV 状态，没有新增模型权重。',d4:'采样接受 / KV 提交无独立模型参数。'}
  };
  function deepseek(family, id, view='flow') {
    const data = JSON.parse(document.getElementById('ds41-data').textContent);
    const key = view === 'overview' ? 'overview' : family;
    const selectors = dsSelectors[key]?.[id] || [];
    const rows = data.weightGroups.filter(w => selectors.some(prefix => w.name.startsWith(prefix))).map(w => {
      const packed = ['experts','draftExperts'].includes(w.category) && w.name.endsWith('.weight') && w.dtype === 'I8';
      const shape = [...w.shape];
      if (packed) shape[shape.length-1] *= 2;
      let rule = packed ? 'FP4 两个元素装入一个字节；逻辑 in_features 是存储末轴的 2 倍。' :
        w.name.endsWith('.scale') ? '这是量化 scale 张量的原始 shape；不按普通权重矩阵扩展。' : 'Checkpoint 参数；不包含 B / T 激活轴。';
      if (w.category === 'engram') rule += ' L'+(w.shape[0]===data.config.text_config.engram_num_embeddings[0]?'1':'14')+' 的独立查表权重。';
      if (w.name === 'layers.*.attn.attn_sink') rule += '每个 query head 一个 sink。';
      return {name:w.name, shape, local:null, dtype:packed?'FP4 packed in INT8':dtypeName(w.dtype),
        storage:format(w.shape), rule, checkpointName:w.name, axes:packed?'逻辑 [out_features, in_features]；存储 [out_features, in_features / 2]。':''};
    });
    rows.sort((a,b) => Number(a.name.endsWith('.scale'))-Number(b.name.endsWith('.scale')));
    const notes = ['来自页面内嵌的官方 checkpoint header（revision '+data.revision.slice(0,12)+'）。* 表示层或专家通配符；每一行是一个参数类型，不是将各层权重合并。',
      '这里明确显示全局逻辑 shape 和 checkpoint 存储 shape；不把它们冒充某个 TP / EP rank 的实际物理布局。',
      'FP4 专家的 INT8 存储容器每字节装两个权重值；FP8 / E8M0 scale 的 shape 保持 header 原值。'];
    if(key==='csa' && id==='out') notes.push('wo_a 的 [8192,4096] 表示 8 groups × rank1024；单 group 的逻辑矩阵是 [1024,4096]，不是普通全连接 [8192,32768]。');
    if(key==='csa' && id==='global') notes.push('只有 global KV 源层拥有 compressor；压缩比 2 的源层有 wgate，比例 1 的 L20 不需要 wgate。复用缓存的层不新增同类权重。');
    if(key==='csa' && id==='idx') notes.push('Indexer query/head-weight 参数属于 index 源层；wk / k_norm 属于 key 源层。表中是对应类型的参数集合，不代表所有层均拥有这些张量。');
    if(key==='draft' && ['blocks','verify'].includes(id) || key==='overview' && ['p1','p3','d1','d3'].includes(id)) notes.push('这是组合阶段，列出所覆盖模块的参数类型；不同层的权重值独立。草稿专家为 128 选 3，主模型为 384 选 6。');
    return {rows, notes, empty:dsNoWeight[key]?.[id] || '此算子没有独立可学习权重。',
      context:'DeepSeek-V4.1-Flash · 全局 checkpoint · 逻辑矩阵 / 打包存储', localHeading:'分片口径'};
  }

  let dialog, opener;
  const textEl = (tag, text, cls) => {const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;};
  function open(result, title, button) {
    if (!dialog) {
      dialog = document.createElement('dialog'); dialog.className='op-weight-dialog'; dialog.id='operator-weight-dialog';
      dialog.setAttribute('aria-labelledby','operator-weight-dialog-title'); document.body.append(dialog);
      dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
      dialog.addEventListener('close',()=>{if(opener?.isConnected)opener.focus();});
    }
    opener=button;dialog.replaceChildren();
    const header=document.createElement('header'), heading=document.createElement('div');
    const h=textEl('h2',title+' · 权重 / 参数 Shape');h.id='operator-weight-dialog-title';heading.append(h,textEl('p',result.rows.length+' 个参数 / 派生视图 · 完整列表'));
    const close=textEl('button','关闭 ×');close.type='button';close.setAttribute('aria-label','关闭权重 Shape 详情');close.addEventListener('click',()=>dialog.close());header.append(heading,close);
    const content=textEl('div','','op-weight-content');content.append(textEl('p',result.context,'op-weight-context'));
    const wrap=textEl('div','','op-weight-table-wrap'),table=textEl('table','','op-weight-table');
    const thead=document.createElement('thead'),headRow=document.createElement('tr');
    ['参数 / 维度含义','完整逻辑 Shape',result.localHeading,'Checkpoint / 精度'].forEach(label=>{const th=textEl('th',label);th.scope='col';headRow.append(th);});thead.append(headRow);table.append(thead);
    const tbody=document.createElement('tbody');
    for(const row of result.rows){
      const tr=document.createElement('tr');tr.dataset.weightName=row.name;
      const name=document.createElement('td');name.append(textEl('code',row.name),textEl('small',row.rule));
      if(row.axes)name.append(textEl('small',row.axes));
      const full=document.createElement('td');full.append(textEl('code',format(row.shape)));
      const local=document.createElement('td');local.append(textEl('code',row.local?format(row.local):'全局参数'),textEl('small',row.local?(row.derived?'派生逻辑视图':'初始 TP / EP 分片'):'未声明某个 rank 的物理布局'));
      const storage=document.createElement('td');storage.append(textEl('code',row.dtype),textEl('small',row.storage));
      tr.append(name,full,local,storage);tbody.append(tr);
    }
    table.append(tbody);wrap.append(table);content.append(wrap);
    if(!result.rows.length)content.append(textEl('p',result.empty));
    const notes=textEl('ul','','op-weight-notes');result.notes.forEach(note=>notes.append(textEl('li',note)));content.append(notes);dialog.append(header,content);dialog.showModal();dialog.scrollTop=0;close.focus();
  }
  function render(root, result, title='算子') {
    let column=root.querySelector('.op-weight-column');
    if(!column){
      column=textEl('div','','op-weight-column');
      const heading=textEl('div','','op-weight-heading'),button=textEl('button','展开 ↗','op-weight-expand');button.type='button';button.setAttribute('aria-haspopup','dialog');button.setAttribute('aria-controls','operator-weight-dialog');
      heading.append(textEl('b','权重 / 参数 Shape'),button);
      const list=textEl('div','','op-weight-list');list.tabIndex=0;list.setAttribute('aria-label','权重与参数 shape，可滚动查看');column.append(heading,list);root.querySelector('.df-io').prepend(column);
    }
    const list=column.querySelector('.op-weight-list'),button=column.querySelector('button');list.replaceChildren();list.scrollTop=0;column.dataset.weightTitle=title;
    button.disabled=!result;button.setAttribute('aria-label','展开 '+title+' 的权重 Shape');button.onclick=()=>result&&open(result,title,button);
    if(!result){list.append(textEl('p','点击算子框，查看权重、精度与分片。','op-weight-empty'));return;}
    if(!result.rows.length){list.append(textEl('p',result.empty,'op-weight-empty'));return;}
    for(const row of result.rows){
      const card=textEl('div','','op-weight-card');card.dataset.weightName=row.name;
      card.append(textEl('code',row.name),textEl('strong',format(row.shape)));
      card.append(textEl('small',(row.local?(row.derived?'图示视图 ':'初始切片 ')+format(row.local)+' · ':'')+row.dtype));
      list.append(card);
    }
  }
  return {glm,deepseek,render,format,catalog};
})();
