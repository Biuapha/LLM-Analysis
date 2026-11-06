/* Fixed, reproducible v13 result; independent of the editable page-four form. */
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('comparison-data').textContent);
  const el = id => document.getElementById('comparison-' + id);
  const atlas850 = data.hardwareProfiles.find(h => h.id === 'a5-128');
  const hardwareNote = `850：${atlas850.memoryGiB} GiB / ${atlas850.memoryBandwidth/1e12} TB/s（${atlas850.memoryBandwidth/1e9} GB/s）/ INT8 KV。850E：96 GiB / 4 TiB/s / FP8 KV，可见容量按 91.75 GiB。`;
  const colors = {8:'#69adff',16:'#ffad71',32:'#61d8ba'};
  const positions = [[46,43],[33,46],[20,40],[31,13],[69,23],[59,38]];
  const visible = new Set(data.series.map((_, i) => i));
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const name = s => `${s.hardware} · DP${s.ranks} EP${s.ranks}`;
  const x = b => 68 + b / 90 * 934;
  const yMax = Math.max(65, Math.ceil(Math.max(...data.series.flatMap(s => s.rows.map(r => r.tpotMs || 0))) / 5) * 5);
  const y = ms => 488 - ms / yMax * 452;
  const text = (px, py, label, attrs = '') => `<text x="${px}" y="${py}" fill="#9fb5c4" ${attrs.includes("font-size=") ? "" : 'font-size="13"'} ${attrs}>${esc(label)}</text>`;
  const path = rows => rows.map((r, i) => `${i ? 'L' : 'M'}${x(r.batchPerDP).toFixed(2)},${y(r.tpotMs).toFixed(2)}`).join(' ');
  const marker = (px, py, color, diamond, size = 5) => diamond
    ? `<path d="M${px},${py-size}l${size},${size}l-${size},${size}l-${size},-${size}Z" fill="#0b1b27" stroke="${color}" stroke-width="2"/>`
    : `<circle cx="${px}" cy="${py}" r="${size}" fill="#0b1b27" stroke="${color}" stroke-width="2"/>`;
  function renderChart() {
    let svg = '<title>GLM-5.3-Flash：850 / 850E 六条纯 Decode 曲线</title><desc>横轴 batch/DP 0 到 90，纵轴平均 TPOT 毫秒。标签是满足 30ms 与显存约束的最佳单卡 QPS；淡线为显存不可行预测。</desc><rect width="1040" height="570" fill="#0b1b27"/>';
    for (let t = 0; t <= yMax; t += 10) svg += `<line x1="68" y1="${y(t)}" x2="1002" y2="${y(t)}" stroke="#263f50"/>` + text(55,y(t)+4,t,'text-anchor="end"');
    for (let b = 0; b <= 90; b += 10) svg += text(x(b),513,b,'text-anchor="middle"');
    svg += `<path d="M68 36V488H1002" fill="none" stroke="#527087"/>`;
    svg += text(535,550,'Batch / DP（每个 DP 的请求数）','text-anchor="middle" font-size="15"');
    svg += text(18,260,'平均 TPOT（ms）','text-anchor="middle" transform="rotate(-90 18 260)"');
    svg += `<line x1="68" y1="${y(30)}" x2="1002" y2="${y(30)}" stroke="#e3cc91" stroke-dasharray="7 5"/>` + text(996,y(30)-10,'30 ms · ITL 105 ms','text-anchor="end"');
    data.series.forEach((s, i) => {
      if (!visible.has(i)) return;
      const color = colors[s.ranks], dash = s.hardware === '850E' ? 'stroke-dasharray="7 5"' : '';
      const rows = s.rows.filter(r => r.batchPerDP > 0);
      const feasible = rows.filter(r => r.memoryFeasible), rejected = rows.filter(r => !r.memoryFeasible);
      svg += `<g data-series="${i}"><title>${esc(name(s))}</title><path data-feasible="true" d="${path(feasible)}" fill="none" stroke="${color}" stroke-width="2.7" ${dash}/>`;
      if (rejected.length) {
        svg += `<path data-feasible="false" d="${path([...feasible.slice(-1),...rejected])}" fill="none" stroke="${color}" stroke-width="2" opacity="0.24" ${dash}/>`;
        rejected.filter((_, j) => j % 5 === 0).forEach(r => {
          const px=x(r.batchPerDP),py=y(r.tpotMs);
          svg += `<path d="M${px-3},${py-3}l6,6m0,-6l-6,6" fill="none" stroke="${color}" opacity="0.5"/>`;
        });
      }
      svg += '</g>';
    });
    // Keep labels in their own layer, with fixed, non-overlapping positions.
    data.series.forEach((s, i) => {
      if (!visible.has(i)) return;
      const p=s.summary,color=colors[s.ranks],diamond=s.hardware==='850E';
      const px=x(p.batchPerDP),py=y(p.tpotMs),lx=x(positions[i][0]),ly=y(positions[i][1]);
      svg += `<g data-best="${i}"><title>${esc(name(s))}: B${p.batchPerDP}, ${p.tpotMs.toFixed(3)} ms, ${p.qpsPerCard.toFixed(6)} QPS/卡</title><path d="M${px},${py}L${lx},${ly}" stroke="${color}" stroke-width="1"/>`;
      svg += marker(px,py,color,diamond,6);
      svg += `<rect data-qps-label="${i}" x="${lx-39}" y="${ly-16}" width="78" height="32" rx="5" fill="#102230" stroke="${color}" ${diamond?'stroke-dasharray="4 2"':''}/>`;
      svg += text(lx,ly+5,p.qpsPerCard.toFixed(3)+(p.limit==='memory'?' *':''),`text-anchor="middle" font-weight="700" style="fill:${color}"`)+'</g>';
    });
    const batch=Number(el('batch').value);
    svg += `<line id="comparison-cursor" x1="${x(batch)}" y1="36" x2="${x(batch)}" y2="488" stroke="#c3d8e5" opacity="0.35" stroke-dasharray="2 5" pointer-events="none"/>`;
    el('chart').innerHTML=svg;
  }
  function inspect() {
    const batch=Number(el('batch').value);
    el('batch-value').value=batch;
    el('batch').setAttribute('aria-valuetext',`每 DP ${batch} 个请求`);
    const cursor=el('cursor');
    if(cursor){cursor.setAttribute('x1',x(batch));cursor.setAttribute('x2',x(batch));}
    el('inspection').innerHTML='<div class="comparison-inspection-grid">'+data.series.map(s => {
      const r=s.rows.find(r => r.batchPerDP===batch);
      const status=batch===0?'无请求':!r.memoryFeasible?'显存超预算，不可用':!r.meets30msLimit?'超过 30 ms 约束':'满足显存与时延约束';
      return `<div class="comparison-inspection-cell" style="--series:${colors[s.ranks]}"><b>${esc(name(s))}</b><strong>${batch?r.tpotMs.toFixed(3)+' ms':'—'}</strong><span>${batch&&r.memoryFeasible?r.qpsPerCard.toFixed(3)+' QPS/卡':'可用 QPS：—'}</span><small class="${batch&&(!r.memoryFeasible||!r.meets30msLimit)?'comparison-rejected':''}">${status}</small></div>`;
    }).join('')+'</div>';
  }
  el('legend').innerHTML=data.series.map((s,i)=>`<button type="button" data-series-toggle="${i}" aria-pressed="true"><span aria-hidden="true" class="comparison-swatch ${s.hardware==='850E'?'dashed':''}" style="--series:${colors[s.ranks]}"></span>${esc(name(s))}</button>`).join('');
  el('summary').innerHTML=data.series.map(s=>`<div class="comparison-summary-row" style="--series:${colors[s.ranks]}"><b>${esc(name(s))}</b><strong>${s.summary.qpsPerCard.toFixed(3)}</strong><small>B${s.summary.batchPerDP} · ${s.summary.tpotMs.toFixed(3)} ms${s.summary.limit==='memory'?' *':''}</small><small>QPS/卡</small></div>`).join('');
  const memoryPoint=data.series.find(s=>s.hardware==='850E'&&s.ranks===8).summary;
  el('memory-note').textContent=`* 850E EP8：B${memoryPoint.batchPerDP} 已到模型显存边界，TPOT ${memoryPoint.tpotMs.toFixed(3)} ms；B${memoryPoint.firstMemoryRejected} 起超预算，不能继续加 batch 到 30 ms。`;
  el('source').textContent=`来源：${data.engine} · 曲线基线 ${data.commit.slice(0,7)} · 540 个非零 batch 测算点`;
  el('hardware-note').textContent=hardwareNote+'每台 8 卡，EP8 / 16 / 32 分别为 1 / 2 / 4 台。';
  const atlasPoint=data.series.find(s=>s.hardware==='850'&&s.ranks===8).summary;
  el('capacity-note').textContent=`850 的显存预算为 ${atlas850.memoryGiB} × 80% = ${atlasPoint.budgetGiB} GiB，包含 4 GiB 运行时预留；EP8 从 B${atlasPoint.firstMemoryRejected} 起超预算。30 ms 最优 B${atlasPoint.batchPerDP} 仍满足显存约束。`;
  el('legend').addEventListener('click',event=>{
    const button=event.target.closest('[data-series-toggle]');if(!button)return;
    const i=Number(button.dataset.seriesToggle);visible.has(i)?visible.delete(i):visible.add(i);
    button.setAttribute('aria-pressed',String(visible.has(i)));renderChart();
  });
  el('reset').addEventListener('click',()=>{
    data.series.forEach((_,i)=>visible.add(i));
    el('legend').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed','true'));renderChart();
  });
  el('batch').addEventListener('input',inspect);
  el('chart').addEventListener('click',event=>{
    const pt=el('chart').createSVGPoint();pt.x=event.clientX;pt.y=event.clientY;
    const local=pt.matrixTransform(el('chart').getScreenCTM().inverse());
    if(local.x<68||local.x>1002||local.y<36||local.y>488)return;
    el('batch').value=Math.max(0,Math.min(90,Math.round((local.x-68)/934*90)));inspect();
  });
  function download(content,mime,extension){
    const url=URL.createObjectURL(new Blob([content],{type:mime}));
    const a=document.createElement('a');a.href=url;a.download=`GLM53-850-850E-six-curves-v13-850-${atlas850.memoryGiB}GiB-${atlas850.memoryBandwidth/1e12}TBps.${extension}`;
    document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  el('json').addEventListener('click',()=>download(JSON.stringify(data,null,2),'application/json','json'));
  el('csv').addEventListener('click',()=>{
    const columns=['hardware','id','batchPerDP','totalInFlight','tpotMs','itlMs','qpsPerCard','peakGiB','memoryBudgetGiB','memoryFeasible','meets30msLimit','calibrationApplied','calibrationTransferred','extrapolated'];
    const rows=data.series.flatMap(s=>s.rows.map(r=>columns.map(k=>r[k]??s[k]??'').join(',')));
    download('\uFEFF'+[columns.join(','),...rows].join('\r\n')+'\r\n','text/csv;charset=utf-8','csv');
  });
  el('svg').addEventListener('click',()=>{
    // Export the visible curves with the fixed case and model scope attached.
    const chart=el('chart').cloneNode(true);chart.querySelector('#comparison-cursor')?.remove();
    const svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 790" width="1040" height="790" font-family="Segoe UI,Microsoft YaHei,sans-serif"><rect width="1040" height="790" fill="#0b1b27"/>${text(24,32,'GLM-5.3-Flash · 850 / 850E · 纯 Decode · v13','font-weight="700"')}${text(24,58,'128K / 1K · P 命中 95% · D 物理共享 95% · MTP5 · ITL/TPOT=3.5 · TP1 · 多流开启')}<g transform="translate(0 130)">${chart.innerHTML}</g>${data.series.filter((_,i)=>visible.has(i)).map((s,i)=>text(24+(i%3)*340,88+Math.floor(i/3)*24,`${s.hardware==='850'?'实线':'虚线'} ${name(s)}: ${s.summary.qpsPerCard.toFixed(3)} QPS/卡`, `style="fill:${colors[s.ranks]}"`)).join('')}${text(24,720,'80% 显存利用率 / 4 GiB 预留；淡线为超显存预算。B0 时延不定义。')}${text(24,746,'850 未本机校准；850E EP8 测点校正，EP16/32 迁移；六条曲线均为预测。')}${text(24,772,hardwareNote)}</svg>`;
    download(svg,'image/svg+xml;charset=utf-8','svg');
  });
  renderChart();inspect();
})();
