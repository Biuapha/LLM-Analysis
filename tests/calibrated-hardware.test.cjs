// No browser or image loading: validate the embedded engine and actual form readers.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../glm53_prefill_decode_operator_block_diagram.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],data={};
for(const [,a,b]of scripts)if(a.includes('application/json'))data[a.match(/id="([^"]+)"/)[1]]=JSON.parse(b);else new vm.Script(b);
const src=scripts.find(([,a,b])=>b.includes('function calculatePDSimulation('))[2];
const ctx=vm.createContext({memoryData:data['weight-data'],GiB:2**30,document:{getElementById:id=>({textContent:JSON.stringify(data[id])})}});
vm.runInContext(src.slice(src.indexOf('function calculateRankMemory('),src.indexOf('function readMemoryInputs('))+src.slice(src.indexOf('const pdData='),src.indexOf('/* Browser UI;')),ctx);
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const base=(hw,input,output=1024)=>({p:{hw,nodes:1,tp:1,pcp:8,ep:hw.startsWith('a3')?16:8,usableGiB:0},d:{hw,nodes:1,tp:hw.startsWith('a3')?2:1,ep:hw.startsWith('a3')?16:8,usableGiB:0},input,output,ttft:2,itl:1,tpot:.02,dConstraint:'tpot',fidelity:'analytic',decodeMode:'mtp',mtpDraft:5,dPrefixLayout:'shared',fixedP:0,fixedD:0,maxP:32,maxD:256,util:80,reserve:4,chunk:16384,pScale:1,dScale:1,computeScale:1,memoryScale:1,overlap:0,imbalance:1.1,prefixHitRate:0,ratioSource:'custom',itlTpotRatio:3.5,constraintStatistic:'mean'});
// Current model estimates for the 14 workbook presets, not measured hardware truth.
// Includes the original workbook's 16396 input. Updated for the BF16 KPool model.
const expected={
  "a3-560": [
    [
      1956.9561699678557,
      3.065985887715898
    ],
    [
      5303.637693022711,
      0.37709966550526886
    ],
    [
      3056.221609435002,
      3.926416841944396
    ],
    [
      49.85282431752751,
      17.255091673131187
    ],
    [
      49.95542494006393,
      15.028060290361557
    ],
    [
      19.736702512628657,
      9.113130599703597
    ],
    [
      19.915238178004277,
      7.853420363339667
    ]
  ],
  "a5-96": [
    [
      1923.8599505882373,
      4.158306844297023
    ],
    [
      3998.9779589669574,
      0.500127787780219
    ],
    [
      6652.1354500703355,
      4.810485330640953
    ],
    [
      19.53674484866552,
      17.612249336863275
    ],
    [
      21.17299155227004,
      17.35921119862512
    ],
    [
      19.53674484866552,
      17.612249336863275
    ],
    [
      19.831114607913342,
      16.562142587261565
    ]
  ]
};
for(const hw of Object.keys(expected))for(let row=3;row<=9;row++){
 const p=row<=5,c=base(hw,[4,5,7,9].includes(row)?131072:row===3&&hw==='a3-560'?16396:16384,p?1:1024);
 c.ttft=row===4?15:row===5?7:2;c.prefixHitRate=row===5?90:0;c.tpot=row===6||row===7?.05:.02;
 c.itl=3.5*c.tpot;c.dConstraint='both';
 if(p){c.chunk=hw==='a5-96'?32768:row===5?16384:4096;c.util=hw==='a5-96'?90:92;c.maxP=256;}
 const r=ctx.calculatePDSimulation(c),b=p?r.p.best:r.d.best;
 assert.ok(r.valid&&b,hw+' row '+row);close((p?b.seconds:b.tpotSeconds)*1000,expected[hw][row-3][0]);close(b.qps,expected[hw][row-3][1]);
 if(!p)close(b.itlTpotRatio,3.5);
 const native=false;
 assert.equal(Boolean(b.calibration.targetNative),native);assert.equal(Boolean(b.calibration.empiricalOnly),false);
 assert.equal(Boolean(b.calibration.transfer),false);
 if(p)assert.notEqual(ctx.pdStageBoundary(r.p).kind,'scan_limit');
 if(p&&row!==3){
  assert.ok(r.p.results.some(x=>x.feasible&&x.batch>b.batch&&x.qps<b.qps));
  assert.equal(ctx.pdStageBoundary(r.p).kind,'throughput_optimum');
 }
}
assert.deepEqual(data['pd-data'].devices.map(d=>d.id).sort(),['a3-560','a3-752','a3-800','a5-128','a5-96']);
for(const hw of ['a3-752','a3-560','a5-96','a5-128']){
 const r=ctx.calculatePDSimulation(base(hw,16384));
 assert.ok(r.p.best&&r.d.best);close(r.d.best.itlTpotRatio,3.5);
 assert.equal(r.d.best.projection.acceptanceMeasured,false);
}
const source={...base('a3-800',16384),fidelity:'calibrated'};source.d.tp=4;source.ratioSource='reference';source.itlTpotRatio=1;
const actual=ctx.calculatePDSimulation(source);assert.ok(actual.p.best&&actual.d.best);assert.equal(actual.d.best.calibration.empiricalOnly,true);
for(const fidelity of ['analytic','ideal']){const r=ctx.calculatePDSimulation({...source,fidelity});assert.equal(r.p.best,null);assert.equal(r.d.best,null);}
const controls=new Map();
const element=()=>{let value='';return {get value(){return value},set value(v){value=String(v)},textContent:'',style:{},addEventListener(){}}};
for(const m of html.matchAll(/<input\b[^>]*id="pd-([^"]+)"[^>]*>/g)){const e=element();e.value=m[0].match(/\bvalue="([^"]*)"/)?.[1]??'';controls.set(m[1],e);}
for(const m of html.matchAll(/<select\b[^>]*id="pd-([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
 const options=[...m[2].matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)],selected=options.filter(o=>/\bselected\b/.test(o[1])).at(-1)||options[0];
 const e=element();e.value=selected?(selected[1].match(/\bvalue="([^"]*)"/)?.[1]??selected[2]):'';controls.set(m[1],e);
}
for(const side of ['p','d'])controls.get(side+'-hw').value='a3-560';
ctx.pdEl=k=>{if(!controls.has(k))controls.set(k,element());return controls.get(k)};
ctx.document={getElementById:id=>ctx.pdEl(id.replace(/^pd-/,''))};
ctx.pdNativeSaved=()=>null;ctx.pdNativeControls=()=>{};ctx.pdModeControls=()=>{};ctx.renderPDSimulation=()=>{};
function load(name){const a=src.indexOf('function '+name+'('),b=src.indexOf('\nfunction ',a+1);assert.ok(a>=0&&b>a);vm.runInContext(src.slice(a,b),ctx);}
load('readPDInputs');load('pdLoadTableScenario');load('pdLoadWorkbookCase');
assert.match(src,/if\(hw.id==='a3-560'\)o.selected=true/);
const initial=ctx.readPDInputs();assert.equal(initial.p.hw,'a3-560');close(initial.itlTpotRatio,3.5);assert.equal(initial.ratioSource,'custom');assert.equal(initial.maxD,256);
for(const hw of ['a3-560','a5-96']){
 ctx.pdLoadTableScenario(hw);const c=ctx.readPDInputs(),r=ctx.calculatePDSimulation(c);
 assert.ok(r.valid&&r.p.best&&r.d.best);assert.equal(c.d.tp,hw==='a3-560'?2:1);close(r.d.best.itlTpotRatio,3.5);
}
assert.equal(initial.d.tp,2);
for(let i=0;i<14;i++){
 ctx.pdLoadWorkbookCase(String(i));const c=ctx.readPDInputs(),r=ctx.calculatePDSimulation(c);
 const hw=i<7?'a3-560':'a5-96',row=i%7+3,p=row<=5,b=p?r.p.best:r.d.best;
 if(p){assert.equal(c.chunk,hw==='a5-96'?32768:row===5?16384:4096);assert.equal(c.util,hw==='a5-96'?90:92);assert.equal(c.maxP,256);}
 assert.ok(b);assert.equal(c.mtpDraft,5);assert.equal(c.dConstraint,'both');close(c.itl,3.5*c.tpot);assert.equal(r.config.p.dp,hw==='a3-560'?2:1);assert.equal(r.config.d.dp,8);
 assert.equal(r.config.d.tp,hw==='a3-560'?2:1);close(b.qps,expected[hw][row-3][1]);close((p?b.seconds:b.tpotSeconds)*1000,expected[hw][row-3][0]);
}
assert.ok(!html.includes('校准验收报告.html'));
console.log('PASS: 14 workbook cases, all five hardware profiles, source isolation, fixed r=3.5, real form defaults, A3/A5 preset handlers and embedded script syntax. No images loaded.');
