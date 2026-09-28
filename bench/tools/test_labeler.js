/*
 * bench/tools/test_labeler.js
 *
 * labeler.html 的核心逻辑测试。真值一旦被工具悄悄改错，后面所有指标都作废，
 * 所以这里用一个最小 DOM stub 在 node 里跑真实源码，重点守住三条不变式：
 *   1. 分段必须覆盖全片且首尾相接（无空隙、无重叠）
 *   2. 边界严格升序、相邻段标签不同
 *   3. CSV 导出 -> 导入 完全等价（round-trip）
 *
 * 用法：node bench/tools/test_labeler.js
 */
const fs = require('fs'), vm = require('vm'), path = require('path');
const HTML = path.join(__dirname, 'labeler.html');
const m = /<script>\n([\s\S]*)\n<\/script>/.exec(fs.readFileSync(HTML, 'utf8'));
if (!m){ console.error('在 labeler.html 里找不到 <script> 块'); process.exit(1); }
const src = m[1];

/* ---- 最小 DOM stub ---- */
const noop = () => {};
function fakeCtx(){
  return {setTransform:noop, clearRect:noop, fillRect:noop, beginPath:noop, moveTo:noop,
          lineTo:noop, stroke:noop, fillText:noop, save:noop, restore:noop,
          fillStyle:'', strokeStyle:'', lineWidth:1, font:'', textBaseline:''};
}
function makeEl(id){
  const el = {
    id, style:{}, dataset:{}, value:'', textContent:'', innerHTML:'', checked:false,
    classList:{add:noop, remove:noop, contains:()=>false, toggle:noop},
    className:'',
    addEventListener:noop, removeEventListener:noop, appendChild:noop, click:noop,
    focus:noop, select:noop, blur:noop, scrollIntoView:noop, setAttribute:noop,
    getAttribute:n => n === 'height' ? '62' : null,
    getContext:fakeCtx, getBoundingClientRect:()=>({left:0,top:0,width:800,height:62}),
    clientWidth:800, clientHeight:62, width:800, height:62,
    querySelector:()=>makeEl('q'), querySelectorAll:()=>[],
    // video bits
    currentTime:0, duration:0, paused:true, playbackRate:1, muted:false, src:'',
    play:noop, pause:noop
  };
  return el;
}
const els = {};
function getEl(id){ return els[id] || (els[id] = makeEl(id)); }
els.selZoom = makeEl('selZoom'); els.selZoom.value = '5000';
els.selRate = makeEl('selRate'); els.selRate.value = '1';
els.inFps   = makeEl('inFps');   els.inFps.value   = '30';

let blobCapture = null;
const ctx = {
  console,
  setTimeout:(f)=>{ f(); return 0; }, clearTimeout:noop,
  requestAnimationFrame:noop,
  devicePixelRatio:1,
  document:{ getElementById:getEl, createElement:()=>makeEl('new'),
             querySelector:()=>makeEl('sel'), querySelectorAll:()=>[], addEventListener:noop },
  ResizeObserver:class{ constructor(){} observe(){} },
  localStorage:(()=>{ const m=new Map(); return {getItem:k=>m.has(k)?m.get(k):null,
             setItem:(k,v)=>m.set(k,v), removeItem:k=>m.delete(k)}; })(),
  Blob:class{ constructor(parts){ blobCapture = parts.join(''); } },
  URL:{ createObjectURL:()=>'blob:x', revokeObjectURL:noop },
  alert:m=>{ ctx.__alerts.push(String(m)); }, confirm:()=>true, prompt:()=>null,
  Date, Math, JSON, isFinite, parseInt, parseFloat, String, Number, Array, Object, Error,
  __alerts:[]
};
ctx.window = ctx;
ctx.window.addEventListener = noop;
ctx.globalThis = ctx;
vm.createContext(ctx);

// 测试钩子：暴露 script 内的 let/const 绑定（它们不是 globalThis 属性）
const epilogue = `
globalThis.__api = {
  get cuts(){ return cuts; }, set cuts(v){ cuts = v; },
  get durationMs(){ return durationMs; }, set durationMs(v){ durationMs = v; },
  get meta(){ return meta; },
  get hasVideo(){ return hasVideo; }, set hasVideo(v){ hasVideo = v; },
  get vid(){ return vid; },
  normalize, segEnd, segIndexAt, nearestCutIdx, computeStats,
  csvCell, parseCsvLine, exportCsv, importCsv,
  setLabelAtPlayhead, deleteSelectedCut, nudgeSelected, undo, nowMs, fmt, parseTime,
  get selT(){ return selT; }, set selT(v){ selT = v; }
};`;
vm.runInContext(src + '\n' + epilogue, ctx, {filename:'labeler.js'});
const A = ctx.__api;

/* ---- 断言 ---- */
let pass = 0, fail = 0;
function ok(name, cond, extra){
  if (cond){ pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (extra !== undefined ? '  => ' + JSON.stringify(extra) : '')); }
}
function eq(name, got, want){ ok(name, JSON.stringify(got) === JSON.stringify(want), {got, want}); }
function invariants(name){
  const c = A.cuts, D = A.durationMs;
  let good = c.length > 0 && c[0].t === 0;
  for (let i = 0; i < c.length; i++){
    if (A.segEnd(i) !== (i + 1 < c.length ? c[i + 1].t : D)) good = false;
    if (i > 0 && c[i].t <= c[i - 1].t) good = false;
    if (i > 0 && c[i].label === c[i - 1].label) good = false;
  }
  ok(name + ' [不变式: 首0/无缝/严格升序/相邻不同标签]', good, c);
}

console.log('\n== 载入 ==');
ok('脚本载入无异常', true);
ok('载入期间无 alert', ctx.__alerts.length === 0, ctx.__alerts);

console.log('\n== normalize ==');
A.durationMs = 100000;
A.cuts = [{t:0,label:'plot',note:''},{t:1000,label:'plot',note:'x'},{t:2000,label:'sex',note:''}];
A.normalize();
eq('相邻同标签合并（备注并入）', A.cuts, [{t:0,label:'plot',note:'x'},{t:2000,label:'sex',note:''}]);

A.cuts = [{t:500,label:'sex',note:''},{t:3000,label:'plot',note:''}];
A.normalize();
ok('首个边界被强制到 0ms', A.cuts[0].t === 0);
invariants('normalize 后');

A.cuts = [{t:0,label:'plot',note:''},{t:5000,label:'sex',note:''},{t:5000,label:'oral',note:''}];
A.normalize();
eq('同一时刻去重（保留后写入的）', A.cuts.map(c=>c.label), ['plot','oral']);

A.cuts = [{t:0,label:'plot',note:''},{t:9000,label:'sex',note:''},{t:3000,label:'oral',note:''}];
A.normalize();
eq('乱序输入被排序', A.cuts.map(c=>c.t), [0,3000,9000]);

console.log('\n== 打点 ==');
A.hasVideo = true; A.durationMs = 60000;
A.cuts = [{t:0,label:'plot',note:''}];
A.vid.currentTime = 12.4;   A.setLabelAtPlayhead('oral');
A.vid.currentTime = 30.0;   A.setLabelAtPlayhead('plot');
A.vid.currentTime = 42.5;   A.setLabelAtPlayhead('sex');
eq('三次打点得到 4 段', A.cuts.map(c=>[c.t,c.label]), [[0,'plot'],[12400,'oral'],[30000,'plot'],[42500,'sex']]);
invariants('打点后');

const n0 = A.cuts.length;
A.vid.currentTime = 12.41;  A.setLabelAtPlayhead('sex');   // 容差内 -> 改标签而非新增
eq('容差内重打改标签不新增边界', [A.cuts.length, A.cuts[1].label], [n0, 'sex']);

A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'sex',note:''}];
A.vid.currentTime = 20;     A.setLabelAtPlayhead('sex');   // 与前段同标签 -> 合并消失
eq('打成与前段相同的标签 -> 自动合并', A.cuts.map(c=>[c.t,c.label]), [[0,'plot'],[10000,'sex']]);

console.log('\n== 删除 / 微调 / 撤销 ==');
A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'sex',note:''},{t:20000,label:'plot',note:''}];
A.selT = 10000; A.deleteSelectedCut();
eq('删除 10000 后只剩 plot 一段（因合并）', A.cuts.map(c=>[c.t,c.label]), [[0,'plot']]);
A.undo();
eq('撤销恢复删除', A.cuts.map(c=>c.t), [0,10000,20000]);

A.selT = 0; const before = JSON.stringify(A.cuts); A.deleteSelectedCut();
eq('0ms 起始边界不可删除', JSON.stringify(A.cuts), before);

A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'sex',note:''},{t:20000,label:'plot',note:''}];
A.selT = 10000; A.nudgeSelected(-33.3);
ok('边界微调生效且落在相邻边界之间', A.cuts[1].t === 9967 && A.cuts[1].t > 0 && A.cuts[1].t < 20000, A.cuts[1].t);
A.selT = 10000 - 33; A.nudgeSelected(-1e9);
ok('微调被钳制，不会越过前一个边界', A.cuts[1].t > 0, A.cuts[1].t);
invariants('微调后');

console.log('\n== 统计 ==');
A.durationMs = 100000;
A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'sex',note:''},{t:30000,label:'plot',note:''},
          {t:40000,label:'oral',note:''},{t:70000,label:'exclude',note:''},{t:80000,label:'plot',note:''}];
let st = A.computeStats();
const sum = Object.values(st.per).reduce((a,b)=>a+b,0);
eq('各标签时长之和 == 片长', sum, A.durationMs);
eq('do↔noise 切换点计数', st.trans, 4);   // noise->do->noise->do->(skip exclude)->noise
eq('段数', st.nseg, 6);

A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'other_adult',note:''},{t:20000,label:'sex',note:''}];
st = A.computeStats();
eq('other_adult 不参与二值，noise->do 记 1 次', st.trans, 1);

A.cuts = [{t:0,label:'plot',note:''},{t:10000,label:'sex',note:''},{t:10400,label:'plot',note:''}];
st = A.computeStats();
eq('短段(400ms)被标记', st.shortIdx, [1]);

console.log('\n== CSV ==');
eq('csvCell 转义逗号', A.csvCell('a,b'), '"a,b"');
eq('csvCell 转义引号', A.csvCell('say "hi"'), '"say ""hi"""');
eq('parseCsvLine 解析引号字段', A.parseCsvLine('1,2,plot,"a,b ""c"""'), ['1','2','plot','a,b "c"']);

A.durationMs = 145000;
A.cuts = [{t:0,label:'plot',note:''},{t:12400,label:'oral',note:''},
          {t:58900,label:'exclude',note:'黑屏, 过场 "x"'},{t:61200,label:'sex',note:''}];
A.meta.video_id = 'v01'; A.meta.annotator = 'tingfu';
blobCapture = null; A.exportCsv();
const csv = blobCapture;
ok('导出含 spec_version 头', /^# xcup-gt spec_version=1/.test(csv), csv && csv.split('\n')[0]);
ok('导出含表头', csv.includes('start_ms,end_ms,label,note'));
const dataRows = csv.split('\n').filter(l => l && !l.startsWith('#') && !l.startsWith('start_ms'));
eq('导出行数 == 段数', dataRows.length, 4);
eq('末行 end_ms == 片长', dataRows[3].split(',')[1], '145000');
// 首尾相接
let contiguous = true;
for (let i = 1; i < dataRows.length; i++)
  if (A.parseCsvLine(dataRows[i])[0] !== A.parseCsvLine(dataRows[i-1])[1]) contiguous = false;
ok('导出的区间首尾相接（无空隙无重叠）', contiguous, dataRows);

const snapshot = JSON.stringify(A.cuts);
A.cuts = [{t:0,label:'plot',note:''}];
ctx.__alerts = [];
A.importCsv(csv);
eq('导入后与导出前完全一致（round-trip）', JSON.stringify(A.cuts), snapshot);
eq('round-trip 无告警', ctx.__alerts, []);
invariants('导入后');

ctx.__alerts = [];
A.importCsv('start_ms,end_ms,label,note\n0,1000,bogus_label,\n');
ok('非法标签被拒绝并提示', ctx.__alerts.length === 1 && /bogus_label/.test(ctx.__alerts[0]), ctx.__alerts);

console.log('\n== 时间格式 ==');
eq('fmt', A.fmt(61234), '01:01.234');
eq('parseTime mm:ss.mmm', A.parseTime('01:01.234'), 61234);
eq('parseTime 纯毫秒', A.parseTime('61234'), 61234);
eq('parseTime h:mm:ss', A.parseTime('1:02:03.500'), 3723500);
eq('parseTime 非法', A.parseTime('abc'), null);

console.log('\n' + (fail ? 'FAILED' : 'ALL PASS') + ': ' + pass + ' passed, ' + fail + ' failed\n');
process.exit(fail ? 1 : 0);
