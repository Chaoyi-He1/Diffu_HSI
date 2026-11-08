/* ---------- Phase 3 figures, charts, tables and slots (helpers el, tx, box, arr, line, mk, rrect, cpath, hover, showTip, hideTip, svgPt, dataTable, clear come from the shared block) ---------- */
function fx(v,n){return v==null||isNaN(v)?'—':Number(v).toFixed(n);}
function sg(v,n){return v==null?'—':(v>0?'+':(v<0?'−':''))+Math.abs(Number(v)).toFixed(n);}
function pill(ok,txt){return '<span class="pill '+(ok?'ok':'no')+'">'+(txt||(ok?'通过':'未通过'))+'</span>';}
function pend(txt){return '<span class="pill nt">'+(txt||'待定')+'</span>';}
function prev(ok){return '<span class="pill wn">'+(ok?'预览：通过':'预览：未通过')+'</span>';}
function seg(attr,draw){const bs=document.querySelectorAll('[data-'+attr+']');bs.forEach(function(b){b.addEventListener('click',function(){bs.forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');draw(b.getAttribute('data-'+attr));});});}
function tbl(host,head,rows,hiRow){const w=document.createElement('table');w.className='t';let h='<thead><tr>'+head.map(function(x){return '<th>'+x+'</th>';}).join('')+'</tr></thead><tbody>';rows.forEach(function(r,i){h+='<tr'+(hiRow&&hiRow(i)?' class="hi"':'')+'>'+r.map(function(c){return typeof c==='object'&&c!==null?'<td class="'+(c.cls||'')+'">'+c.v+'</td>':'<td>'+c+'</td>';}).join('')+'</tr>';});w.innerHTML=h+'</tbody>';host.appendChild(w);}
function num(v){return {v:v,cls:'num'};}
function sci(v){if(v==null)return '—';const e=Math.floor(Math.log10(v));return (v/Math.pow(10,e)).toFixed(1)+' × 10'+String(e).split('').map(function(c){return '⁰¹²³⁴⁵⁶⁷⁸⁹'['0123456789'.indexOf(c)]||(c==='-'?'⁻':c);}).join('');}

const S=D.settings,V=D.verdict,ST=D.status,F0=D.facts;
const MC={L:'--ink-2',L_demosaic:'--de',D:'--accent',V:'--s3',C:'--s2',C1:'--s2',B:'--s4',Bv:'--s4'};
const MN={L:'L，MMSE',L_demosaic:'L，先去马赛克再 MMSE',D:'D，回归器',V:'V，方差头',C:'C，32 样本均值',C1:'C1，单次前向',B:'B，32 样本均值（50 个 cube）',Bv:'B，v 预测先验（已归档）'};
/* zh: display names for strings that come from page_data.json */
const PN={blind:'输入无关预测器 E[v | t]',vpred:'v 预测先验（第 100 轮）',g0:'均匀加权试验（第 10 轮）',tt:'时间令牌试验（第 10 轮）',x0pilot:'x₀ 预测试验（第 10 轮）',x0:'x₀ 预测先验（第 100 轮）'};
const ROLE={'B prior (x₀-prediction)':'B 先验（x₀ 预测）','C, conditional diffusion':'C，条件扩散模型','D, regressor':'D，回归器','V, variance head':'V，方差头','B prior (v-prediction, failed)':'B 先验（v 预测，失败）'};
const MODEZ={sequential:'顺序',snapshot:'快照'};
const ZW={passed:'通过',failed:'未通过'};
const DEVN={device1:'设备 1',device2:'设备 2',ph5:'PH5'};
function set(dev,mode,snr){return S[dev+'_'+mode+'_'+snr];}
function bOk(s){return s&&s.B&&s.B.status==='final';}
function rmse(s,m){if(!s)return null;if(m==='B')return bOk(s)?s.B.means.mean_rmse_pct:null;const x=s.methods[m];if(!x)return null;return (m==='D'||m==='C1')?x.rmse_pct:x.mean_rmse_pct;}
function met(s,m,k){if(!s)return null;if(m==='B')return bOk(s)?s.B.means[k]:null;const x=s.methods[m];return x?x[k]:null;}
function vk(snr){return Number(snr).toFixed(1);}
function bStatusTxt(s){if(!s||!s.B)return '待定';return s.B.status==='dropped'?'已撤销':(s.B.status==='final'?'':'待定');}
function legendInto(name,items){const h=document.querySelector('[data-legend="'+name+'"]');if(!h)return;h.innerHTML=items.map(function(q){return '<span><i'+(q.dash?' class="dash" style="border-top-color:var('+q.c+')"':' style="background:var('+q.c+')'+(q.faint?';opacity:.5':'')+'"')+'></i>'+q.n+'</span>';}).join('');}

/* ---------- generic x-y chart with optional log y ---------- */
function xy(host,o){
  const W=o.w||720,H=o.h||320,m={l:o.ml||56,r:(o.mr==null?120:o.mr),t:22,b:44};
  const svg=mk(host,W,H,o.minW||520);const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const X=function(v){return m.l+(v-o.x[0])/(o.x[1]-o.x[0])*pw;};
  let Y;if(o.log){const a=Math.log10(o.y[0]),b=Math.log10(o.y[1]);Y=function(v){return m.t+ph-(Math.log10(Math.max(v,o.y[0]))-a)/(b-a)*ph;};}else{Y=function(v){return m.t+ph-(v-o.y[0])/(o.y[1]-o.y[0])*ph;};}
  o.yTicks.forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,(o.yFmt||String)(t),'tick','end');});
  line(svg,m.l,m.t+ph,m.l+pw,m.t+ph,'axis');
  o.xTicks.forEach(function(t){tx(svg,X(t),m.t+ph+18,(o.xFmt||String)(t),'tick','middle');});
  if(o.yTitle)tx(svg,4,13,o.yTitle,'axt','start');if(o.xTitle)tx(svg,m.l+pw,m.t+ph+38,o.xTitle,'axt','end');
  const live=o.series.filter(function(s){return s.data.length;});
  live.forEach(function(s){
    const d=s.data.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ');
    if(s.data.length>1)el('path',{d:d,class:'ln',style:'stroke:var('+s.c+');'+(s.faint?'opacity:.55;':'')+(s.dash?'stroke-dasharray:'+(s.da||'5 4')+';':'')+(s.thin?'stroke-width:1.3;':'')},svg);
    if(s.marks!==false)s.data.forEach(function(p){el('circle',{cx:X(p[0]),cy:Y(p[1]),r:s.r||3.2,class:'dot',style:'fill:var('+s.c+');'+(s.faint?'opacity:.6':'')},svg);});
    if(s.label){const L=s.data[o.labelAt==='first'?0:s.data.length-1];tx(svg,X(L[0])+(o.labelAt==='first'?-8:8),Y(L[1])+(s.ldy||4),s.label,'endlbl',o.labelAt==='first'?'end':'start');}
  });
  (o.refs||[]).forEach(function(r){const x1=r.x0==null?m.l:X(r.x0),x2=r.x1==null?m.l+pw:X(r.x1);line(svg,x1,Y(r.y),x2,Y(r.y),r.cls||'ref');tx(svg,r.lx==null?x2+6:X(r.lx),Y(r.y)+(r.dy==null?4:r.dy),r.label,'reflbl',r.anchor||'start');});
  const cross=el('line',{y1:m.t,y2:m.t+ph,class:'cross',visibility:'hidden'},svg);
  const hit=el('rect',{x:m.l,y:m.t,width:pw,height:ph,class:'hit'},svg);
  hit.addEventListener('pointermove',function(e){const p=svgPt(svg,e);const xv=o.x[0]+(p.x-m.l)/pw*(o.x[1]-o.x[0]);
    let best=null;live.forEach(function(s){s.data.forEach(function(q){if(best==null||Math.abs(q[0]-xv)<Math.abs(best-xv))best=q[0];});});if(best==null)return;
    const rows=[];live.forEach(function(s){s.data.forEach(function(q){if(q[0]===best)rows.push('<div class="tr"><i style="background:var('+s.c+')"></i>'+s.name+'<b>'+(o.tipY||function(v){return v.toFixed(3);})(q[1])+'</b></div>');});});
    cross.setAttribute('x1',X(best));cross.setAttribute('x2',X(best));cross.setAttribute('visibility','visible');
    showTip(e,'<div class="sub">'+(o.tipX||String)(best)+'</div>'+rows.join(''));});
  hit.addEventListener('pointerleave',function(){cross.setAttribute('visibility','hidden');hideTip();});
  if(o.table)dataTable(host,o.table.head,o.table.rows);
  return {svg:svg,X:X,Y:Y,m:m,pw:pw,ph:ph};
}
/* grouped bars; a null value draws a 'pending' / 'dropped' tag at the base */
function bars(h,o){const W=o.w||720,H=o.h||300,m={l:56,r:16,t:22,b:50};const svg=mk(h,W,H,o.minW||560);const pw=W-m.l-m.r,ph=H-m.t-m.b;const Y=function(v){return m.t+ph-(v-o.y[0])/(o.y[1]-o.y[0])*ph;};
  o.ticks.forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,o.fmt(t),'tick','end');});line(svg,m.l,Y(o.y[0]),m.l+pw,Y(o.y[0]),'axis');tx(svg,4,13,o.yTitle,'axt');
  const band=pw/o.groups.length;const nb=o.series.length;const bw=Math.min(44,band/(nb+1.4));
  (o.hrefs||[]).forEach(function(r){line(svg,m.l,Y(r.y),m.l+pw,Y(r.y),'refacc');tx(svg,m.l+pw-4,Y(r.y)-6,r.label,'reflbl','end');});
  o.groups.forEach(function(g,gi){const cx=m.l+band*gi+band/2;tx(svg,cx,H-m.b+22,g.label,'bs','middle');
    o.series.forEach(function(s,si){const v=g.values[si];const x=cx+(si-(nb-1)/2)*(bw+5)-bw/2;
      if(v==null){const t=(g.tags&&g.tags[si])||'待定';const tt=tx(svg,x+bw/2,Y(o.y[0])-6,t,'sm','middle');tt.setAttribute('transform','rotate(-90 '+(x+bw/2)+' '+(Y(o.y[0])-6)+')');tt.setAttribute('text-anchor','start');return;}
      const vv=Math.min(v,o.y[1]);el('path',{d:cpath(x,Y(o.y[0]),bw,Y(o.y[0])-Y(vv),4),style:'fill:var('+s.c+')'+(s.faint?';opacity:.55':'')},svg);
      tx(svg,x+bw/2,Y(vv)-6,o.vfmt(v),'vlbl','middle');
      const hit=el('rect',{x:x,y:m.t,width:bw+5,height:ph,fill:'transparent',class:'barhit'},svg);hover(hit,function(){return '<div><b style="margin:0;padding:0">'+g.label+' · '+s.name+'</b></div><div class="sub">'+o.vfmt(v)+(g.notes&&g.notes[si]?' · '+g.notes[si]:'')+'</div>';});});});
  const lg=document.createElement('div');lg.className='legend';lg.innerHTML=o.series.map(function(s){return '<span><i class="sq" style="background:var('+s.c+')"></i>'+s.name+'</span>';}).join('');h.parentNode.insertBefore(lg,h);
  dataTable(h,[o.gname||'组'].concat(o.series.map(function(s){return s.name;})),o.groups.map(function(g){return [g.label].concat(g.values.map(function(v,i){return v==null?((g.tags&&g.tags[i])||'待定'):o.vfmt(v);}));}));}

/* ---------- figures ---------- */
const F={};
F.pipeline=function(h){
  const s=mk(h,960,280,760);
  tx(s,10,20,'测量','tag');
  box(s,10,32,170,62,['场景 x（31 个波段）','HFD 门控 cube'],'box');
  arr(s,[[184,63],[222,63]]);
  box(s,226,32,232,62,['传感器 Rᵀ，30 个状态','顺序或 5 × 6 快照'],'box');
  arr(s,[[462,63],[500,63]]);
  box(s,504,32,170,62,['y = Rᵀx + n','σ 由信噪比与全局 P_y 确定'],'box');
  tx(s,288,128,'经由充分统计量','tag');
  box(s,10,140,232,62,['L：闭式 MMSE','x̂ 与后验标准差 s'],'box');
  arr(s,[[588,98],[588,118],[126,118],[126,136]]);
  arr(s,[[246,171],[284,171]]);
  box(s,288,140,120,62,['D','单次前向'],'box box-acc');
  box(s,418,140,120,62,['V','高斯宽度'],'box box-s3');
  box(s,548,140,126,62,['C','32 个样本'],'box box-s2');
  tx(s,10,236,'直接从测量出发','tag');
  box(s,10,206+42,664,26,['B：无条件先验（仅用光谱训练）+ 使用精确 H、σ 与掩膜的 ΠGDM 引导；32 个样本'],'box box-s4');
  arr(s,[[678,63],[730,63],[730,261],[678,261]],'ar');
  tx(s,750,20,'评分','tag');
  box(s,750,32,200,170,['E1 设备 1，5 个信噪比','E2 设备 2、PH5','E3 快照','RMSE · CRPS','覆盖率 · AUSE','条款 T1–T4'],'box');
  arr(s,[[678,171],[746,171]]);
};

const CH={};
CH.sensorR=function(h){const R=D.sensor.R_device1,wl=D.sensor.wavelengths;const n=R[0].length;
  const series=[];for(let j=0;j<n;j++){series.push({name:'状态 '+(j+1),c:'--accent',data:wl.map(function(w,i){return [w,R[i][j]];}),marks:false,thin:true});}
  const o=xy(h,{w:520,h:300,mr:20,minW:380,x:[451,855],y:[0,1],xTicks:[451,550,650,750,855],yTicks:[0,0.25,0.5,0.75,1],yFmt:function(v){return v.toFixed(2);},yTitle:'归一化响应',xTitle:'波长（nm，旧版波长轴）',series:series,tipX:function(v){return v.toFixed(0)+' nm';},tipY:function(v){return v.toFixed(3);}});
  o.svg.querySelectorAll('path.ln').forEach(function(p,j){p.style.opacity=(0.25+0.75*j/(n-1)).toFixed(2);});
  dataTable(h,['波段（nm）'].concat(R[0].map(function(_,j){return 's'+(j+1);})),wl.map(function(w,i){return [w.toFixed(1)].concat(R[i].map(function(v){return v.toFixed(3);}));}));};
CH.sensorSV=function(h){const dv=D.sensor.devices;const cs={device1:'--accent',device2:'--s3',ph5:'--s2'};
  const ser=['device1','device2','ph5'].map(function(k){return {name:DEVN[k],c:cs[k],label:DEVN[k],marks:false,data:dv[k].sv_rel.map(function(v,i){return [i+1,Math.max(v,1e-12)];})};});
  xy(h,{w:520,h:300,mr:70,minW:380,log:true,x:[1,30],y:[1e-12,1],xTicks:[1,5,10,15,20,25,30],yTicks:[1e-12,1e-9,1e-6,1e-3,1],yFmt:function(v){return '1e'+Math.round(Math.log10(v));},yTitle:'σᵢ / σ₁（对数）',xTitle:'序号 i',series:ser,refs:[{y:0.01,label:'1 %',lx:30,anchor:'end',dy:-6}],tipX:function(v){return '第 '+v+' 个奇异值';},tipY:function(v){return v.toExponential(2);},
    table:{head:['i','设备 1','设备 2','PH5'],rows:dv.device1.sv_rel.map(function(_,i){return [i+1,dv.device1.sv_rel[i].toExponential(2),dv.device2.sv_rel[i].toExponential(2),dv.ph5.sv_rel[i].toExponential(2)];})}});};
CH.stage0=function(h){const L=D.stage0.L_curves,raw=D.stage0.curves_raw_device1;const snr=[20,30,40,50,60];
  const g=function(o,dev){return snr.map(function(x){return [x,o[vk(x)]];});};
  const ser=[{name:'设备 1',c:'--accent',label:'设备 1',data:g(L.device1.sequential)},{name:'设备 1，原始响应',c:'--accent',dash:true,faint:true,data:g(raw.sequential)},
    {name:'设备 2',c:'--s3',label:'设备 2',data:g(L.device2.sequential)},{name:'PH5',c:'--s2',label:'PH5',data:g(L.ph5.sequential)}];
  xy(h,{log:true,x:[11,62],y:[0.8,15],xTicks:snr,xFmt:function(v){return v===60?'60（噪声底）':String(v);},yTicks:[1,2,5,10],yTitle:'MMSE RMSE %（对数）',xTitle:'信噪比（dB）',labelAt:'first',ml:56,mr:20,series:ser,tipX:function(v){return v+' dB';},
    table:{head:['信噪比（dB）','设备 1','设备 1 原始 R','设备 2','PH5'],rows:snr.slice().reverse().map(function(x){return [x,fx(L.device1.sequential[vk(x)],3),fx(raw.sequential[vk(x)],3),fx(L.device2.sequential[vk(x)],3),fx(L.ph5.sequential[vk(x)],3)];})}});
  const svg=h.querySelector('svg');};
function e1series(metric){const snrs=[20,30,40,50,60];const out={};
  ['L','D','V','C','C1','B'].forEach(function(m){out[m]=[];snrs.forEach(function(x){const s=set('device1','sequential',x);const v=metric==='rmse'?rmse(s,m):met(s,m,'crps');if(v!=null)out[m].push([x,v]);});});
  out.Bv=[];snrs.forEach(function(x){const b=D.b_vpred[String(x)];if(b){const v=metric==='rmse'?b.means.mean_rmse_pct:b.means.crps;out.Bv.push([x,v]);}});return out;}
CH.e1rmse=function(h){const d=e1series('rmse');const ms=['L','D','C','C1','B','Bv'];
  legendInto('e1',ms.map(function(m){return {n:MN[m]+(m==='B'&&!d.B.length?' — 待定':''),c:MC[m],dash:m==='C1'||m==='Bv',faint:m==='Bv'};}));
  const ser=ms.map(function(m){return {name:MN[m],c:MC[m],data:d[m],dash:m==='C1'||m==='Bv',faint:m==='Bv',label:{L:'L',D:'D',C:'C',B:'B'}[m]};});
  const snrs=[60,50,40,30,20];
  xy(h,{log:true,x:[12,62],y:[0.8,10],xTicks:[20,30,40,50,60],xFmt:function(v){return v===60?'60（噪声底）':String(v);},yTicks:[1,2,3,5,10],yTitle:'RMSE，占量程 %（对数）',xTitle:'信噪比（dB）',labelAt:'first',mr:20,series:ser,tipX:function(v){return v+' dB';},
    refs:[{y:D.stage0.sanity_reference_rmse_pct,x0:55,lx:54,anchor:'end',dy:4,label:'噪声底参考 '+fx(D.stage0.sanity_reference_rmse_pct,3)}],
    table:{head:['信噪比（dB）','L','D','C 32 样本均值','C1','C 单样本','B 32 样本均值（50）','B v 预测（已归档）'],rows:snrs.map(function(x){const s=set('device1','sequential',x);const bv=D.b_vpred[String(x)];return [x,fx(rmse(s,'L'),3),fx(rmse(s,'D'),3),fx(rmse(s,'C'),3),fx(rmse(s,'C1'),3),fx(met(s,'C','single_rmse_pct'),3),bOk(s)?fx(rmse(s,'B'),3):bStatusTxt(s),bv?fx(bv.means.mean_rmse_pct,3):'—'];})}});};
CH.e1crps=function(h){const d=e1series('crps');const ms=['L','V','C','B','Bv'];
  legendInto('e1c',ms.map(function(m){return {n:(m==='C'?'C，32 个样本':(m==='B'?'B，32 个样本（50 个 cube）':MN[m]))+(m==='B'&&!d.B.length?' — 待定':''),c:MC[m],dash:m==='Bv',faint:m==='Bv'};}));
  const ser=ms.map(function(m){return {name:MN[m],c:MC[m],data:d[m],dash:m==='Bv',faint:m==='Bv',label:{L:'L',V:'V',C:'C',B:'B'}[m],ldy:{L:-3,V:12}[m]};});
  xy(h,{log:true,x:[12,62],y:[0.4,8],xTicks:[20,30,40,50,60],xFmt:function(v){return v===60?'60（噪声底）':String(v);},yTicks:[0.5,1,2,4,8],yTitle:'CRPS %（对数）',xTitle:'信噪比（dB）',labelAt:'first',mr:20,series:ser,tipX:function(v){return v+' dB';},
    table:{head:['信噪比（dB）','L','V','C','B（50）','B v 预测（已归档）'],rows:[60,50,40,30,20].map(function(x){const s=set('device1','sequential',x);const bv=D.b_vpred[String(x)];return [x,fx(met(s,'L','crps'),3),fx(met(s,'V','crps'),3),fx(met(s,'C','crps'),3),bOk(s)?fx(met(s,'B','crps'),3):bStatusTxt(s),bv?fx(bv.means.crps,3):'—'];})}});};
const T2S=[30,20];
CH.t2crps=function(h){const ms=['L','V','C','B'];
  bars(h,{y:[0,4],ticks:[0,1,2,3,4],fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(3);},yTitle:'平均 CRPS %',gname:'信噪比',
    series:ms.map(function(m){return {name:{L:'L 后验',V:'V 方差头',C:'C 扩散模型',B:'B 引导先验（50 个 cube）'}[m],c:MC[m]};}),
    groups:T2S.map(function(x){const s=set('device1','sequential',x);return {label:x+' dB',values:ms.map(function(m){return met(s,m,'crps');}),tags:ms.map(function(m){return m==='B'?bStatusTxt(s)||'待定':null;})};})});};
CH.t2ause=function(h){const ms=['L','V','C','B'];
  bars(h,{w:520,minW:380,h:300,y:[0,0.5],ticks:[0,0.1,0.2,0.3,0.4,0.5],fmt:function(v){return v.toFixed(1);},vfmt:function(v){return v.toFixed(3);},yTitle:'AUSE',gname:'信噪比',
    series:ms.map(function(m){return {name:m==='B'?'B（50 个 cube）':m,c:MC[m]};}),
    groups:T2S.map(function(x){const s=set('device1','sequential',x);return {label:x+' dB',values:ms.map(function(m){return met(s,m,'ause');})};})});};
CH.t2cov=function(h){const ms=['L','V','C','B'];const W=520,H=300,m={l:46,r:12,t:16,b:50};const svg=mk(h,W,H,380);const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const Y=function(v){return m.t+ph-(v-0)/(1.0-0)*ph;};
  el('rect',{x:m.l,y:Y(0.95),width:pw,height:Y(0.85)-Y(0.95),style:'fill:var(--good-bg)'},svg);
  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,t.toFixed(1),'tick','end');});
  line(svg,m.l,Y(0.9),m.l+pw,Y(0.9),'refacc');line(svg,m.l,Y(0.5),m.l+pw,Y(0.5),'refacc');
  tx(svg,4,12,'覆盖率','axt');
  const slots=[];T2S.forEach(function(x){ms.forEach(function(mm){slots.push([x,mm]);});});const sw=pw/slots.length;
  T2S.forEach(function(x,gi){tx(svg,m.l+sw*(gi*ms.length+ms.length/2),H-m.b+36,x+' dB','bs','middle');if(gi)line(svg,m.l+sw*gi*ms.length,m.t,m.l+sw*gi*ms.length,m.t+ph,'axis');});
  const rows=[];
  slots.forEach(function(q,i){const s=set('device1','sequential',q[0]);const cx=m.l+sw*(i+0.5);tx(svg,cx,H-m.b+18,q[1],'tick','middle');
    const c50=met(s,q[1],'cov50'),c90=met(s,q[1],'cov90');
    if(c90==null){tx(svg,cx,Y(0.5),'待定','sm','middle').setAttribute('transform','rotate(-90 '+cx+' '+Y(0.5)+')');rows.push([q[0],q[1],'待定','待定']);return;}
    const a=el('circle',{cx:cx,cy:Y(c90),r:5.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    const b=el('rect',{x:cx-4.5,y:Y(c50)-4.5,width:9,height:9,rx:1.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    [a,b].forEach(function(n){hover(n,function(){return '<div><b style="margin:0;padding:0">'+q[1]+' · '+q[0]+' dB</b></div><div class="sub">cov50 '+c50.toFixed(3)+' · cov90 '+c90.toFixed(3)+'</div>';});});
    rows.push([q[0],q[1],c50.toFixed(3),c90.toFixed(3)]);});
  dataTable(h,['信噪比','方法（B 为 50 个 cube）','cov50','cov90'],rows);};
CH.e2=function(h){const ms=['L','D','C','B'];
  function draw(x){clear(h);const prevLg=h.parentNode.querySelector('.legend');if(prevLg)prevLg.remove();
    bars(h,{y:[0,25],ticks:[0,5,10,15,20,25],fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(2);},yTitle:'RMSE %（'+x+' dB）',gname:'传感器',
      series:ms.map(function(m){return {name:{L:'L，MMSE',D:'D，回归器',C:'C，32 样本均值',B:'B，32 样本均值（50 个 cube）'}[m],c:MC[m]};}),
      groups:['device1','device2','ph5'].map(function(dev){const s=set(dev,'sequential',x);return {label:DEVN[dev]+(dev==='device1'?'（训练所用）':''),values:ms.map(function(m){return rmse(s,m);}),tags:ms.map(function(m){return m==='B'?(bStatusTxt(s)||'待定'):null;})};})});}
  draw('30');seg('e2',draw);};
CH.e3rmse=function(h){const snrs=[20,30,40,60];const ms=['L','L_demosaic','D','C','C1','B'];
  legendInto('e3',ms.map(function(m){const has=snrs.some(function(x){return rmse(set('device1','snapshot',x),m)!=null;});return {n:(m==='L'?'L，逐像素精确后验':MN[m])+(m==='B'&&!has?' — 待定':''),c:MC[m],dash:m==='C1'};}));
  const ser=ms.map(function(m){return {name:m==='L'?'L 精确':MN[m],c:MC[m],dash:m==='C1',label:{L:'L 精确',L_demosaic:'L 去马赛克',D:'D',C:'C',B:'B'}[m],ldy:{C:-2,L_demosaic:7,B:18}[m],data:snrs.map(function(x){const v=rmse(set('device1','snapshot',x),m);return v==null?null:[x,v];}).filter(Boolean)};});
  const t=D.facts.Ldem_target_30;
  xy(h,{log:true,x:[9,62],y:[2,16],xTicks:[20,30,40,60],xFmt:function(v){return v===60?'60（噪声底）':String(v);},yTicks:[2,3,5,8,12,16],yTitle:'RMSE %（对数）',xTitle:'信噪比（dB）',labelAt:'first',mr:20,ml:56,series:ser,tipX:function(v){return v+' dB';},
    refs:[{y:t,x0:26,x1:34,lx:35,label:'T4 目标 '+fx(t,2),cls:'ref'}],
    table:{head:['设置','L 精确','L 去马赛克','D','C 32 样本均值','C1','C 单样本','B（50）'],rows:[['device1',60],['device1',40],['device1',30],['device1',20],['device2',30]].map(function(q){const x=q[1];const s=set(q[0],'snapshot',x);if(!s)return [q[0]+' '+x,'—','—','—','—','—','—','待定'];return [(q[0]==='device2'?'E4 设备 2，':'E3，')+x+' dB',fx(rmse(s,'L'),3),fx(rmse(s,'L_demosaic'),3),fx(rmse(s,'D'),3),fx(rmse(s,'C'),3),fx(rmse(s,'C1'),3),fx(met(s,'C','single_rmse_pct'),3),bOk(s)?fx(rmse(s,'B'),3):bStatusTxt(s)];})}});};
CH.bprobe=function(h){const P=D.b_prior_probe;const cs={blind:['--ink-2',true,false,'7 4'],vpred:['--s2',false,false],g0:['--muted',false,false],tt:['--muted',true,false,'2 3'],x0pilot:['--accent',false,false],x0:['--s4',false,false]};
  legendInto('probe',P.series.map(function(s){return {n:(PN[s.key]||s.name),c:cs[s.key][0],dash:cs[s.key][1],faint:cs[s.key][2]};}).concat([{n:'均值 cube 预测器（细虚线）',c:'--muted',dash:true}]));
  xy(h,{log:true,x:[0,920],y:[2,32],xTicks:[25,100,200,300,400,500,700,900],yTicks:[2,3,5,10,20,30],yTitle:'x̂₀ 误差 %（对数）',xTitle:'扩散时间步 t',mr:20,
    series:P.series.map(function(s){return {name:(PN[s.key]||s.name),c:cs[s.key][0],dash:cs[s.key][1],da:cs[s.key][3],faint:cs[s.key][2],thin:s.key==='g0'||s.key==='tt',data:P.t.map(function(t,i){return [t,s.err[i]];})};}),
    refs:[{y:P.mean_cube,label:'均值 cube '+fx(P.mean_cube,2),lx:520,anchor:'start',dy:-6}],tipX:function(v){return 't = '+v;},tipY:function(v){return v.toFixed(1)+' %';},
    table:{head:['t'].concat(P.series.map(function(s){return (PN[s.key]||s.name);})),rows:P.t.map(function(t,i){return [t].concat(P.series.map(function(s){return s.err[i].toFixed(1);}));})}});};

/* ---------- tables ---------- */
const T={};
T.e3unc=function(h){const ms=['L','L_demosaic','V','C','B'];const nm={L:'L 精确',L_demosaic:'L 去马赛克',V:'V',C:'C',B:'B（50）'};
  const rowsK=[['device1',60],['device1',40],['device1',30],['device1',20],['device2',30]];
  tbl(h,['设置'].concat(ms.map(function(m){return nm[m];})),rowsK.map(function(q){const s=set(q[0],'snapshot',q[1]);return [(q[0]==='device2'?'E4 设备 2 · ':'E3 · ')+q[1]+' dB'].concat(ms.map(function(m){if(!s)return pend();const c=met(s,m,'crps');if(c==null)return pend(bStatusTxt(s)||'待定');return num(fx(c,2)+'<br>'+fx(met(s,m,'cov50'),2)+'/'+fx(met(s,m,'cov90'),2)+'<br>'+fx(met(s,m,'ause'),2));}));}),function(i){return i===2;});
  const p=document.createElement('p');p.style.fontSize='14px';p.style.color='var(--muted)';p.textContent='每格依次为：CRPS %、cov50 / cov90、AUSE。高亮的 30 dB 行是 T4 的设置；E4 仅报告。';h.appendChild(p);};
T.toy=function(h){const t=D.toy;if(!t){h.innerHTML=pend('未找到玩具问题记录');return;}const g=t.gate;
  const inb=function(r){return r.rmse_excess<=g.rmse_excess&&Math.abs(r.var_ratio-1)<=g.var_tol&&r.cov90>=g.cov90[0]&&r.cov90<=g.cov90[1]&&r.cov50>=g.cov50[0]&&r.cov50<=g.cov50[1];};
  const rk=function(r){return (r.C_kind==='isotropic'?0:2)+(r.method==='pigdm'?0:1);};const rows=t.rows.slice().sort(function(a,b){return rk(a)-rk(b)||(a.mode>b.mode)-(a.mode<b.mode)||a.n_steps-b.n_steps;});
  tbl(h,['先验','方法','模式','步数','RMSE 超出量','方差比','cov50','cov90','CRPS 比','玩具问题门限'],rows.map(function(r){return [r.C_kind==='isotropic'?'各向同性 N(0, I)':'HFD 协方差（仅报告）',r.method==='pigdm'?'ΠGDM':'DPS',(MODEZ[r.mode]||r.mode),num(r.n_steps),num(fx(r.rmse_excess,4)),num(fx(r.var_ratio,3)),num(fx(r.cov50,3)),num(fx(r.cov90,3)),num(fx(r.crps_ratio,4)),r.C_kind==='isotropic'&&r.method==='pigdm'?pill(inb(r),inb(r)?'达标':'未达标'):'—'];}),function(i){const r=rows[i];return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});};
function bRow(label,s,b,exp){if(!b||b.status!=='final'&&b.status!=='archived'&&b.status!=='stale'){return [label,b&&b.status==='dropped'?pend('已撤销'):pend(),'','','','','',''];}
  const mm=b.means,sub=b.on_subset,p=b.paired;const ref=s.mode==='snapshot'?'L_demosaic':'D';const pr=p['rmse_vs_'+ref];const keyRef=ref==='D'?'rmse_pct':'mean_rmse_pct';
  return [label+(b.status==='final'?' <span class="pill nt">x₀</span>':' <span class="pill nt">v 预测</span>'),num(fx(mm.mean_rmse_pct,2)+' / '+fx(mm.single_rmse_pct,1)),num(fx(mm.crps,2)),num(fx(mm.cov50,2)+' / '+fx(mm.cov90,2)),num(sg(mm.single_hf,1)),
    num(s.mode==='snapshot'?((sub.L?fx(sub.L.mean_rmse_pct,2):'—')+' / '+(sub.L_demosaic?fx(sub.L_demosaic.mean_rmse_pct,2):'—')+' / '+(sub.D?fx(sub.D.rmse_pct,2):'—')+' / '+(sub.C?fx(sub.C.mean_rmse_pct,2):'—')):((sub.L?fx(sub.L.mean_rmse_pct,2):'—')+' / '+(sub.D?fx(sub.D.rmse_pct,2):'—')+' / '+(sub.C?fx(sub.C.mean_rmse_pct,2):'—'))),
    num(pr?sg(pr.diff,2)+' ± '+fx(pr.se_cluster,2)+' ('+fx(pr.ratio,2)+'×)':'—'),num(fx(b.hours,2))];}
T.brec=function(h){const rows=[];
  [['E1','device1','sequential',[60,50,40,30,20]],['E2','device2','sequential',[40,30,20]],['E3','device1','snapshot',[60,40,30,20]],['E4','device2','snapshot',[30]]].forEach(function(e){e[3].forEach(function(x){const s=set(e[1],e[2],x);if(!s)return;const lab=e[0]+' · '+DEVN[e[1]]+(e[2]==='snapshot'?' 快照':'')+' · '+x+' dB';
    if(e[0]==='E1'&&D.b_vpred[String(x)])rows.push(bRow(lab,s,D.b_vpred[String(x)],e[0]));rows.push(bRow(lab,s,s.B,e[0]));});});
  tbl(h,['设置 · 先验','B RMSE 32 样本均值 / 单样本（%）','B CRPS','B cov50 / cov90','B 单样本 log HF','同一批 cube 上的 L / D / C（快照：L 精确 / L 去马赛克 / D / C）','B − D（快照：− L 去马赛克）± 聚类标准误（比值）','GPU 小时'],rows);};
function clauseCellT1(c,limit){if(!c)return pend();return {v:pill(c.passed)+' <span class="num">比值 '+fx(c.ratio,3)+' · 2 倍标准误上界 '+fx(c.ratio_2se,3)+'（上限 '+fx(limit,2)+'）</span>',cls:''};}
function previewT1(p,limit){if(!p)return null;return {v:prev(p.passed)+' <span class="num">比值 '+fx(p.ratio,3)+' · 2 倍标准误上界 '+fx(p.ratio_2se,3)+'（上限 '+fx(limit,2)+'，n '+p.n+'）</span>',cls:''};}
function bClause(fromVerdict,fallback){if(V.b_valid&&fromVerdict)return fromVerdict;return fallback||{v:pend(),cls:''};}
T.clauses=function(h){const th=V.thresholds||{};const rows=[];
  [60,50,40,30,20].forEach(function(x){const c=V.T1&&V.T1.C?V.T1.C[vk(x)]:null;const vb=V.T1&&V.T1.B?V.T1.B[vk(x)]:null;const s=set('device1','sequential',x);
    rows.push(['<b>T1</b> · '+x+' dB','32 样本均值 RMSE ≤ '+fx(th.T1_RATIO,2)+' × D（2 倍聚类标准误）',clauseCellT1(c,th.T1_RATIO),bClause(vb&&clauseCellT1(vb,th.T1_RATIO),bOk(s)?previewT1(s.B.paired.rmse_vs_D,th.T1_RATIO):null)]);});
  T2S.forEach(function(x){const c=V.T2&&V.T2.C?V.T2.C[vk(x)]:null;const vb=V.T2&&V.T2.B?V.T2.B[vk(x)]:null;const s=set('device1','sequential',x);
    const cell=function(c){if(!c)return pend();return {v:pill(c.passed)+' <span class="num">CRPS − L '+sg(c.crps_vs_L.mean,3)+' ± '+fx(c.crps_vs_L.se_cluster,3)+' · − V '+sg(c.crps_vs_V.mean,3)+' ± '+fx(c.crps_vs_V.se_cluster,3)+' · cov90 '+fx(c.cov90,3)+(c.cov90_passed?'':'（超出区间）')+'</span>',cls:''};};
    let pv=null;if(bOk(s)){const p=s.B.paired,cv=s.B.means.cov90;const ok=p.crps_vs_L.diff+2*p.crps_vs_L.se_cluster<0&&p.crps_vs_V.diff+2*p.crps_vs_V.se_cluster<0&&cv>=th.T2_COV90[0]&&cv<=th.T2_COV90[1];
      pv={v:prev(ok)+' <span class="num">CRPS − L '+sg(p.crps_vs_L.diff,3)+' ± '+fx(p.crps_vs_L.se_cluster,3)+' · − V '+sg(p.crps_vs_V.diff,3)+' ± '+fx(p.crps_vs_V.se_cluster,3)+' · cov90 '+fx(cv,3)+'</span>',cls:''};}
    rows.push(['<b>T2</b> · '+x+' dB','CRPS 低于 L 与 V（2 倍聚类标准误），cov90 ∈ ['+th.T2_COV90.join(', ')+']',cell(c),bClause(vb&&cell(vb),pv)]);});
  const t3=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">相对 L 的比值：设备 1 '+fx(c.ratio_dev1,3)+' · 设备 2 '+fx(c.ratio_dev2,3)+'（上限 '+fx(th.T3_RATIO*c.ratio_dev1,3)+'）· PH5 '+(c.ph5==null?'未运行':fx(c.ph5,3)+'（仅报告）')+'</span>',cls:''};};
  let t3p=null;const s1=set('device1','sequential',30),s2=set('device2','sequential',30);if(bOk(s1)&&bOk(s2)){const r1=s1.B.paired.rmse_vs_L.ratio,r2=s2.B.paired.rmse_vs_L.ratio;t3p={v:prev(r2<=th.T3_RATIO*r1)+' <span class="num">相对 L 的比值：设备 1 '+fx(r1,3)+' · 设备 2 '+fx(r2,3)+'（上限 '+fx(th.T3_RATIO*r1,3)+'）</span>',cls:''};}
  rows.push(['<b>T3</b> · 30 dB','设备 2 上的（B 或 C）/ L ≤ '+fx(th.T3_RATIO,2)+' × 设备 1 上的同一比值',t3(V.T3&&V.T3.C)||pend(),bClause(t3(V.T3&&V.T3.B),t3p)]);
  const t4=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">比值 '+fx(c.ratio,3)+' · 2 倍标准误上界 '+fx(c.ratio_2se,3)+'（上限 '+fx(th.T4_RATIO,2)+'）'+(c.D?'；D：'+(c.D.passed?'通过':'未通过')+'，比值 '+fx(c.D.ratio,3):'')+'</span>',cls:''};};
  const s4=set('device1','snapshot',30);rows.push(['<b>T4</b> · 快照 30 dB','RMSE ≤ '+fx(th.T4_RATIO,2)+' × 先去马赛克再 MMSE（2 倍聚类标准误）',t4(V.T4&&V.T4.C)||pend(),bClause(t4(V.T4&&V.T4.B),bOk(s4)?previewT1(s4.B.paired.rmse_vs_L_demosaic,th.T4_RATIO):null)]);
  const sc=V.sanity||{};const fp=sc.floor_pass||{};const mo=sc.monotone||{};
  const seqC=E1S.map(function(x){return fx(rmse(set('device1','sequential',x),'C'),3);}).join(' / '),seqB=E1S.map(function(x){return fx(rmse(set('device1','sequential',x),'B'),3);}).join(' / ');
  const sanCell=function(line,seq){const ok=mo[line]&&fp[line];return {v:pill(ok)+' <span class="num">单调 '+(mo[line]?'是':'否')+'（60→20 dB：'+seq+' %）· 噪声底 '+(fp[line]?'通过':'未通过')+'（'+fx(rmse(set('device1','sequential',60),line),3)+' 对 '+fx(sc.floor_reference_rmse_pct,3)+' %）</span>'+(line==='B'&&!mo.B?'<br><span style="font-size:13px;color:var(--muted)">受先验限制：在噪声范围内持平，40 到 30 dB 间上升 +'+fx(F0.B_e1_rmse['30']-F0.B_e1_rmse['40'],2)+'</span>':''),cls:''};};
  rows.push(['<b>合理性检查</b>','RMSE 随信噪比升高而不增；60 dB 时 ≤ '+fx(sc.floor_reference_rmse_pct,3)+' % + 2 倍标准误',sanCell('C',seqC),sanCell('B',seqB)]);
  const pc=function(line){const p=passes(line);return num('<b>5 项中 '+p.length+' 项</b>（'+(joinAnd(p)||'无')+'）');};
  rows.push(['<b>通过条款数</b>','T1 计一次（须五个信噪比全部通过），T2 每个信噪比各计一次，另加 T3、T4：最多 5 项；平局 → 取 20 dB 处配对 CRPS − V 较低者',pc('C'),V.b_valid?pc('B'):bInvalid()]);
  const mlB=function(){if(!V.b_valid)return bInvalid();return num(V.main_line==='B'?'<b>B</b>（决胜规则：'+tieInfo(3).short+'）':'—');};
  rows.push(['<b>主线</b>','',num(!V.b_valid?pend():(V.main_line==='C'?'<b>C</b>':'—')),mlB()]);
  tbl(h,['条款 · 设置','规则','C 线（200 个 cube）','B 线（50 个 cube）'],rows,function(i){return i>=rows.length-2;});};
T.runs=function(h){const rows=D.runs.map(function(r){return [r.run+(r.failed?' <span class="pill no">失败的先验</span>':''),(ROLE[r.role]||r.role),(r.prediction_type||'v')+' · '+(r.loss_type||'')+(r.snr_gamma&&r.prediction_type!=='direct'?' · min-SNR γ '+r.snr_gamma:''),num(r.epochs+' / '+r.planned),num(fx(r.sec_per_epoch,0)),num(fx(r.hours,2)),num(String(r.selected)),num(fx(r.val_at_sel,5))];});
  D.pilots.forEach(function(p){rows.push([p.run,'先验试验',(p.prediction_type||'v')+' · '+(p.loss_type||'')+(p.snr_gamma!=null?' · min-SNR γ '+p.snr_gamma:''),num(p.epochs+' / '+p.planned),num(fx(p.sec_per_epoch,0)),num(fx(p.hours,2)),num('—'),num('—')]);});
  tbl(h,['运行','角色','参数化 · 损失','轮数','秒 / 轮','GPU 小时','选定轮次','该轮的监控损失'],rows);};

/* ---------- slots ---------- */
const SL={};
function bList(keys){return keys.map(function(k){const s=S[k];return (s.exp+' '+DEVN[s.device]+(s.mode==='snapshot'?' 快照':'')+' '+s.snr+' dB');}).join(', ');}
SL.normcheck=function(h){const sc=D.stage0.shape_check;h.textContent='归一化检查'+(D.stage0.normalisation_check_passed?'通过':'未通过')+'：使用原始（未归一化）响应时，曲线形状保持在 10 % 规则以内（最大偏差：顺序模式 '+fx(100*sc.sequential.max_dev,1)+' %，快照模式 '+fx(100*sc.snapshot.max_dev,1)+' %），因此归一化后的 R 就是论文所用的传感器，原始曲线（虚线）放入附录。';};
SL.angles=function(h){const a=D.stage0.principal_angles;h.innerHTML='<b>传感器列空间之间的主角。</b>设备 1 与设备 2 共享一个 '+F0.ang_d1d2_shared+' 维子空间：'+a.device1_device2.n+' 个主角中只有 '+F0.ang_d1d2_nonzero+' 个非零（'+fx(F0.ang_d1d2_max,1)+'°）。因此迁移测试考察的是两个传感器如何对各方向加权，而不是子空间是否不同。PH5 的响应秩为 '+F0.ph5_rank+'，它与设备 1 列空间之间的每个主角都不超过 '+fx(F0.ang_d1ph5_max,1)+'°，所以 PH5 测量的是设备 1 所测内容中的 '+F0.ph5_rank+' 维部分，而条件数差得多。';};
SL.t1marks=function(h){const th=V.thresholds||{};h.innerHTML='<span><b>T1</b>（上限 '+fx(th.T1_RATIO,2)+'）：</span>'+[60,50,40,30,20].map(function(x){const c=V.T1&&V.T1.C&&V.T1.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T1.B&&V.T1.B[vk(x)];
    let b;if(vb)b=pill(vb.passed,'B '+fx(vb.ratio,2));else if(bOk(s))b='<span class="pill wn">B '+fx(s.B.paired.rmse_vs_D.ratio,2)+'（预览）</span>';else b=pend('B 待定');
    return '<span>'+x+' dB '+(c?pill(c.passed,'C '+fx(c.ratio,2)):pend('C'))+' '+b+'</span>';}).join('');};
SL.t2marks=function(h){h.innerHTML=T2S.map(function(x){const c=V.T2&&V.T2.C&&V.T2.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T2.B&&V.T2.B[vk(x)];
  return '<span><b>T2</b> '+x+' dB '+(c?pill(c.passed,'C '+(c.passed?'通过':'未通过')):pend('C'))+' '+(vb?pill(vb.passed,'B '+(vb.passed?'通过':'未通过')):(bOk(s)?'<span class="pill wn">B 预览</span>':pend('B 待定')))+'</span>';}).join('');};
SL.d2D=function(h){h.textContent='在 40 / 30 / 20 dB 下比 MMSE 低 '+[40,30,20].map(function(x){return fx(-F0['device2_D_vs_L_'+x],0)+' %';}).join(' / ')+'（在设备 1 上为 '+[40,30,20].map(function(x){return fx(F0['D_below_L_'+x],0)+' %';}).join(' / ')+'）。';};
SL.toytext=function(h){const t=D.toy;if(!t){h.textContent='构建时未找到玩具问题记录。';return;}const r=t.rows.filter(function(r){return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});
  h.textContent='在 Stage 0 之前，ΠGDM 必须在使用真实设备 1 算子的各向同性玩具问题上复现精确的高斯后验（每格 1 000 次抽样，30 dB）。在 '+t.steps_for_B+' 步时，它在两种模式下都满足全部门限（'+r.map(function(q){return (MODEZ[q.mode]||q.mode)+'模式：平均超出量 '+fx(q.rmse_excess,4)+'，方差比 '+fx(q.var_ratio,3)+'，cov90 '+fx(q.cov90,3);}).join('；')+'），因此 B 按预注册设定以 '+t.steps_for_B+' 步运行。在相关的 HFD 协方差上（仅报告），各向同性近似使 RMSE 增加约 4 %，并在顺序模式下欠离散。'+((passes('B').indexOf('30 dB 下的 T2')<0&&passes('B').indexOf('20 dB 下的 T2')<0)?'B 的覆盖率确实未通过 T2（第 3 节），因此这一近似是首先要检验的解释。':'若 B 的校准条款未通过，这将是首先要检验的解释。');};
/* verdict helpers: the verdict's own counting rule (sd_verdict._counts) */
const E1S=[60,50,40,30,20];
function passes(line){const out=[];const t1=V.T1&&V.T1[line]||{};if(E1S.every(function(x){return t1[vk(x)]&&t1[vk(x)].passed;}))out.push('T1');
  T2S.forEach(function(x){const c=V.T2&&V.T2[line]&&V.T2[line][vk(x)];if(c&&c.passed)out.push(x+' dB 下的 T2');});
  if(V.T3&&V.T3[line]&&V.T3[line].passed)out.push('T3');if(V.T4&&V.T4[line]&&V.T4[line].passed)out.push('T4');return out;}
function t2v(line,x){return V.T2&&V.T2[line]&&V.T2[line][vk(x)];}
/* B's verdict numbers are shown only while the verdict's B clauses are current (V.b_valid), as in the clause table */
function bInvalid(){return pend('B：判定非最新');}
function bInvalidWhy(){return '判定 <code>'+ST.verdict_file+'</code> 读取的 B 记录已不是最新'+(V.b_invalid_reason?'（'+V.b_invalid_reason+'）':'');}
/* the tie-break as the verdict computed it: on the cubes the tied lines share (V.tie_break), else each line's own T2 mean */
function tieInfo(n){const t=V.tie_break,tb=t2v('B',20),tc=t2v('C',20);
  if(t&&t.lines&&t.lines.B&&t.lines.C){const b=t.lines.B,c=t.lines.C;
    return {common:true,winner:t.winner,txt:'在两条路线共有的 '+t.n_common+' 个 cube 上，B 为 '+sg(b.mean_common,n)+' ± '+fx(b.se_cluster_common,n)+'，C 为 '+sg(c.mean_common,n)+' ± '+fx(c.se_cluster_common,n)+'（聚类标准误；在各自的 cube 上，B 为 '+sg(b.mean_own,n)+'，共 '+b.n_own+' 个，C 为 '+sg(c.mean_own,n)+'，共 '+c.n_own+' 个）',
      short:'共有的 '+t.n_common+' 个 cube 上 B '+sg(b.mean_common,n)+' 对 C '+sg(c.mean_common,n)};}
  return {common:false,winner:V.main_line,txt:'B 在其 50 个 cube 上为 '+sg(tb.crps_vs_V.mean,n)+'，C 在 200 个 cube 上为 '+sg(tc.crps_vs_V.mean,n),
    short:sg(tb.crps_vs_V.mean,n)+' 对 C '+sg(tc.crps_vs_V.mean,n)};}
function joinAnd(a){return a.length<2?a.join(''):a.slice(0,-1).join('、')+' 和 '+a[a.length-1];}
function bestSnrs(m){const b=F0.e1_best_common||{};return E1S.filter(function(x){return b[String(x)]===m;}).sort(function(a,b){return b-a;});}
function rng(a){return a.length?(a.length>1?a[0]+'–'+a[a.length-1]:String(a[0]))+' dB':'无';}
SL.status=function(h){const p=ST.pending;h.innerHTML=p.length?'<b>构建时记录中尚缺：</b>'+p.join('、')+'。依赖这些记录的单元格显示“待定”。':'';};
SL.outcome=function(h){const cB=passes('B'),cC=passes('C');const tb=t2v('B',20);const ml=V.main_line;
  const cC40=[40,30,20].map(function(x){return met(set('device1','sequential',x),'C','cov90');});
  const bD=bestSnrs('D'),bB=bestSnrs('B'),bC=bestSnrs('C');
  const t2ok=function(line){return T2S.some(function(x){const c=t2v(line,x);return c&&c.passed;});};
  const demo=F0.e3_D_30!=null&&F0.Ldem_30!=null&&F0.e3_D_30<F0.Ldem_30;
  const ph5None=F0.ph5_D_worse_min!=null&&F0.ph5_C_ratio_min!=null&&F0.ph5_D_worse_min>0&&F0.ph5_C_ratio_min>1;
  const dItem='<li><b>回归器 D：</b>高信噪比下的最佳点估计（在共有的 cube 上为 '+rng(bD)+'），也是唯一能去马赛克的方法：快照模式 30 dB 下为 '+fx(F0.e3_D_30,2)+' %，而先去马赛克再 MMSE 为 '+fx(F0.Ldem_30,2)+' %。</li>';
  const cItem='<li><b>C，条件扩散模型：</b>在其训练所用传感器上覆盖率接近名义值（40–20 dB 下 cov90 为 '+fx(Math.min.apply(null,cC40),3)+'–'+fx(Math.max.apply(null,cC40),3)+'），并且在 ≤ 40 dB 时，是覆盖率接近名义值的方法中 CRPS 最低者（比 L 和 V 低 '+fx(F0.C_crps_gain_lo_min,0)+'–'+fx(F0.C_crps_gain_lo_max,0)+' %）。它的均值始终比不上回归器。</li>';
  const dOpen='当测量信息充分时——高信噪比'+(bD.length?'（'+rng(bD)+'）':'')+'，或在网络训练所用传感器上的快照模式——单次前向的回归器 D 是最佳点估计'+(demo?'，也是唯一能去马赛克的方法':'')+'。';
  const cOpen=bC.length?'条件扩散模型 C 在 '+rng(bC)+' 下是最佳点估计。':(t2ok('C')&&!t2ok('B')?'条件扩散模型 C 在自身传感器上给出校准良好的方法中最好的不确定性（唯一通过不确定性条款 T2 的路线），但均值从未最好。':'条件扩散模型 C 的均值从未最好；其不确定性的比较见第 3 节。');
  const ph5Txt=ph5None?'在设备 1 上学到的东西都无法迁移到 PH5（B 未在 PH5 上运行）。':'PH5 上的迁移结果见第 4 节。';
  if(!V.b_valid){h.innerHTML='<p><b>没有哪一种方法在所有条件下都最好。</b>'+dOpen+cOpen+ph5Txt+'</p><ul>'+dItem+cItem+'</ul><p><b>预注册判定：</b>'+bInvalid()+' C 在 5 项中通过 '+cC.length+' 项（'+(joinAnd(cC)||'无')+'）；'+bInvalidWhy()+'，因此 B 的条款、通过数和主线需等待重新生成的判定。</p>';return;}
  const r30=V.T1.B&&V.T1.B['30.0'],r20=V.T1.B&&V.T1.B['20.0'];const c30=t2v('B',30);const ti=tieInfo(2);
  const d2=set('device2','sequential',30);const d2ok=bOk(d2)&&d2.B.on_subset&&d2.B.on_subset.D;
  const bTrans=d2ok&&d2.B.means.mean_rmse_pct<d2.B.on_subset.D.rmse_pct;
  const bOpen=bB.length?'当噪声占主导（'+rng(bB)+'）时，引导先验 B 成为最佳点估计'+(bTrans?'，而且是唯一不经重新训练就能迁移到第二台传感器的学习方法':'')+'。':'在共有的 cube 上，引导先验 B 在任何信噪比下都不是最佳点估计。';
  const tie=cB.length===cC.length?'平局按 20 dB 处配对 CRPS − CRPS<sub>V</sub> 较低者决出（'+ti.txt+'），因此主线是 <b>'+(ti.winner||ml||'—')+'</b>':'因此主线是 <b>'+(ml||'—')+'</b>';
  h.innerHTML='<p><b>没有哪一种方法在所有条件下都最好。</b>'+dOpen+bOpen+cOpen+ph5Txt+'</p>'+
   '<ul>'+dItem+
   '<li><b>B，引导先验：</b>在 '+rng(bB)+' 下是最佳点估计（在共有的 cube 上，其 32 样本均值在 30 / 20 dB 下分别是回归器的 '+fx(r30&&r30.ratio,3)+' / '+fx(r20&&r20.ratio,3)+' 倍），并能零样本迁移到设备 2（'+(d2ok?'30 dB 下为 '+fx(d2.B.means.mean_rmse_pct,2)+' %，回归器为 '+fx(d2.B.on_subset.D.rmse_pct,2)+' %':pend())+'），但在所有设置下都过度自信。</li>'+
   cItem+'</ul>'+
   '<p><b>预注册判定</b>（<code>'+ST.verdict_file+'</code>）<b>：</b>B 在 5 项条款中通过 '+cB.length+' 项（'+(joinAnd(cB)||'无')+'），C 通过 '+cC.length+' 项（'+(joinAnd(cC)||'无')+'）；'+tie+
   (ml==='B'?'——它胜在均值的准确性，而非校准：其样本过度自信（30 / 20 dB 下 cov90 为 '+fx(c30&&c30.cov90,3)+' / '+fx(tb&&tb.cov90,3)+'，名义值为 0.90）':'')+'。两条路线的完整结果均在下文报告。</p>';};
SL.kpiB=function(h){if(!V.b_valid){h.innerHTML=bInvalid();return;}const r=V.T1.B&&V.T1.B['20.0'];h.textContent=r?fx(r.ratio,3)+' ×':'—';};
SL.kpiBsub=function(h){if(!V.b_valid){h.textContent='判定中 B 的条款不是最新的';return;}const r=V.T1.B&&V.T1.B['30.0'];h.textContent=r?('30 dB 下为 '+fx(r.ratio,3)+' ×；T1 在 30 与 20 dB 通过'):'';};
SL.countBC=function(h){if(!V.b_valid){h.innerHTML='B '+pend('非最新')+' · C '+passes('C').length;return;}h.textContent='B '+passes('B').length+' · C '+passes('C').length;};
SL.countsub=function(h){if(!V.b_valid){h.textContent='C：'+(joinAnd(passes('C'))||'无')+' · B 与主线等待重新生成的判定';return;}
  h.textContent='B：'+(joinAnd(passes('B'))||'无')+' · C：'+(joinAnd(passes('C'))||'无')+' · 按决胜规则主线为 '+(V.main_line||'—');};
SL.tbRuling=function(h){const t=V.tie_break;if(!t||!V.b_valid)return;   /* keeps the typed fallback sentence */
  h.textContent='决胜规则在两份记录共有的 '+t.n_common+' 个 cube 上比较 B 与 C（20 dB 处配对 CRPS − CRPS_V，聚类标准误）；判定文件中同时保留了各路线在自身 cube 上的值。';};
SL.e1lead=function(h){const bD=bestSnrs('D'),bB=bestSnrs('B');h.textContent='这是论文的主图。在 B 与其他方法共有的 cube 上，回归器 D 在 '+rng(bD)+' 下最准确，B 的 32 样本均值在 '+rng(bB)+' 下最准确。B 的误差几乎不随信噪比变化，因此它受限于先验而非噪声。条款 T1 检验较好的扩散路线的 32 样本均值 RMSE 是否在回归器的 5 % 以内（按 2 倍聚类标准误）。';};
SL.dbest=function(h){const c=V.T1.C;const bB=bestSnrs('B');
  h.innerHTML='<b>没有哪一种方法在所有条件下都是最佳点估计。</b>D 在 '+rng(bestSnrs('D'))+' 下最准确，并在 20 dB 时把 MMSE 的误差减半（'+fx(F0.D_20,2)+' 对 '+fx(F0.L_20,2)+' %）。B 在其 50 个 cube 上于 '+rng(bB)+' 下最准确（'+bB.map(function(x){return x+' dB 下为 D 的 '+fx(V.T1.B[vk(x)].ratio,3)+' 倍';}).join('，')+'）。C 的 32 样本均值从未胜过 D。规格说明预测 C 会在每个信噪比下通过 T1，结果它在每个信噪比下都未通过：高信噪比端落后约 '+fx(100*(c['60.0'].ratio-1),0)+' %，30 dB 及以下落后 '+fx(100*(c['30.0'].ratio-1),0)+'–'+fx(100*(c['20.0'].ratio-1),0)+' %。';};
SL.t1text=function(h){const c=V.T1.C,b=V.T1.B;const bp=E1S.filter(function(x){return b[vk(x)]&&b[vk(x)].passed;});
  h.innerHTML='C 在全部五个信噪比下均未通过 T1；它与 D 的比值在 60 / 50 / 40 / 30 / 20 dB 下依次为 '+E1S.map(function(x){return fx(c[vk(x)].ratio,3);}).join(' / ')+'。在高信噪比端，差距的一部分来自 32 样本均值的 1/K 蒙特卡洛项，单次前向的 C1 更接近（60 dB 下为 '+fx(F0.C1_60,3)+' %，D 为 '+fx(F0.D_60,3)+' %）。B 在 '+(bp.join(' 和 ')||'无')+' dB 通过，但 T1 对全部五个信噪比统一计一次，因此两条路线都未通过 T1。B 在 30 dB 处的 2 倍标准误上界为 '+fx(b['30.0'].ratio_2se,3)+'，余量很窄，且只来自一个训练种子。';};
SL.e1crpscap=function(h){const b30=t2v('B',30),b20=t2v('B',20);
  h.innerHTML='在 60 dB 下，三种 200-cube 方法不相上下。从 40 dB 起，C 的 CRPS 比 MMSE 后验和方差头都低 '+fx(F0.C_crps_gain_lo_min,0)+'–'+fx(F0.C_crps_gain_lo_max,0)+' %。这与第 2a 阶段相反，当时方差头在 CRPS 上胜出。B 在其 50 个 cube 上于 30 与 20 dB 取得最低 CRPS（相对 L 的比值为 '+fx(b30.crps_vs_L.ratio_of_means,2)+' 和 '+fx(b20.crps_vs_L.ratio_of_means,2)+'）。这来自其准确的均值，而非其离散程度（第 3 节）。';};
SL.e1B=function(h){const m=F0.B_e1_rmse;const mono=V.sanity&&V.sanity.monotone;
  h.innerHTML='<b>B 几乎不随信噪比变化。</b>其 32 样本均值在 60 / 50 / 40 / 30 / 20 dB 下为 '+E1S.map(function(x){return fx(m[String(x)],3);}).join(' / ')+' %。在噪声底处它远高于 MMSE（未通过噪声底合理性检查），其误差从 40 到 30 dB 甚至上升了 '+fx(m['30']-m['40'],2)+'，未通过预注册的单调性检查'+(mono&&mono.B===false?'':'（判定：通过）')+'。决定其误差的是先验，而不是测量噪声。这就是它只在噪声主导其他方法误差的地方胜出的原因。';};
SL.t2text=function(h){const a=t2v('C',30),b=t2v('C',20),ba=t2v('B',30),bb=t2v('B',20);const band=V.thresholds.T2_COV90;
  h.innerHTML='L、V 与 C 基于 200 个门控 cube，B 基于其 50 个 cube（在共有文件上配对）。<b>C</b> 在 30 dB '+(a.passed?'通过':'未通过')+' T2：CRPS − L = '+sg(a.crps_vs_L.mean,3)+' ± '+fx(a.crps_vs_L.se_cluster,3)+'，− V = '+sg(a.crps_vs_V.mean,3)+' ± '+fx(a.crps_vs_V.se_cluster,3)+'（聚类），cov90 '+fx(a.cov90,3)+'。在 20 dB 下两项差异都很明显（'+sg(b.crps_vs_L.mean,3)+' ± '+fx(b.crps_vs_L.se_cluster,3)+' 与 '+sg(b.crps_vs_V.mean,3)+' ± '+fx(b.crps_vs_V.se_cluster,3)+'），但 cov90 = '+fx(b.cov90,3)+(b.cov90_passed?'，位于区间内':'，比区间上限高 '+fx(b.cov90-band[1],3))+'，因此它'+(b.passed?'通过':'仅因覆盖率未通过')+'。'+
   '<b>B</b> 在两个信噪比下 CRPS 都最低（− V = '+sg(ba.crps_vs_V.mean,2)+' ± '+fx(ba.crps_vs_V.se_cluster,2)+' 与 '+sg(bb.crps_vs_V.mean,2)+' ± '+fx(bb.crps_vs_V.se_cluster,2)+'）。但它在两处都因覆盖率未通过 T2（cov90 '+fx(ba.cov90,3)+' / '+fx(bb.cov90,3)+'）。CRPS 对准确均值的奖励大于对过窄离散的惩罚，因此均值良好但过度自信的采样器可以得到最低的 CRPS。';};
SL.covtext=function(h){const c20=met(set('device1','sequential',20),'C','cov50'),c30=met(set('device1','sequential',30),'C','cov50');const B30=set('device1','sequential',30).B.means,B20=set('device1','sequential',20).B.means;
  h.textContent='C 略微过离散，且信噪比越低越明显（30 dB 下 cov50 为 '+fx(c30,3)+'，20 dB 下为 '+fx(c20,3)+'，名义值 0.50）。方差头是四者中校准最好的。B 严重过度自信：30 / 20 dB 下其 50 % 区间覆盖 '+fx(B30.cov50,2)+' / '+fx(B20.cov50,2)+'，90 % 区间覆盖 '+fx(B30.cov90,2)+' / '+fx(B20.cov90,2)+'。';};
SL.t3text=function(h){const c=V.T3.C,b=V.T3.B,th=V.thresholds;
  h.innerHTML='L、D 与 C 基于 200 个门控 cube，B 基于其 50 个 cube。30 dB 下的 T3：<b>B '+(b&&b.passed?'通过':'未通过')+'</b>。其相对 MMSE 的 RMSE 比值在设备 1 上为 '+fx(b.ratio_dev1,3)+'，在设备 2 上为 '+fx(b.ratio_dev2,3)+'，上限为 '+fx(th.T3_RATIO*b.ratio_dev1,3)+'。<b>C '+(c.passed?'通过':'未通过')+'</b>：其比值为 '+fx(c.ratio_dev1,3)+' 和 '+fx(c.ratio_dev2,3)+'，上限为 '+fx(th.T3_RATIO*c.ratio_dev1,3)+'。在 PH5 上（仅报告），C 的比值为 '+fx(c.ph5,3)+'。';};
SL.ph5V=function(h){h.textContent=[40,30,20].map(function(x){const s=set('ph5','sequential',x);return fx(met(s,'V','crps'),2)+' 对 '+fx(met(s,'L','crps'),2);}).join('；')+'，依次对应 40 / 30 / 20 dB';};
SL.e2B=function(h){const xs=[40,30,20];
  h.innerHTML='<b>设备 2：B 可零样本迁移，并且'+(F0.B_best_dev2_all?'在每个信噪比下都是最准确的方法':'保留了大部分准确度')+'，但仍过度自信。</b>在其 50 个 cube 上，它的得分为：'+xs.map(function(x){const b=set('device2','sequential',x).B;return x+' dB 下 '+fx(b.means.mean_rmse_pct,2)+' %（L '+fx(b.on_subset.L.mean_rmse_pct,2)+'，D '+fx(b.on_subset.D.rmse_pct,2)+'）';}).join('；')+'。其 cov90 为 '+xs.map(function(x){return fx(set('device2','sequential',x).B.means.cov90,2);}).join(' / ')+'。B 从未在任何传感器上训练，因此迁移对它没有代价；其误差由先验决定。';};
SL.t4text=function(h){const c=V.T4.C,b=V.T4.B;
  h.textContent='L 精确、L 去马赛克、D 与 C 基于 200 个门控 cube，B 基于其 50 个 cube。30 dB 处的虚线段是 T4 的目标。两条路线都未通过 T4。C：比值 '+fx(c.ratio,3)+'（2 倍标准误上界 '+fx(c.ratio_2se,3)+'），其 32 样本均值（'+fx(F0.e3_C_30,2)+' %）比先去马赛克再 MMSE（'+fx(F0.Ldem_30,2)+' %）更差。B：比值 '+fx(b.ratio,3)+'（2 倍标准误上界 '+fx(b.ratio_2se,3)+'）。回归器以很大余量通过同一条款，为 '+fx(F0.e3_D_30,2)+' %（比值 '+fx(c.D.ratio,3)+'），并且在每个信噪比下都低于基线。';};
SL.e3B=function(h){const xs=[60,40,30,20].filter(function(x){return bOk(set('device1','snapshot',x));});
  const r=xs.map(function(x){return set('device1','snapshot',x).B.paired.rmse_vs_L_demosaic.ratio;});
  h.innerHTML='<b>B 无法去马赛克。</b>在使用掩膜算子的零样本设定下，它落在去马赛克基线的水平：在其 50 个 cube 上，'+xs.join(' / ')+' dB 下相对先去马赛克再 MMSE 的比值为 '+fx(Math.min.apply(null,r),2)+'–'+fx(Math.max.apply(null,r),2)+'。其离散仍然过窄（cov90 '+xs.map(function(x){return fx(set('device1','snapshot',x).B.means.cov90,2);}).join(' / ')+'）。在完整 cube 上学到的光谱–空间先验无法提供回归器从快照训练 cube 中学到的跨像素插值。';};
SL.e4=function(h){const s=set('device2','snapshot',30);if(!s||!s.methods.L){h.innerHTML='<b>E4（设备 2，快照模式，30 dB，仅报告）：</b>'+pend();return;}
  const bR=(bOk(s)&&s.B.paired&&s.B.paired.rmse_vs_L_demosaic)?s.B.paired.rmse_vs_L_demosaic.ratio:null;const bNear=bR!=null&&bR<1.15;
  const d1=set('device1','snapshot',30);const d1R=(d1&&bOk(d1)&&d1.B.paired&&d1.B.paired.rmse_vs_L_demosaic)?d1.B.paired.rmse_vs_L_demosaic.ratio:null;
  h.innerHTML='<b>E4，设备 2 快照模式，30 dB（仅报告）：</b>'+(F0.e4_D_over_dem>1&&F0.e4_C_over_dem>1?'在设备 1 上训练的条件网络都无法迁移到另一传感器的马赛克'+(bNear?'；由先验引导的 B 可以，并达到经典流程的水平':'')+'。':'')+'先去马赛克再 MMSE 为 '+fx(F0.e4_Ldem,2)+' %（逐像素精确后验为 '+fx(F0.e4_Lexact,2)+' %）。D 为 '+fx(F0.e4_D,2)+' %（基线的 '+fx(F0.e4_D_over_dem,2)+' 倍），C 为 '+fx(F0.e4_C,2)+' %（'+fx(F0.e4_C_over_dem,2)+' 倍），C1 为 '+fx(F0.e4_C1,2)+' %。B：'+(bOk(s)?'在其 50 个 cube 上为 '+fx(s.B.means.mean_rmse_pct,2)+' %（在这些 cube 上相对基线的比值为 '+fx(bR,3)+(d1R!=null?'；在设备 1 上为 '+fx(d1R,3):'')+'），CRPS '+fx(s.B.means.crps,2)+'，cov90 '+fx(s.B.means.cov90,2)+(bNear&&F0.e4_D_over_dem>1?'——它唯一的传感器知识是引导算子，因此它是唯一能迁移到新马赛克的学习方法，但并未胜过经典流程':''):pend())+'。'+(F0.e4_D_over_dem>1?'回归器的去马赛克能力专属于其训练所用的传感器。':'')+'';};
SL.probetext=function(h){const P=D.b_prior_probe;const g=function(k){return P.series.filter(function(s){return s.key===k;})[0].err;};const i299=P.t.indexOf(299);const v=g('vpred'),bl=g('blind'),x0=g('x0'),g0=g('g0'),tt=g('tt');
  const gap=P.t.map(function(t,i){return Math.abs(v[i]-bl[i]);});const lo=P.t.filter(function(t){return t<=50;}).map(function(t){return gap[P.t.indexOf(t)];});const mid=P.t.filter(function(t){return t>=100&&t<=200;}).map(function(t){return gap[P.t.indexOf(t)];});
  const pg=Math.max.apply(null,P.t.filter(function(t){return t<=200;}).map(function(t){const i=P.t.indexOf(t);return Math.max(Math.abs(g0[i]-bl[i]),Math.abs(tt[i]-bl[i]));}));
  const worse=P.t.filter(function(t,i){return v[i]>P.mean_cube;});
  h.textContent='v 预测先验（第 100 轮）跟随输入无关预测器 E[v | t]：两条曲线在 t = 25 / 50 处相差不超过 '+fx(Math.max.apply(null,lo),1)+'，在 t = 100 到 200 间相差 '+fx(Math.min.apply(null,mid),1)+'–'+fx(Math.max.apply(null,mid),1)+' 个百分点（两个试验模型相差不超过 '+fx(pg,1)+'）。在 t = '+worse.join('、')+' 处，它比直接预测均值 cube（'+fx(P.mean_cube,2)+' %）还差。均匀损失加权和时间令牌路径都没有改变这条曲线。采用 x₀ 预测时，低 t 处的目标接近恒等映射，迫使网络读取 x_t。重新训练的先验在 t = '+P.t[0]+' 处误差为 '+fx(x0[0],1)+' %，在 t = 299 处为 '+fx(x0[i299],1)+' %，而此处 v 预测先验为 '+fx(v[i299],1)+' %，输入无关预测器为 '+fx(bl[i299],1)+' %；其误差单调增长，并在每个 t 都低于均值 cube。无引导的 DDIM-50 样本中，v 预测先验有 '+fx(P.saturation_pct.vpred,1)+' % 的元素饱和，x₀ 预测先验为 '+fx(P.saturation_pct.x0,1)+' %。';};
SL.x0rule=function(h){const P=D.b_prior_probe;const g=function(k){return P.series.filter(function(s){return s.key===k;})[0].err;};const v=g('vpred'),p=g('x0pilot');
  const idx=P.t.map(function(t,i){return i;}).filter(function(i){return P.t[i]<=300;});const drops=idx.map(function(i){return 1-p[i]/v[i];});const ok=idx.filter(function(i,j){return drops[j]>=0.3;}).map(function(i){return P.t[i];});const bad=idx.filter(function(i,j){return drops[j]<0.3;});
  h.innerHTML='<b>B 的先验改用 x₀ 预测重新训练（这是一项修订，并非预注册）。</b>此项仅针对先验修订了规格说明 §5.5 中的“v 预测”（第 6 节）。判定规则是在第一个先验失败之后、试验运行之前确定的：试验模型的曲线必须单调，且在 t ≤ 300 处比失败的先验低至少 30 %。它在 t = '+ok[0]+'–'+ok[ok.length-1]+' 处满足这一条件'+(bad.length?'；在 t = '+bad.map(function(i){return P.t[i]+' 处降幅为 '+fx(100*(1-p[i]/v[i]),0)+' %';}).join('，'):'')+'。C 仍为 v 预测。评测框架根据检查点自身的预测类型构建 B 的采样器。因此 B 的判定评分的是一条经过修订的路线。';};
SL.bnow=function(h){const keys=ST.b_have;const under=keys.filter(function(k){return S[k].B.means.cov90<0.85;});
  const worse=keys.filter(function(k){const p=S[k].B.paired;const q=S[k].mode==='snapshot'?p.rmse_vs_L_demosaic:p.rmse_vs_D;return q&&q.diff-2*q.se_cluster>0;});
  const vk_=Object.keys(D.b_vpred).sort(function(a,b){return b-a;});
  h.innerHTML='<b>B 的现状。</b>使用 x₀ 预测先验后，样本不再退化。在 '+keys.length+' 个设置上，单样本 RMSE 为 '+fx(Math.min.apply(null,keys.map(function(k){return S[k].B.means.single_rmse_pct;})),1)+'–'+fx(Math.max.apply(null,keys.map(function(k){return S[k].B.means.single_rmse_pct;})),1)+' %，而 v 预测先验为 '+vk_.map(function(k){return k+' dB 下 '+fx(D.b_vpred[k].means.single_rmse_pct,1)+' %';}).join(' 和 ')+'。在 '+keys.length+' 个设置中有 '+worse.length+' 个，32 样本均值比参考方法（D；快照模式下为先去马赛克再 MMSE）差 2 倍聚类标准误以上。在 '+keys.length+' 个设置中有 '+under.length+' 个样本欠离散（cov90 低于 0.85）。';};
SL.mainline=function(h){const cB=passes('B'),cC=passes('C'),tb=t2v('B',20),tc=t2v('C',20);
  const rule='判定对 T1 只计一次（五个信噪比全部通过才算通过），T2 每个信噪比各计一次，再加 T3 与 T4：每条路线最多通过 5 项。';
  if(!V.b_valid){h.innerHTML=rule+'C 通过 '+cC.length+' 项（'+(joinAnd(cC)||'无')+'）。B：'+bInvalid()+' — '+bInvalidWhy()+'，因此 B 的计数与主线需等待重新生成的判定。两条路线的完整结果均已报告。';return;}
  const t=V.tie_break;let tie;
  if(t&&t.lines&&t.lines.B&&t.lines.C){const b=t.lines.B,c=t.lines.C,w=t.winner,l=w==='B'?'C':'B';
    tie='在两条路线共有的 '+t.n_common+' 个 cube 上，'+w+' 的 '+sg(t.lines[w].mean_common,3)+' ± '+fx(t.lines[w].se_cluster_common,3)+' 低于 '+l+' 的 '+sg(t.lines[l].mean_common,3)+' ± '+fx(t.lines[l].se_cluster_common,3)+'（聚类标准误；在各自的 cube 上，B 为 '+sg(b.mean_own,3)+'，共 '+b.n_own+' 个，C 为 '+sg(c.mean_own,3)+'，共 '+c.n_own+' 个），因此主线为 <b>'+V.main_line+'</b>。';}
  else tie='B 的 '+sg(tb.crps_vs_V.mean,3)+'（50 个 cube）低于 C 的 '+sg(tc.crps_vs_V.mean,3)+'（200 个 cube），因此主线为 <b>'+V.main_line+'</b>。';
  h.innerHTML=rule+'B 通过 '+cB.length+' 项（'+(joinAnd(cB)||'无')+'），C 通过 '+cC.length+' 项（'+(joinAnd(cC)||'无')+'）。平局时取 20 dB 处配对 CRPS − CRPS<sub>V</sub> 较低的路线，无论它在该处是否通过 T2。'+tie+'B 胜在均值的准确性；其覆盖率在两个信噪比下都未通过 T2。两条路线的完整结果均已报告。';};
SL.verdictnote=function(h){h.innerHTML='判定 <code>'+ST.verdict_file+'</code>，生成于 '+(V.created||'—')+'。B 的单元格基于 B 的 50 个 cube，在共有文件上与 200-cube 参考方法配对（至少需要 '+V.min_common+' 个共有文件）。';};
SL.budget=function(h){const b=D.budget,e=b.eval_hours;const ev=e.main+e.C+e.B+e.B_vpred+(e.B_stale||0);   /* B_stale: non-x0 B records still in eval/ (not counted in e.B) */const tot=b.train_hours+b.pilot_hours+ev;
  h.innerHTML='<b>预算。</b>五个训练运行（含失败的 v 预测先验）共用 '+fx(b.train_hours,1)+' GPU 小时，另有三个先验试验用 '+fx(b.pilot_hours,1)+' GPU 小时。评测方面：L / D / V 记录用 '+fx(e.main,2)+' GPU 小时，C 用 '+fx(e.C,1)+'，已归档的 v 预测 B 记录（E1 60 / 50 dB 以及误启动的 E4 单元）用 '+fx(e.B_vpred,1)+'，'+b.b_records+' 条 x₀ 先验 B 记录用 '+fx(e.B,1)+(e.B_stale?'（另有 '+fx(e.B_stale,1)+' 用于尚未归档的过期非 x₀ B 记录）':'')+'（E4 的 L / D / V 与 C 记录已计入上述总数）。一条 B 记录约耗 '+fx(b.b_hours_mean,2)+' GPU 小时，而规格说明为全部 B 设定的上限为 12 GPU 小时。总计：RTX A4500 上约 '+fx(tot,0)+' GPU 小时。';};
/* horizontal-scroll cue for charts and tables that overflow their card (phones) */
function hints(){document.querySelectorAll('.scroll,.tbl').forEach(function(h){const over=h.scrollWidth>h.clientWidth+4;let n=h.nextElementSibling;const has=n&&n.classList&&n.classList.contains('hint');
  if(over&&!has){const d=document.createElement('p');d.className='hint';d.textContent='左右滑动查看完整'+(h.classList.contains('tbl')?'表格':'图表')+' →';h.parentNode.insertBefore(d,h.nextSibling);}else if(!over&&has)n.remove();});}

/* ---------- data-bound values ---------- */
function resolve(path){return path.split('.').reduce(function(o,k){return o==null?undefined:o[k];},D);}
document.querySelectorAll('[data-v]').forEach(function(h){const sp=h.dataset.v.split(':');const v=resolve(sp[0]);const f=sp[1];
  if(v==null){h.innerHTML=pend();return;}
  if(f==='e')h.textContent=sci(v);else if(f!=null&&typeof v==='number')h.textContent=Number(v).toFixed(+f);else h.textContent=ZW[String(v)]||String(v);});
document.querySelectorAll('[data-fig]').forEach(function(h){const f=F[h.dataset.fig];if(f)f(h);});
document.querySelectorAll('[data-chart]').forEach(function(h){const f=CH[h.dataset.chart];if(f)f(h);});
document.querySelectorAll('[data-table]').forEach(function(h){const f=T[h.dataset.table];if(f)f(h);});
document.querySelectorAll('[data-slot]').forEach(function(h){const f=SL[h.dataset.slot];if(f)f(h);});
hints();window.addEventListener('resize',hints);
document.querySelectorAll('details.fold').forEach(function(d){d.addEventListener('toggle',hints);});
