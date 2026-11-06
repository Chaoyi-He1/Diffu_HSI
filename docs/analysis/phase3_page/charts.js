/* ---------- Phase 3 figures, charts, tables and slots (helpers el, tx, box, arr, line, mk, rrect, cpath, hover, showTip, hideTip, svgPt, dataTable, clear come from the shared block) ---------- */
function fx(v,n){return v==null||isNaN(v)?'—':Number(v).toFixed(n);}
function sg(v,n){return v==null?'—':(v>0?'+':(v<0?'−':''))+Math.abs(Number(v)).toFixed(n);}
function pill(ok,txt){return '<span class="pill '+(ok?'ok':'no')+'">'+(txt||(ok?'PASS':'FAIL'))+'</span>';}
function pend(txt){return '<span class="pill nt">'+(txt||'pending')+'</span>';}
function prev(ok){return '<span class="pill wn">'+(ok?'preview: pass':'preview: fail')+'</span>';}
function seg(attr,draw){const bs=document.querySelectorAll('[data-'+attr+']');bs.forEach(function(b){b.addEventListener('click',function(){bs.forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');draw(b.getAttribute('data-'+attr));});});}
function tbl(host,head,rows,hiRow){const w=document.createElement('table');w.className='t';let h='<thead><tr>'+head.map(function(x){return '<th>'+x+'</th>';}).join('')+'</tr></thead><tbody>';rows.forEach(function(r,i){h+='<tr'+(hiRow&&hiRow(i)?' class="hi"':'')+'>'+r.map(function(c){return typeof c==='object'&&c!==null?'<td class="'+(c.cls||'')+'">'+c.v+'</td>':'<td>'+c+'</td>';}).join('')+'</tr>';});w.innerHTML=h+'</tbody>';host.appendChild(w);}
function num(v){return {v:v,cls:'num'};}
function sci(v){if(v==null)return '—';const e=Math.floor(Math.log10(v));return (v/Math.pow(10,e)).toFixed(1)+' × 10'+String(e).split('').map(function(c){return '⁰¹²³⁴⁵⁶⁷⁸⁹'['0123456789'.indexOf(c)]||(c==='-'?'⁻':c);}).join('');}

const S=D.settings,V=D.verdict,ST=D.status,F0=D.facts;
const MC={L:'--ink-2',L_demosaic:'--de',D:'--accent',V:'--s3',C:'--s2',C1:'--s2',B:'--s4',Bv:'--s4'};
const MN={L:'L, MMSE',L_demosaic:'L, demosaic-then-MMSE',D:'D, regressor',V:'V, variance head',C:'C, mean of 32',C1:'C1, one call',B:'B, mean of 32 (50 cubes)',Bv:'B, v-prediction prior (archived)'};
const DEVN={device1:'Device 1',device2:'Device 2',ph5:'PH5'};
function set(dev,mode,snr){return S[dev+'_'+mode+'_'+snr];}
function bOk(s){return s&&s.B&&s.B.status==='final';}
function rmse(s,m){if(!s)return null;if(m==='B')return bOk(s)?s.B.means.mean_rmse_pct:null;const x=s.methods[m];if(!x)return null;return (m==='D'||m==='C1')?x.rmse_pct:x.mean_rmse_pct;}
function met(s,m,k){if(!s)return null;if(m==='B')return bOk(s)?s.B.means[k]:null;const x=s.methods[m];return x?x[k]:null;}
function vk(snr){return Number(snr).toFixed(1);}
function bStatusTxt(s){if(!s||!s.B)return 'pending';return s.B.status==='dropped'?'dropped':(s.B.status==='final'?'':'pending');}
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
  (o.refs||[]).forEach(function(r){const x1=r.x0==null?m.l:X(r.x0),x2=r.x1==null?m.l+pw:X(r.x1);line(svg,x1,Y(r.y),x2,Y(r.y),r.cls||'ref');tx(svg,r.lx==null?x2+6:X(r.lx),Y(r.y)+(r.dy==null?4:r.dy),r.label,'reflbl',r.anchor||'start');});
  const live=o.series.filter(function(s){return s.data.length;});
  live.forEach(function(s){
    const d=s.data.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ');
    if(s.data.length>1)el('path',{d:d,class:'ln',style:'stroke:var('+s.c+');'+(s.faint?'opacity:.55;':'')+(s.dash?'stroke-dasharray:5 4;':'')+(s.thin?'stroke-width:1.3;':'')},svg);
    if(s.marks!==false)s.data.forEach(function(p){el('circle',{cx:X(p[0]),cy:Y(p[1]),r:s.r||3.2,class:'dot',style:'fill:var('+s.c+');'+(s.faint?'opacity:.6':'')},svg);});
    if(s.label){const L=s.data[o.labelAt==='first'?0:s.data.length-1];tx(svg,X(L[0])+(o.labelAt==='first'?-8:8),Y(L[1])+(s.ldy||4),s.label,'endlbl',o.labelAt==='first'?'end':'start');}
  });
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
      if(v==null){const t=(g.tags&&g.tags[si])||'pending';const tt=tx(svg,x+bw/2,Y(o.y[0])-6,t,'sm','middle');tt.setAttribute('transform','rotate(-90 '+(x+bw/2)+' '+(Y(o.y[0])-6)+')');tt.setAttribute('text-anchor','start');return;}
      const vv=Math.min(v,o.y[1]);el('path',{d:cpath(x,Y(o.y[0]),bw,Y(o.y[0])-Y(vv),4),style:'fill:var('+s.c+')'+(s.faint?';opacity:.55':'')},svg);
      tx(svg,x+bw/2,Y(vv)-6,o.vfmt(v),'vlbl','middle');
      const hit=el('rect',{x:x,y:m.t,width:bw+5,height:ph,fill:'transparent',class:'barhit'},svg);hover(hit,function(){return '<div><b style="margin:0;padding:0">'+g.label+' · '+s.name+'</b></div><div class="sub">'+o.vfmt(v)+(g.notes&&g.notes[si]?' · '+g.notes[si]:'')+'</div>';});});});
  const lg=document.createElement('div');lg.className='legend';lg.innerHTML=o.series.map(function(s){return '<span><i class="sq" style="background:var('+s.c+')"></i>'+s.name+'</span>';}).join('');h.parentNode.insertBefore(lg,h);
  dataTable(h,[o.gname||'group'].concat(o.series.map(function(s){return s.name;})),o.groups.map(function(g){return [g.label].concat(g.values.map(function(v,i){return v==null?((g.tags&&g.tags[i])||'pending'):o.vfmt(v);}));}));}

/* ---------- figures ---------- */
const F={};
F.pipeline=function(h){
  const s=mk(h,960,280,760);
  tx(s,10,20,'MEASUREMENT','tag');
  box(s,10,32,170,62,['Scene x (31 bands)','HFD gate cubes'],'box');
  arr(s,[[184,63],[222,63]]);
  box(s,226,32,232,62,['Sensor Rᵀ, 30 states','sequential or 5 × 6 snapshot'],'box');
  arr(s,[[462,63],[500,63]]);
  box(s,504,32,170,62,['y = Rᵀx + n','σ from SNR, global P_y'],'box');
  tx(s,288,128,'THROUGH THE SUFFICIENT STATISTIC','tag');
  box(s,10,140,232,62,['L: closed-form MMSE','x̂ and posterior std s'],'box');
  arr(s,[[588,98],[588,118],[126,118],[126,136]]);
  arr(s,[[246,171],[284,171]]);
  box(s,288,140,120,62,['D','one call'],'box box-acc');
  box(s,418,140,120,62,['V','Gaussian width'],'box box-s3');
  box(s,548,140,126,62,['C','32 samples'],'box box-s2');
  tx(s,10,236,'FROM THE MEASUREMENT DIRECTLY','tag');
  box(s,10,206+42,664,26,['B: unconditional prior (spectra only) + ΠGDM with the exact H, σ and mask; 32 samples'],'box box-s4');
  arr(s,[[678,63],[730,63],[730,261],[678,261]],'ar');
  tx(s,750,20,'SCORED','tag');
  box(s,750,32,200,170,['E1 Device 1, 5 SNRs','E2 Device 2, PH5','E3 snapshot','RMSE · CRPS','coverage · AUSE','clauses T1–T4'],'box');
  arr(s,[[678,171],[746,171]]);
};

const CH={};
CH.sensorR=function(h){const R=D.sensor.R_device1,wl=D.sensor.wavelengths;const n=R[0].length;
  const series=[];for(let j=0;j<n;j++){series.push({name:'state '+(j+1),c:'--accent',data:wl.map(function(w,i){return [w,R[i][j]];}),marks:false,thin:true});}
  const o=xy(h,{w:520,h:300,mr:20,minW:380,x:[451,855],y:[0,1],xTicks:[451,550,650,750,855],yTicks:[0,0.25,0.5,0.75,1],yFmt:function(v){return v.toFixed(2);},yTitle:'normalised response',xTitle:'wavelength (nm, legacy axis)',series:series,tipX:function(v){return v.toFixed(0)+' nm';},tipY:function(v){return v.toFixed(3);}});
  o.svg.querySelectorAll('path.ln').forEach(function(p,j){p.style.opacity=(0.25+0.75*j/(n-1)).toFixed(2);});
  dataTable(h,['band (nm)'].concat(R[0].map(function(_,j){return 's'+(j+1);})),wl.map(function(w,i){return [w.toFixed(1)].concat(R[i].map(function(v){return v.toFixed(3);}));}));};
CH.sensorSV=function(h){const dv=D.sensor.devices;const cs={device1:'--accent',device2:'--s3',ph5:'--s2'};
  const ser=['device1','device2','ph5'].map(function(k){return {name:DEVN[k],c:cs[k],label:DEVN[k],marks:false,data:dv[k].sv_rel.map(function(v,i){return [i+1,Math.max(v,1e-12)];})};});
  xy(h,{w:520,h:300,mr:70,minW:380,log:true,x:[1,30],y:[1e-12,1],xTicks:[1,5,10,15,20,25,30],yTicks:[1e-12,1e-9,1e-6,1e-3,1],yFmt:function(v){return '1e'+Math.round(Math.log10(v));},yTitle:'σᵢ / σ₁ (log)',xTitle:'index i',series:ser,refs:[{y:0.01,label:'1 %',lx:30,anchor:'end',dy:-6}],tipX:function(v){return 'singular value '+v;},tipY:function(v){return v.toExponential(2);},
    table:{head:['i','Device 1','Device 2','PH5'],rows:dv.device1.sv_rel.map(function(_,i){return [i+1,dv.device1.sv_rel[i].toExponential(2),dv.device2.sv_rel[i].toExponential(2),dv.ph5.sv_rel[i].toExponential(2)];})}});};
CH.stage0=function(h){const L=D.stage0.L_curves,raw=D.stage0.curves_raw_device1;const snr=[20,30,40,50,60];
  const g=function(o,dev){return snr.map(function(x){return [x,o[vk(x)]];});};
  const ser=[{name:'Device 1',c:'--accent',label:'Device 1',data:g(L.device1.sequential)},{name:'Device 1, raw response',c:'--accent',dash:true,faint:true,data:g(raw.sequential)},
    {name:'Device 2',c:'--s3',label:'Device 2',data:g(L.device2.sequential)},{name:'PH5',c:'--s2',label:'PH5',data:g(L.ph5.sequential)}];
  xy(h,{log:true,x:[11,62],y:[0.8,15],xTicks:snr,xFmt:function(v){return v===60?'60 (floor)':String(v);},yTicks:[1,2,5,10],yTitle:'MMSE RMSE % (log)',xTitle:'SNR (dB)',labelAt:'first',ml:56,mr:20,series:ser,tipX:function(v){return v+' dB';},
    table:{head:['SNR (dB)','Device 1','Device 1 raw R','Device 2','PH5'],rows:snr.slice().reverse().map(function(x){return [x,fx(L.device1.sequential[vk(x)],3),fx(raw.sequential[vk(x)],3),fx(L.device2.sequential[vk(x)],3),fx(L.ph5.sequential[vk(x)],3)];})}});
  const svg=h.querySelector('svg');};
function e1series(metric){const snrs=[20,30,40,50,60];const out={};
  ['L','D','V','C','C1','B'].forEach(function(m){out[m]=[];snrs.forEach(function(x){const s=set('device1','sequential',x);const v=metric==='rmse'?rmse(s,m):met(s,m,'crps');if(v!=null)out[m].push([x,v]);});});
  out.Bv=[];snrs.forEach(function(x){const b=D.b_vpred[String(x)];if(b){const v=metric==='rmse'?b.means.mean_rmse_pct:b.means.crps;out.Bv.push([x,v]);}});return out;}
CH.e1rmse=function(h){const d=e1series('rmse');const ms=['L','D','C','C1','B','Bv'];
  legendInto('e1',ms.map(function(m){return {n:MN[m]+(m==='B'&&!d.B.length?' — pending':''),c:MC[m],dash:m==='C1'||m==='Bv',faint:m==='Bv'};}));
  const ser=ms.map(function(m){return {name:MN[m],c:MC[m],data:d[m],dash:m==='C1'||m==='Bv',faint:m==='Bv',label:{L:'L',D:'D',C:'C',B:'B'}[m]};});
  const snrs=[60,50,40,30,20];
  xy(h,{log:true,x:[12,62],y:[0.8,10],xTicks:[20,30,40,50,60],xFmt:function(v){return v===60?'60 (floor)':String(v);},yTicks:[1,2,3,5,10],yTitle:'RMSE % of range (log)',xTitle:'SNR (dB)',labelAt:'first',mr:20,series:ser,tipX:function(v){return v+' dB';},
    refs:[{y:D.stage0.sanity_reference_rmse_pct,x0:55,lx:54,anchor:'end',dy:4,label:'floor reference '+fx(D.stage0.sanity_reference_rmse_pct,3)}],
    table:{head:['SNR (dB)','L','D','C mean-of-32','C1','C single','B mean-of-32 (50)','B v-pred (archived)'],rows:snrs.map(function(x){const s=set('device1','sequential',x);const bv=D.b_vpred[String(x)];return [x,fx(rmse(s,'L'),3),fx(rmse(s,'D'),3),fx(rmse(s,'C'),3),fx(rmse(s,'C1'),3),fx(met(s,'C','single_rmse_pct'),3),bOk(s)?fx(rmse(s,'B'),3):bStatusTxt(s),bv?fx(bv.means.mean_rmse_pct,3):'—'];})}});};
CH.e1crps=function(h){const d=e1series('crps');const ms=['L','V','C','B','Bv'];
  legendInto('e1c',ms.map(function(m){return {n:(m==='C'?'C, 32 samples':(m==='B'?'B, 32 samples (50 cubes)':MN[m]))+(m==='B'&&!d.B.length?' — pending':''),c:MC[m],dash:m==='Bv',faint:m==='Bv'};}));
  const ser=ms.map(function(m){return {name:MN[m],c:MC[m],data:d[m],dash:m==='Bv',faint:m==='Bv',label:{L:'L',V:'V',C:'C',B:'B'}[m],ldy:{L:-3,V:12}[m]};});
  xy(h,{log:true,x:[12,62],y:[0.4,8],xTicks:[20,30,40,50,60],xFmt:function(v){return v===60?'60 (floor)':String(v);},yTicks:[0.5,1,2,4,8],yTitle:'CRPS % (log)',xTitle:'SNR (dB)',labelAt:'first',mr:20,series:ser,tipX:function(v){return v+' dB';},
    table:{head:['SNR (dB)','L','V','C','B (50)','B v-pred (archived)'],rows:[60,50,40,30,20].map(function(x){const s=set('device1','sequential',x);const bv=D.b_vpred[String(x)];return [x,fx(met(s,'L','crps'),3),fx(met(s,'V','crps'),3),fx(met(s,'C','crps'),3),bOk(s)?fx(met(s,'B','crps'),3):bStatusTxt(s),bv?fx(bv.means.crps,3):'—'];})}});};
const T2S=[30,20];
CH.t2crps=function(h){const ms=['L','V','C','B'];
  bars(h,{y:[0,4],ticks:[0,1,2,3,4],fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(3);},yTitle:'mean CRPS %',gname:'SNR',
    series:ms.map(function(m){return {name:{L:'L posterior',V:'V variance head',C:'C diffusion',B:'B guided prior (50 cubes)'}[m],c:MC[m]};}),
    groups:T2S.map(function(x){const s=set('device1','sequential',x);return {label:x+' dB',values:ms.map(function(m){return met(s,m,'crps');}),tags:ms.map(function(m){return m==='B'?bStatusTxt(s)||'pending':null;})};})});};
CH.t2ause=function(h){const ms=['L','V','C','B'];
  bars(h,{w:520,minW:380,h:300,y:[0,0.5],ticks:[0,0.1,0.2,0.3,0.4,0.5],fmt:function(v){return v.toFixed(1);},vfmt:function(v){return v.toFixed(3);},yTitle:'AUSE',gname:'SNR',
    series:ms.map(function(m){return {name:m,c:MC[m]};}),
    groups:T2S.map(function(x){const s=set('device1','sequential',x);return {label:x+' dB',values:ms.map(function(m){return met(s,m,'ause');})};})});};
CH.t2cov=function(h){const ms=['L','V','C','B'];const W=520,H=300,m={l:46,r:12,t:16,b:50};const svg=mk(h,W,H,380);const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const Y=function(v){return m.t+ph-(v-0.2)/(1.0-0.2)*ph;};
  el('rect',{x:m.l,y:Y(0.95),width:pw,height:Y(0.85)-Y(0.95),style:'fill:var(--good-bg)'},svg);
  [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,t.toFixed(1),'tick','end');});
  line(svg,m.l,Y(0.9),m.l+pw,Y(0.9),'refacc');line(svg,m.l,Y(0.5),m.l+pw,Y(0.5),'refacc');
  tx(svg,4,12,'coverage','axt');
  const slots=[];T2S.forEach(function(x){ms.forEach(function(mm){slots.push([x,mm]);});});const sw=pw/slots.length;
  T2S.forEach(function(x,gi){tx(svg,m.l+sw*(gi*ms.length+ms.length/2),H-m.b+36,x+' dB','bs','middle');if(gi)line(svg,m.l+sw*gi*ms.length,m.t,m.l+sw*gi*ms.length,m.t+ph,'axis');});
  const rows=[];
  slots.forEach(function(q,i){const s=set('device1','sequential',q[0]);const cx=m.l+sw*(i+0.5);tx(svg,cx,H-m.b+18,q[1],'tick','middle');
    const c50=met(s,q[1],'cov50'),c90=met(s,q[1],'cov90');
    if(c90==null){tx(svg,cx,Y(0.62),'pending','sm','middle').setAttribute('transform','rotate(-90 '+cx+' '+Y(0.62)+')');rows.push([q[0],q[1],'pending','pending']);return;}
    const a=el('circle',{cx:cx,cy:Y(c90),r:5.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    const b=el('rect',{x:cx-4.5,y:Y(c50)-4.5,width:9,height:9,rx:1.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    [a,b].forEach(function(n){hover(n,function(){return '<div><b style="margin:0;padding:0">'+q[1]+' · '+q[0]+' dB</b></div><div class="sub">cov50 '+c50.toFixed(3)+' · cov90 '+c90.toFixed(3)+'</div>';});});
    rows.push([q[0],q[1],c50.toFixed(3),c90.toFixed(3)]);});
  dataTable(h,['SNR','method','cov50','cov90'],rows);};
CH.e2=function(h){const ms=['L','D','C','B'];
  function draw(x){clear(h);const prevLg=h.parentNode.querySelector('.legend');if(prevLg)prevLg.remove();
    bars(h,{y:[0,25],ticks:[0,5,10,15,20,25],fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(2);},yTitle:'RMSE % ('+x+' dB)',gname:'sensor',
      series:ms.map(function(m){return {name:{L:'L, MMSE',D:'D, regressor',C:'C, mean of 32',B:'B, mean of 32 (50 cubes)'}[m],c:MC[m]};}),
      groups:['device1','device2','ph5'].map(function(dev){const s=set(dev,'sequential',x);return {label:DEVN[dev]+(dev==='device1'?' (trained)':''),values:ms.map(function(m){return rmse(s,m);}),tags:ms.map(function(m){return m==='B'?(bStatusTxt(s)||'pending'):null;})};})});}
  draw('30');seg('e2',draw);};
CH.e3rmse=function(h){const snrs=[20,30,40,60];const ms=['L','L_demosaic','D','C','C1','B'];
  legendInto('e3',ms.map(function(m){const has=snrs.some(function(x){return rmse(set('device1','snapshot',x),m)!=null;});return {n:(m==='L'?'L, exact per-pixel posterior':MN[m])+(m==='B'&&!has?' — pending':''),c:MC[m],dash:m==='C1'};}));
  const ser=ms.map(function(m){return {name:m==='L'?'L exact':MN[m],c:MC[m],dash:m==='C1',label:{L:'L exact',L_demosaic:'L demosaic',D:'D',C:'C',B:'B'}[m],data:snrs.map(function(x){const v=rmse(set('device1','snapshot',x),m);return v==null?null:[x,v];}).filter(Boolean)};});
  const t=D.facts.Ldem_target_30;
  xy(h,{log:true,x:[9,62],y:[2,16],xTicks:[20,30,40,60],xFmt:function(v){return v===60?'60 (floor)':String(v);},yTicks:[2,3,5,8,12,16],yTitle:'RMSE % (log)',xTitle:'SNR (dB)',labelAt:'first',mr:20,ml:56,series:ser,tipX:function(v){return v+' dB';},
    refs:[{y:t,x0:26,x1:34,lx:35,label:'T4 target '+fx(t,2),cls:'refacc'}],
    table:{head:['SNR (dB)','L exact','L demosaic','D','C mean-of-32','C1','C single','B (50)'],rows:[60,40,30,20].map(function(x){const s=set('device1','snapshot',x);return [x,fx(rmse(s,'L'),3),fx(rmse(s,'L_demosaic'),3),fx(rmse(s,'D'),3),fx(rmse(s,'C'),3),fx(rmse(s,'C1'),3),fx(met(s,'C','single_rmse_pct'),3),bOk(s)?fx(rmse(s,'B'),3):bStatusTxt(s)];})}});};
CH.bprobe=function(h){const P=D.b_prior_probe;const cs={blind:['--ink-2',true,false],vpred:['--s2',false,false],g0:['--de',false,true],tt:['--de',true,true],x0pilot:['--accent',false,true],x0:['--s4',false,false]};
  legendInto('probe',P.series.map(function(s){return {n:s.name,c:cs[s.key][0],dash:cs[s.key][1],faint:cs[s.key][2]};}).concat([{n:'mean-cube predictor',c:'--muted',dash:true}]));
  xy(h,{log:true,x:[0,920],y:[2,32],xTicks:[25,100,200,300,400,500,700,900],yTicks:[2,3,5,10,20,30],yTitle:'x̂₀ error % (log)',xTitle:'diffusion timestep t',mr:20,
    series:P.series.map(function(s){return {name:s.name,c:cs[s.key][0],dash:cs[s.key][1],faint:cs[s.key][2],data:P.t.map(function(t,i){return [t,s.err[i]];})};}),
    refs:[{y:P.mean_cube,label:'mean cube '+fx(P.mean_cube,2),lx:560,anchor:'start',dy:-6}],tipX:function(v){return 't = '+v;},tipY:function(v){return v.toFixed(1)+' %';},
    table:{head:['t'].concat(P.series.map(function(s){return s.name;})),rows:P.t.map(function(t,i){return [t].concat(P.series.map(function(s){return s.err[i].toFixed(1);}));})}});};

/* ---------- tables ---------- */
const T={};
T.e3unc=function(h){const ms=['L','L_demosaic','V','C','B'];const nm={L:'L exact',L_demosaic:'L demosaic',V:'V',C:'C',B:'B (50 cubes)'};
  tbl(h,['SNR'].concat(ms.map(function(m){return nm[m];})),[60,40,30,20].map(function(x){const s=set('device1','snapshot',x);return [x+' dB'].concat(ms.map(function(m){const c=met(s,m,'crps');if(c==null)return pend(bStatusTxt(s)||'pending');return num('CRPS '+fx(c,2)+'<br>'+fx(met(s,m,'cov50'),2)+' / '+fx(met(s,m,'cov90'),2)+' · '+fx(met(s,m,'ause'),2));}));}),function(i){return i===2;});
  const p=document.createElement('p');p.style.fontSize='14px';p.style.color='var(--muted)';p.textContent='Each cell: CRPS %, then cov50 / cov90 · AUSE. The 30 dB row (highlighted) is T4’s setting.';h.appendChild(p);};
T.toy=function(h){const t=D.toy;if(!t){h.innerHTML=pend('toy record not found');return;}const g=t.gate;
  const inb=function(r){return r.rmse_excess<=g.rmse_excess&&Math.abs(r.var_ratio-1)<=g.var_tol&&r.cov90>=g.cov90[0]&&r.cov90<=g.cov90[1]&&r.cov50>=g.cov50[0]&&r.cov50<=g.cov50[1];};
  const rk=function(r){return (r.C_kind==='isotropic'?0:2)+(r.method==='pigdm'?0:1);};const rows=t.rows.slice().sort(function(a,b){return rk(a)-rk(b)||(a.mode>b.mode)-(a.mode<b.mode)||a.n_steps-b.n_steps;});
  tbl(h,['prior','method','mode','steps','RMSE excess','variance ratio','cov50','cov90','CRPS ratio','toy gate'],rows.map(function(r){return [r.C_kind==='isotropic'?'isotropic N(0, I)':'HFD covariance (report only)',r.method==='pigdm'?'ΠGDM':'DPS',r.mode,num(r.n_steps),num(fx(r.rmse_excess,4)),num(fx(r.var_ratio,3)),num(fx(r.cov50,3)),num(fx(r.cov90,3)),num(fx(r.crps_ratio,4)),r.C_kind==='isotropic'&&r.method==='pigdm'?pill(inb(r),inb(r)?'met':'missed'):'—'];}),function(i){const r=rows[i];return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});};
function bRow(label,s,b,exp){if(!b||b.status!=='final'&&b.status!=='archived'&&b.status!=='stale'){return [label,b&&b.status==='dropped'?pend('dropped'):pend(),'','','','','',''];}
  const mm=b.means,sub=b.on_subset,p=b.paired;const ref=s.mode==='snapshot'?'L_demosaic':'D';const pr=p['rmse_vs_'+ref];const keyRef=ref==='D'?'rmse_pct':'mean_rmse_pct';
  return [label+(b.status==='final'?' <span class="pill nt">x₀</span>':' <span class="pill nt">v-pred</span>'),num(fx(mm.mean_rmse_pct,2)+' / '+fx(mm.single_rmse_pct,1)),num(fx(mm.crps,2)),num(fx(mm.cov50,2)+' / '+fx(mm.cov90,2)),num(sg(mm.single_hf,1)),
    num((sub.L?fx(sub.L.mean_rmse_pct,2):'—')+' / '+(sub[ref]?fx(sub[ref][keyRef],2):'—')+' / '+(sub.C?fx(sub.C.mean_rmse_pct,2):'—')),
    num(pr?sg(pr.diff,2)+' ± '+fx(pr.se_cluster,2)+' ('+fx(pr.ratio,2)+'×)':'—'),num(fx(b.hours,2))];}
T.brec=function(h){const rows=[];
  [['E1','device1','sequential',[60,50,40,30,20]],['E2','device2','sequential',[40,30,20]],['E3','device1','snapshot',[60,40,30,20]]].forEach(function(e){e[3].forEach(function(x){const s=set(e[1],e[2],x);const lab=e[0]+' · '+DEVN[e[1]]+(e[2]==='snapshot'?' snapshot':'')+' · '+x+' dB';
    if(e[0]==='E1'&&D.b_vpred[String(x)])rows.push(bRow(lab,s,D.b_vpred[String(x)],e[0]));rows.push(bRow(lab,s,s.B,e[0]));});});
  tbl(h,['setting · prior','B RMSE mean-of-32 / single (%)','B CRPS','B cov50 / cov90','B single log HF','L / D (snapshot: L demosaic) / C on the same cubes','B − D (snapshot: − L demosaic) ± cl. SE (ratio)','GPU-h'],rows);};
function clauseCellT1(c,limit){if(!c)return pend();return {v:pill(c.passed)+' <span class="num">ratio '+fx(c.ratio,3)+' · 2-SE bound '+fx(c.ratio_2se,3)+' (limit '+fx(limit,2)+')</span>',cls:''};}
function previewT1(p,limit){if(!p)return null;return {v:prev(p.passed)+' <span class="num">ratio '+fx(p.ratio,3)+' · 2-SE bound '+fx(p.ratio_2se,3)+' (limit '+fx(limit,2)+', n '+p.n+')</span>',cls:''};}
function bClause(fromVerdict,fallback){if(V.b_valid&&fromVerdict)return fromVerdict;return fallback||{v:pend(),cls:''};}
T.clauses=function(h){const th=V.thresholds||{};const rows=[];
  [60,50,40,30,20].forEach(function(x){const c=V.T1&&V.T1.C?V.T1.C[vk(x)]:null;const vb=V.T1&&V.T1.B?V.T1.B[vk(x)]:null;const s=set('device1','sequential',x);
    rows.push(['<b>T1</b> · '+x+' dB','mean-of-32 RMSE ≤ '+fx(th.T1_RATIO,2)+' × D (2 cl. SE)',clauseCellT1(c,th.T1_RATIO),bClause(vb&&clauseCellT1(vb,th.T1_RATIO),bOk(s)?previewT1(s.B.paired.rmse_vs_D,th.T1_RATIO):null)]);});
  T2S.forEach(function(x){const c=V.T2&&V.T2.C?V.T2.C[vk(x)]:null;const vb=V.T2&&V.T2.B?V.T2.B[vk(x)]:null;const s=set('device1','sequential',x);
    const cell=function(c){if(!c)return pend();return {v:pill(c.passed)+' <span class="num">CRPS − L '+sg(c.crps_vs_L.mean,3)+' ± '+fx(c.crps_vs_L.se_cluster,3)+' · − V '+sg(c.crps_vs_V.mean,3)+' ± '+fx(c.crps_vs_V.se_cluster,3)+' · cov90 '+fx(c.cov90,3)+(c.cov90_passed?'':' (out of band)')+'</span>',cls:''};};
    let pv=null;if(bOk(s)){const p=s.B.paired,cv=s.B.means.cov90;const ok=p.crps_vs_L.diff+2*p.crps_vs_L.se_cluster<0&&p.crps_vs_V.diff+2*p.crps_vs_V.se_cluster<0&&cv>=th.T2_COV90[0]&&cv<=th.T2_COV90[1];
      pv={v:prev(ok)+' <span class="num">CRPS − L '+sg(p.crps_vs_L.diff,3)+' ± '+fx(p.crps_vs_L.se_cluster,3)+' · − V '+sg(p.crps_vs_V.diff,3)+' ± '+fx(p.crps_vs_V.se_cluster,3)+' · cov90 '+fx(cv,3)+'</span>',cls:''};}
    rows.push(['<b>T2</b> · '+x+' dB','CRPS below L and V (2 cl. SE), cov90 ∈ ['+th.T2_COV90.join(', ')+']',cell(c),bClause(vb&&cell(vb),pv)]);});
  const t3=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">ratio to L: Device 1 '+fx(c.ratio_dev1,3)+' · Device 2 '+fx(c.ratio_dev2,3)+' (limit '+fx(th.T3_RATIO*c.ratio_dev1,3)+') · PH5 '+fx(c.ph5,3)+' (report)</span>',cls:''};};
  let t3p=null;const s1=set('device1','sequential',30),s2=set('device2','sequential',30);if(bOk(s1)&&bOk(s2)){const r1=s1.B.paired.rmse_vs_L.ratio,r2=s2.B.paired.rmse_vs_L.ratio;t3p={v:prev(r2<=th.T3_RATIO*r1)+' <span class="num">ratio to L: Device 1 '+fx(r1,3)+' · Device 2 '+fx(r2,3)+' (limit '+fx(th.T3_RATIO*r1,3)+')</span>',cls:''};}
  rows.push(['<b>T3</b> · 30 dB','(B or C / L) on Device 2 ≤ '+fx(th.T3_RATIO,2)+' × the same on Device 1',t3(V.T3&&V.T3.C)||pend(),bClause(t3(V.T3&&V.T3.B),t3p)]);
  const t4=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">ratio '+fx(c.ratio,3)+' · 2-SE bound '+fx(c.ratio_2se,3)+' (limit '+fx(th.T4_RATIO,2)+')'+(c.D?'; D: '+(c.D.passed?'pass':'fail')+' at '+fx(c.D.ratio,3):'')+'</span>',cls:''};};
  const s4=set('device1','snapshot',30);rows.push(['<b>T4</b> · snapshot 30 dB','RMSE ≤ '+fx(th.T4_RATIO,2)+' × demosaic-then-MMSE (2 cl. SE)',t4(V.T4&&V.T4.C)||pend(),bClause(t4(V.T4&&V.T4.B),bOk(s4)?previewT1(s4.B.paired.rmse_vs_L_demosaic,th.T4_RATIO):null)]);
  const sc=V.sanity||{};const fp=sc.floor_pass||{};const mo=sc.monotone||{};
  rows.push(['<b>sanity</b>','RMSE non-increasing in SNR; at 60 dB ≤ '+fx(sc.floor_reference_rmse_pct,3)+' % + 2 SE',{v:(mo.C&&fp.C?pill(true):pill(false))+' <span class="num">C monotone '+(mo.C?'yes':'no')+' · floor '+(fp.C?'pass':'fail')+'</span>',cls:''},V.b_valid?{v:(mo.B&&fp.B?pill(true):pill(false))+' <span class="num">floor '+(fp.B?'pass':'fail')+'</span>',cls:''}:{v:pend(),cls:''}]);
  const cnt=V.counts||{};rows.push(['<b>clauses passed</b>','of T1–T4 (a clause passes only if every setting does)',num('<b>'+(cnt.C==null?'—':cnt.C)+' of 4</b>'),V.b_valid?num('<b>'+cnt.B+' of 4</b>'):{v:pend(),cls:''}]);
  tbl(h,['clause · setting','rule','line C (200 cubes)','line B (50 cubes)'],rows,function(i){return i>=rows.length-1;});};
T.runs=function(h){const rows=D.runs.map(function(r){return [r.run+(r.failed?' <span class="pill no">failed prior</span>':''),r.role,(r.prediction_type||'v')+' · '+(r.loss_type||'')+(r.snr_gamma&&r.prediction_type!=='direct'?' · min-SNR γ '+r.snr_gamma:''),num(r.epochs+' / '+r.planned),num(fx(r.sec_per_epoch,0)),num(fx(r.hours,2)),num(String(r.selected)),num(fx(r.val_at_sel,5))];});
  D.pilots.forEach(function(p){rows.push([p.run,'prior pilot',(p.prediction_type||'v')+' · '+(p.loss_type||'')+(p.snr_gamma!=null?' · min-SNR γ '+p.snr_gamma:''),num(p.epochs+' / '+p.planned),num(fx(p.sec_per_epoch,0)),num(fx(p.hours,2)),num('—'),num('—')]);});
  tbl(h,['run','role','parameterisation · loss','epochs','s / epoch','GPU-h','selected epoch','monitoring loss there'],rows);};

/* ---------- slots ---------- */
const SL={};
function bList(keys){return keys.map(function(k){const s=S[k];return (s.exp+' '+DEVN[s.device]+(s.mode==='snapshot'?' snapshot':'')+' '+s.snr+' dB');}).join(', ');}
SL.status=function(h){const nf=ST.final.length,np=ST.pending.length;const nB=ST.b_have.length,eB=ST.b_expected.length;
  h.innerHTML='<b>Status (records as of '+D.generated+').</b> '+(nf-nB)+' main and C records final (L, D, V, C, C1 on 200 cubes in every E1–E3 setting). Line B: <b>'+nB+' of '+eB+'</b> x₀-prior records present'+(nB?' ('+bList(ST.b_have)+')':'')+(ST.b_pending.length?'; <b>pending</b>: '+bList(ST.b_pending)+'.':'; none pending.')+
  ' Verdict file <code>'+ST.verdict_file+'</code>'+(V.report_only?' (report-only)':'')+'; '+(V.b_valid?'its B clauses are current.':'its B clauses are not used here ('+V.b_invalid_reason+'). B cells show the page’s paired preview or “pending”.')+(np&&np>ST.b_pending.length?' Other pending records: '+ST.pending.filter(function(r){return r.slice(-2)!=='_B';}).join(', ')+'.':'');};
SL.countC=function(h){h.textContent=(V.counts&&V.counts.C!=null?V.counts.C:'—')+' of 4';};
SL.countB=function(h){h.textContent=V.b_valid&&V.counts?V.counts.B+' of 4':'pending';};
SL.countBsub=function(h){h.textContent=ST.b_have.length+' of '+ST.b_expected.length+' x₀-prior records present';};
SL.normcheck=function(h){const sc=D.stage0.shape_check;h.textContent='The normalisation check '+(D.stage0.normalisation_check_passed?'passed':'FAILED')+': with the raw (un-normalised) response the curve keeps its shape within the 10 % rule (largest deviation '+fx(100*sc.sequential.max_dev,1)+' % sequential, '+fx(100*sc.snapshot.max_dev,1)+' % snapshot), so the normalised R is the paper’s sensor and the raw curve (dashed) goes to the appendix.';};
SL.angles=function(h){const a=D.stage0.principal_angles;h.innerHTML='<b>Principal angles between the sensors’ column spaces.</b> Device 1 and Device 2 share a '+F0.ang_d1d2_shared+'-dimensional subspace: only '+F0.ang_d1d2_nonzero+' of the '+a.device1_device2.n+' angles is non-zero ('+fx(F0.ang_d1d2_max,1)+'°). The transfer test is therefore about how the two sensors weight directions, not about different subspaces. PH5’s response has rank '+F0.ph5_rank+', and every angle to Device 1’s space is at most '+fx(F0.ang_d1ph5_max,1)+'°, so PH5 measures a '+F0.ph5_rank+'-dimensional part of what Device 1 measures, with far worse conditioning.';};
SL.t1marks=function(h){const th=V.thresholds||{};h.innerHTML='<span><b>T1</b> (limit '+fx(th.T1_RATIO,2)+'):</span>'+[60,50,40,30,20].map(function(x){const c=V.T1&&V.T1.C&&V.T1.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T1.B&&V.T1.B[vk(x)];
    let b;if(vb)b=pill(vb.passed,'B '+fx(vb.ratio,2));else if(bOk(s))b='<span class="pill wn">B '+fx(s.B.paired.rmse_vs_D.ratio,2)+'*</span>';else b=pend('B pending');
    return '<span>'+x+' dB '+(c?pill(c.passed,'C '+fx(c.ratio,2)):pend('C'))+' '+b+'</span>';}).join('');};
SL.t1text=function(h){const c=V.T1&&V.T1.C||{};const ks=[60,50,40,30,20];const nf=ks.filter(function(x){return c[vk(x)]&&!c[vk(x)].passed;}).length;
  h.innerHTML='C fails T1 at '+nf+' of 5 SNRs. Its ratio to D is '+ks.map(function(x){return fx(c[vk(x)]&&c[vk(x)].ratio,3);}).join(' / ')+' at 60 / 50 / 40 / 30 / 20 dB. At the high-SNR end the gap is about 5 %, partly the 1/K Monte-Carlo term of a 32-sample mean, and the one-call C1 is closer ('+fx(F0.C1_60,3)+' % against D’s '+fx(F0.D_60,3)+' % at 60 dB). From 30 dB down the gap opens to '+fx(100*(c['30.0'].ratio-1),0)+'–'+fx(100*(c['20.0'].ratio-1),0)+' %. A starred B marker is the page’s paired preview on the 50 cubes, not a verdict.';};
SL.dbest=function(h){h.textContent=F0.D_best_every_snr?'the most accurate method at every SNR':'not the most accurate method at every SNR';};
SL.e1B=function(h){const have=[60,50,40,30,20].filter(function(x){return bOk(set('device1','sequential',x));});
  if(!have.length){h.innerHTML='<b>Line B:</b> its E1 records with the x₀-prediction prior are pending.';return;}
  const parts=have.map(function(x){const b=set('device1','sequential',x).B;return x+' dB: '+fx(b.means.mean_rmse_pct,2)+' % against D’s '+fx(b.on_subset.D.rmse_pct,2)+' % on the same cubes (cov90 '+fx(b.means.cov90,2)+')';});
  const pend_=[60,50,40,30,20].filter(function(x){return have.indexOf(x)<0;});
  const under=have.filter(function(x){return set('device1','sequential',x).B.means.cov90<0.85;}).length;
  h.innerHTML='<b>Line B with the retrained x₀-prediction prior</b> ('+have.length+' of 5 E1 records present): '+parts.join('; ')+'. '+(under?'Its samples are under-dispersed at '+under+' of these SNRs (cov90 below 0.85). ':'')+(pend_.length?'Pending: '+pend_.join(', ')+' dB.':'');};
SL.t2marks=function(h){h.innerHTML=T2S.map(function(x){const c=V.T2&&V.T2.C&&V.T2.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T2.B&&V.T2.B[vk(x)];
  return '<span><b>T2</b> '+x+' dB '+(c?pill(c.passed,'C '+(c.passed?'PASS':'FAIL')):pend('C'))+' '+(vb?pill(vb.passed,'B '+(vb.passed?'PASS':'FAIL')):(bOk(s)?'<span class="pill wn">B preview</span>':pend('B pending')))+'</span>';}).join('');};
SL.t2text=function(h){const c=V.T2&&V.T2.C||{};const a=c['30.0'],b=c['20.0'];if(!a||!b){h.textContent='T2 for C is pending.';return;}const band=V.thresholds.T2_COV90;
  h.innerHTML='L, V and C on the 200 gate cubes; B on its 50. At 30 dB C '+(a.passed?'passes':'fails')+' T2: CRPS − L = '+sg(a.crps_vs_L.mean,3)+' ± '+fx(a.crps_vs_L.se_cluster,3)+', − V = '+sg(a.crps_vs_V.mean,3)+' ± '+fx(a.crps_vs_V.se_cluster,3)+' (clustered), cov90 '+fx(a.cov90,3)+'. At 20 dB both CRPS differences are clear ('+sg(b.crps_vs_L.mean,3)+' ± '+fx(b.crps_vs_L.se_cluster,3)+' and '+sg(b.crps_vs_V.mean,3)+' ± '+fx(b.crps_vs_V.se_cluster,3)+'; C’s CRPS is '+fx(100*(1-b.crps_vs_L.ratio_of_means),0)+' % below L’s), but cov90 = '+fx(b.cov90,3)+(b.cov90_passed?' lies inside the band':' lies '+fx(b.cov90-band[1],3)+' above the band')+', so the clause '+(b.passed?'passes':'fails')+(b.passed?'':' on coverage only')+'.';};
SL.covtext=function(h){const c20=met(set('device1','sequential',20),'C','cov50'),c30=met(set('device1','sequential',30),'C','cov50');const b30=set('device1','sequential',30),b20=set('device1','sequential',20);
  let t='C is slightly over-dispersed, and more so as the SNR falls (cov50 '+fx(c30,3)+' at 30 dB, '+fx(c20,3)+' at 20 dB, against 0.50 nominal). The variance head is the best calibrated of the four.';
  if(bOk(b30)||bOk(b20)){const xs=[30,20].filter(function(x){return bOk(set('device1','sequential',x));});t+=' B: '+xs.map(function(x){const m=set('device1','sequential',x).B.means;return 'cov50 '+fx(m.cov50,2)+' / cov90 '+fx(m.cov90,2)+' at '+x+' dB';}).join('; ')+'.';}else t+=' B’s 30 and 20 dB records are pending.';
  h.textContent=t;};
SL.t3text=function(h){const c=V.T3&&V.T3.C;const th=V.thresholds;if(!c){h.textContent='T3 pending.';return;}
  let t='L, D and C on the 200 gate cubes; B on its 50, where present. T3 for C '+(c.passed?'passes':'fails')+': C’s RMSE ratio to the MMSE is '+fx(c.ratio_dev1,3)+' on Device 1 and '+fx(c.ratio_dev2,3)+' on Device 2, against the limit '+fx(th.T3_RATIO,2)+' × '+fx(c.ratio_dev1,3)+' = '+fx(th.T3_RATIO*c.ratio_dev1,3)+'. On PH5 (report-only) the ratio is '+fx(c.ph5,3)+'.';
  const s1=set('device1','sequential',30),s2=set('device2','sequential',30);
  if(V.b_valid&&V.T3.B)t+=' T3 for B '+(V.T3.B.passed?'passes':'fails')+' (ratios '+fx(V.T3.B.ratio_dev1,3)+' and '+fx(V.T3.B.ratio_dev2,3)+').';
  else if(bOk(s1)&&bOk(s2))t+=' B (page preview on the 50 cubes): ratio '+fx(s1.B.paired.rmse_vs_L.ratio,3)+' on Device 1, '+fx(s2.B.paired.rmse_vs_L.ratio,3)+' on Device 2.';
  else t+=' T3 for B waits for its 30 dB records on '+[[s1,'Device 1'],[s2,'Device 2']].filter(function(q){return !bOk(q[0]);}).map(function(q){return q[1];}).join(' and ')+'.';
  h.textContent=t;};
SL.d2D=function(h){h.textContent=[40,30,20].map(function(x){return fx(-F0['device2_D_vs_L_'+x],0)+' %';}).join(' / ')+' below the MMSE at 40 / 30 / 20 dB (on Device 1: '+[40,30,20].map(function(x){return fx(F0['D_below_L_'+x],0)+' %';}).join(' / ')+').';};
SL.e2B=function(h){const xs=[40,30,20].filter(function(x){return bOk(set('device2','sequential',x));});
  if(!xs.length){h.innerHTML='<b>Line B on Device 2:</b> pending. B needs no retraining for any sensor by construction, so its transfer test is only whether its error ratio to the MMSE holds.';return;}
  h.innerHTML='<b>Line B on Device 2</b> ('+xs.length+' of 3 records): '+xs.map(function(x){const b=set('device2','sequential',x).B;return x+' dB: '+fx(b.means.mean_rmse_pct,2)+' % against L’s '+fx(b.on_subset.L.mean_rmse_pct,2)+' % on the same cubes';}).join('; ')+'.';};
SL.t4text=function(h){const c=V.T4&&V.T4.C;if(!c){h.textContent='T4 pending.';return;}
  let t='L, L demosaic, D and C on the 200 gate cubes; B on its 50. The dashed segment at 30 dB is T4’s target. T4 for C '+(c.passed?'passes':'fails')+': ratio '+fx(c.ratio,3)+' (2-SE bound '+fx(c.ratio_2se,3)+'). C’s posterior mean ('+fx(F0.e3_C_30,2)+' %) is '+(c.ratio>1?'worse than':'better than')+' demosaic-then-MMSE ('+fx(F0.Ldem_30,2)+' %).'+(c.D?' The regressor passes the same clause '+(c.D.passed?'easily':'—')+': '+fx(F0.e3_D_30,2)+' %, ratio '+fx(c.D.ratio,3)+', at every SNR below the baseline.':'');
  const s4=set('device1','snapshot',30);if(V.b_valid&&V.T4.B)t+=' T4 for B '+(V.T4.B.passed?'passes':'fails')+' (ratio '+fx(V.T4.B.ratio,3)+').';else if(bOk(s4))t+=' B (preview): ratio '+fx(s4.B.paired.rmse_vs_L_demosaic.ratio,3)+' to the demosaic baseline on its 50 cubes.';else t+=' T4 for B waits for its 30 dB snapshot record.';
  h.textContent=t;};
SL.e3B=function(h){const xs=[60,40,30,20].filter(function(x){return bOk(set('device1','snapshot',x));});
  if(!xs.length){h.innerHTML='<b>Line B in snapshot mode</b> (zero-shot, masked operator): pending.';return;}
  h.innerHTML='<b>Line B in snapshot mode</b> (zero-shot, masked operator; '+xs.length+' of 4 records): '+xs.map(function(x){const b=set('device1','snapshot',x).B;const p=b.paired.rmse_vs_L_demosaic;return x+' dB: '+fx(b.means.mean_rmse_pct,2)+' % against demosaic-then-MMSE '+fx(p.b,2)+' % on the same cubes (ratio '+fx(p.ratio,3)+'), cov50 / cov90 '+fx(b.means.cov50,2)+' / '+fx(b.means.cov90,2);}).join('; ')+'.';};
SL.probetext=function(h){const P=D.b_prior_probe;const g=function(k){return P.series.filter(function(s){return s.key===k;})[0].err;};const i299=P.t.indexOf(299);const v=g('vpred'),bl=g('blind'),x0=g('x0');
  const worse=P.t.filter(function(t,i){return v[i]>P.mean_cube;});
  h.textContent='The v-prediction prior (epoch 100) tracks the input-blind predictor E[v | t] to within a point for t ≤ 200, and is worse than predicting the mean cube ('+fx(P.mean_cube,2)+' %) at t = '+worse.join(', ')+'. Neither uniform loss weighting nor a time-token path moves the curve. With x₀-prediction the low-t target is close to the identity, which forces the network to read x_t: the retrained prior’s error is '+fx(x0[0],1)+' % at t = '+P.t[0]+' and '+fx(x0[i299],1)+' % at t = 299, where the v-prediction prior had '+fx(v[i299],1)+' % and the blind predictor '+fx(bl[i299],1)+' %. Its error grows monotonically and stays below the mean cube at every t. Unguided DDIM-50 samples saturated '+fx(P.saturation_pct.vpred,1)+' % of their elements with the v-prediction prior and '+fx(P.saturation_pct.x0,1)+' % with the x₀-prediction prior.';};
SL.toytext=function(h){const t=D.toy;if(!t){h.textContent='The toy record was not found at build time.';return;}const r=t.rows.filter(function(r){return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});
  h.textContent='Before Stage 0, ΠGDM had to reproduce the exact Gaussian posterior on an isotropic toy with the real Device 1 operator (1 000 draws per cell, 30 dB). At '+t.steps_for_B+' steps it meets every gate in both modes ('+r.map(function(q){return q.mode+': mean excess '+fx(q.rmse_excess,4)+', variance ratio '+fx(q.var_ratio,3)+', cov90 '+fx(q.cov90,3);}).join('; ')+'), so B runs at '+t.steps_for_B+' steps as pre-registered. On the correlated HFD covariance (report only) the isotropic approximation costs about 4 % in RMSE and under-disperses in sequential mode. That is the first explanation to test if B’s calibration clauses fail.';};
SL.bnow=function(h){const have=ST.b_have.length;if(!have){h.innerHTML='No x₀-prior B record yet: the B story ends at the prior.';return;}
  const keys=ST.b_have;const under=keys.filter(function(k){return S[k].B.means.cov90<0.85;});
  const worseD=keys.filter(function(k){const p=S[k].B.paired;const q=S[k].mode==='snapshot'?p.rmse_vs_L_demosaic:p.rmse_vs_D;return q&&q.diff-2*q.se_cluster>0;});
  h.innerHTML='<b>Where B stands ('+have+' of '+ST.b_expected.length+' records).</b> With the x₀-prediction prior the samples are no longer degenerate (single-sample RMSE '+keys.map(function(k){return fx(S[k].B.means.single_rmse_pct,1)+' %';}).join(', ')+' against the v-prediction prior’s '+Object.keys(D.b_vpred).map(function(k){return fx(D.b_vpred[k].means.single_rmse_pct,1)+' %';}).join(', ')+'). '+
    (worseD.length?'Its mean-of-32 is still worse than the reference method (D; in snapshot mode, demosaic-then-MMSE) by more than 2 clustered SE in '+worseD.length+' of '+have+' settings. ':'')+(under.length?'Its samples are under-dispersed (cov90 below 0.85) in '+under.length+' of '+have+'. ':'')+(ST.b_pending.length?'The remaining '+ST.b_pending.length+' records are pending, and the verdict on B waits for them.':'All B records are in.');};
SL.verdictnote=function(h){h.innerHTML='Verdict <code>'+ST.verdict_file+'</code>, created '+(V.created||'—')+(V.report_only?', report-only':'')+'. Main line: '+(V.main_line?'<b>'+V.main_line+'</b>':'not named (the verdict needs B)')+'. '+(V.b_valid?'':'B cells marked “preview” are the page’s own paired computation on the 50-cube subset, using the verdict’s rule and thresholds; they become verdict cells when the final verdict that reads the x₀-prior records is built ('+V.b_invalid_reason+').')+(V.missing&&V.missing.length?' Settings the verdict lists as missing: '+V.missing.length+'.':'');};
SL.budget=function(h){const b=D.budget,e=b.eval_hours;h.innerHTML='<b>Budget.</b> Training '+fx(b.train_hours,1)+' GPU-h for the five runs (the failed v-prediction prior included), plus '+fx(b.pilot_hours,1)+' GPU-h for the three prior pilots. Evaluation so far: '+fx(e.main,2)+' GPU-h for the L / D / V records, '+fx(e.C,1)+' for C, '+fx(e.B_vpred,1)+' for the two archived v-prediction B records and '+fx(e.B,1)+' for the '+b.b_records+' x₀-prior B records present. A B record costs ≈ '+fx(b.b_hours_mean,2)+' GPU-h, so the '+b.b_expected+' planned B records come to ≈ '+fx(b.b_hours_projected,0)+' GPU-h, against the spec’s 12 GPU-h cap for B.';};

/* ---------- data-bound values ---------- */
function resolve(path){return path.split('.').reduce(function(o,k){return o==null?undefined:o[k];},D);}
document.querySelectorAll('[data-v]').forEach(function(h){const sp=h.dataset.v.split(':');const v=resolve(sp[0]);const f=sp[1];
  if(v==null){h.innerHTML=pend();return;}
  if(f==='e')h.textContent=sci(v);else if(f!=null&&typeof v==='number')h.textContent=Number(v).toFixed(+f);else h.textContent=String(v);});
document.querySelectorAll('[data-fig]').forEach(function(h){const f=F[h.dataset.fig];if(f)f(h);});
document.querySelectorAll('[data-chart]').forEach(function(h){const f=CH[h.dataset.chart];if(f)f(h);});
document.querySelectorAll('[data-table]').forEach(function(h){const f=T[h.dataset.table];if(f)f(h);});
document.querySelectorAll('[data-slot]').forEach(function(h){const f=SL[h.dataset.slot];if(f)f(h);});
