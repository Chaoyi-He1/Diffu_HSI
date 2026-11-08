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
    series:ms.map(function(m){return {name:m==='B'?'B (50 cubes)':m,c:MC[m]};}),
    groups:T2S.map(function(x){const s=set('device1','sequential',x);return {label:x+' dB',values:ms.map(function(m){return met(s,m,'ause');})};})});};
CH.t2cov=function(h){const ms=['L','V','C','B'];const W=520,H=300,m={l:46,r:12,t:16,b:50};const svg=mk(h,W,H,380);const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const Y=function(v){return m.t+ph-(v-0)/(1.0-0)*ph;};
  el('rect',{x:m.l,y:Y(0.95),width:pw,height:Y(0.85)-Y(0.95),style:'fill:var(--good-bg)'},svg);
  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,t.toFixed(1),'tick','end');});
  line(svg,m.l,Y(0.9),m.l+pw,Y(0.9),'refacc');line(svg,m.l,Y(0.5),m.l+pw,Y(0.5),'refacc');
  tx(svg,4,12,'coverage','axt');
  const slots=[];T2S.forEach(function(x){ms.forEach(function(mm){slots.push([x,mm]);});});const sw=pw/slots.length;
  T2S.forEach(function(x,gi){tx(svg,m.l+sw*(gi*ms.length+ms.length/2),H-m.b+36,x+' dB','bs','middle');if(gi)line(svg,m.l+sw*gi*ms.length,m.t,m.l+sw*gi*ms.length,m.t+ph,'axis');});
  const rows=[];
  slots.forEach(function(q,i){const s=set('device1','sequential',q[0]);const cx=m.l+sw*(i+0.5);tx(svg,cx,H-m.b+18,q[1],'tick','middle');
    const c50=met(s,q[1],'cov50'),c90=met(s,q[1],'cov90');
    if(c90==null){tx(svg,cx,Y(0.5),'pending','sm','middle').setAttribute('transform','rotate(-90 '+cx+' '+Y(0.5)+')');rows.push([q[0],q[1],'pending','pending']);return;}
    const a=el('circle',{cx:cx,cy:Y(c90),r:5.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    const b=el('rect',{x:cx-4.5,y:Y(c50)-4.5,width:9,height:9,rx:1.5,class:'dot',style:'fill:var('+MC[q[1]]+')'},svg);
    [a,b].forEach(function(n){hover(n,function(){return '<div><b style="margin:0;padding:0">'+q[1]+' · '+q[0]+' dB</b></div><div class="sub">cov50 '+c50.toFixed(3)+' · cov90 '+c90.toFixed(3)+'</div>';});});
    rows.push([q[0],q[1],c50.toFixed(3),c90.toFixed(3)]);});
  dataTable(h,['SNR','method (B on 50 cubes)','cov50','cov90'],rows);};
CH.e2=function(h){const ms=['L','D','C','B'];
  function draw(x){clear(h);const prevLg=h.parentNode.querySelector('.legend');if(prevLg)prevLg.remove();
    bars(h,{y:[0,25],ticks:[0,5,10,15,20,25],fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(2);},yTitle:'RMSE % ('+x+' dB)',gname:'sensor',
      series:ms.map(function(m){return {name:{L:'L, MMSE',D:'D, regressor',C:'C, mean of 32',B:'B, mean of 32 (50 cubes)'}[m],c:MC[m]};}),
      groups:['device1','device2','ph5'].map(function(dev){const s=set(dev,'sequential',x);return {label:DEVN[dev]+(dev==='device1'?' (trained)':''),values:ms.map(function(m){return rmse(s,m);}),tags:ms.map(function(m){return m==='B'?(bStatusTxt(s)||'pending'):null;})};})});}
  draw('30');seg('e2',draw);};
CH.e3rmse=function(h){const snrs=[20,30,40,60];const ms=['L','L_demosaic','D','C','C1','B'];
  legendInto('e3',ms.map(function(m){const has=snrs.some(function(x){return rmse(set('device1','snapshot',x),m)!=null;});return {n:(m==='L'?'L, exact per-pixel posterior':MN[m])+(m==='B'&&!has?' — pending':''),c:MC[m],dash:m==='C1'};}));
  const ser=ms.map(function(m){return {name:m==='L'?'L exact':MN[m],c:MC[m],dash:m==='C1',label:{L:'L exact',L_demosaic:'L demosaic',D:'D',C:'C',B:'B'}[m],ldy:{C:-2,L_demosaic:7,B:18}[m],data:snrs.map(function(x){const v=rmse(set('device1','snapshot',x),m);return v==null?null:[x,v];}).filter(Boolean)};});
  const t=D.facts.Ldem_target_30;
  xy(h,{log:true,x:[9,62],y:[2,16],xTicks:[20,30,40,60],xFmt:function(v){return v===60?'60 (floor)':String(v);},yTicks:[2,3,5,8,12,16],yTitle:'RMSE % (log)',xTitle:'SNR (dB)',labelAt:'first',mr:20,ml:56,series:ser,tipX:function(v){return v+' dB';},
    refs:[{y:t,x0:26,x1:34,lx:35,label:'T4 target '+fx(t,2),cls:'ref'}],
    table:{head:['setting','L exact','L demosaic','D','C mean-of-32','C1','C single','B (50)'],rows:[['device1',60],['device1',40],['device1',30],['device1',20],['device2',30]].map(function(q){const x=q[1];const s=set(q[0],'snapshot',x);if(!s)return [q[0]+' '+x,'—','—','—','—','—','—','pending'];return [(q[0]==='device2'?'E4 Device 2, ':'E3, ')+x+' dB',fx(rmse(s,'L'),3),fx(rmse(s,'L_demosaic'),3),fx(rmse(s,'D'),3),fx(rmse(s,'C'),3),fx(rmse(s,'C1'),3),fx(met(s,'C','single_rmse_pct'),3),bOk(s)?fx(rmse(s,'B'),3):bStatusTxt(s)];})}});};
CH.bprobe=function(h){const P=D.b_prior_probe;const cs={blind:['--ink-2',true,false,'7 4'],vpred:['--s2',false,false],g0:['--muted',false,false],tt:['--muted',true,false,'2 3'],x0pilot:['--accent',false,false],x0:['--s4',false,false]};
  legendInto('probe',P.series.map(function(s){return {n:s.name,c:cs[s.key][0],dash:cs[s.key][1],faint:cs[s.key][2]};}).concat([{n:'mean-cube predictor (thin dashed rule)',c:'--muted',dash:true}]));
  xy(h,{log:true,x:[0,920],y:[2,32],xTicks:[25,100,200,300,400,500,700,900],yTicks:[2,3,5,10,20,30],yTitle:'x̂₀ error % (log)',xTitle:'diffusion timestep t',mr:20,
    series:P.series.map(function(s){return {name:s.name,c:cs[s.key][0],dash:cs[s.key][1],da:cs[s.key][3],faint:cs[s.key][2],thin:s.key==='g0'||s.key==='tt',data:P.t.map(function(t,i){return [t,s.err[i]];})};}),
    refs:[{y:P.mean_cube,label:'mean cube '+fx(P.mean_cube,2),lx:520,anchor:'start',dy:-6}],tipX:function(v){return 't = '+v;},tipY:function(v){return v.toFixed(1)+' %';},
    table:{head:['t'].concat(P.series.map(function(s){return s.name;})),rows:P.t.map(function(t,i){return [t].concat(P.series.map(function(s){return s.err[i].toFixed(1);}));})}});};

/* ---------- tables ---------- */
const T={};
T.e3unc=function(h){const ms=['L','L_demosaic','V','C','B'];const nm={L:'L exact',L_demosaic:'L demosaic',V:'V',C:'C',B:'B (50)'};
  const rowsK=[['device1',60],['device1',40],['device1',30],['device1',20],['device2',30]];
  tbl(h,['setting'].concat(ms.map(function(m){return nm[m];})),rowsK.map(function(q){const s=set(q[0],'snapshot',q[1]);return [(q[0]==='device2'?'E4 Device 2 · ':'E3 · ')+q[1]+' dB'].concat(ms.map(function(m){if(!s)return pend();const c=met(s,m,'crps');if(c==null)return pend(bStatusTxt(s)||'pending');return num(fx(c,2)+'<br>'+fx(met(s,m,'cov50'),2)+'/'+fx(met(s,m,'cov90'),2)+'<br>'+fx(met(s,m,'ause'),2));}));}),function(i){return i===2;});
  const p=document.createElement('p');p.style.fontSize='14px';p.style.color='var(--muted)';p.textContent='Each cell: CRPS %, then cov50 / cov90, then AUSE. The 30 dB row (highlighted) is T4’s setting; E4 is report-only.';h.appendChild(p);};
T.toy=function(h){const t=D.toy;if(!t){h.innerHTML=pend('toy record not found');return;}const g=t.gate;
  const inb=function(r){return r.rmse_excess<=g.rmse_excess&&Math.abs(r.var_ratio-1)<=g.var_tol&&r.cov90>=g.cov90[0]&&r.cov90<=g.cov90[1]&&r.cov50>=g.cov50[0]&&r.cov50<=g.cov50[1];};
  const rk=function(r){return (r.C_kind==='isotropic'?0:2)+(r.method==='pigdm'?0:1);};const rows=t.rows.slice().sort(function(a,b){return rk(a)-rk(b)||(a.mode>b.mode)-(a.mode<b.mode)||a.n_steps-b.n_steps;});
  tbl(h,['prior','method','mode','steps','RMSE excess','variance ratio','cov50','cov90','CRPS ratio','toy gate'],rows.map(function(r){return [r.C_kind==='isotropic'?'isotropic N(0, I)':'HFD covariance (report only)',r.method==='pigdm'?'ΠGDM':'DPS',r.mode,num(r.n_steps),num(fx(r.rmse_excess,4)),num(fx(r.var_ratio,3)),num(fx(r.cov50,3)),num(fx(r.cov90,3)),num(fx(r.crps_ratio,4)),r.C_kind==='isotropic'&&r.method==='pigdm'?pill(inb(r),inb(r)?'met':'missed'):'—'];}),function(i){const r=rows[i];return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});};
function bRow(label,s,b,exp){if(!b||b.status!=='final'&&b.status!=='archived'&&b.status!=='stale'){return [label,b&&b.status==='dropped'?pend('dropped'):pend(),'','','','','',''];}
  const mm=b.means,sub=b.on_subset,p=b.paired;const ref=s.mode==='snapshot'?'L_demosaic':'D';const pr=p['rmse_vs_'+ref];const keyRef=ref==='D'?'rmse_pct':'mean_rmse_pct';
  return [label+(b.status==='final'?' <span class="pill nt">x₀</span>':' <span class="pill nt">v-pred</span>'),num(fx(mm.mean_rmse_pct,2)+' / '+fx(mm.single_rmse_pct,1)),num(fx(mm.crps,2)),num(fx(mm.cov50,2)+' / '+fx(mm.cov90,2)),num(sg(mm.single_hf,1)),
    num(s.mode==='snapshot'?((sub.L?fx(sub.L.mean_rmse_pct,2):'—')+' / '+(sub.L_demosaic?fx(sub.L_demosaic.mean_rmse_pct,2):'—')+' / '+(sub.D?fx(sub.D.rmse_pct,2):'—')+' / '+(sub.C?fx(sub.C.mean_rmse_pct,2):'—')):((sub.L?fx(sub.L.mean_rmse_pct,2):'—')+' / '+(sub.D?fx(sub.D.rmse_pct,2):'—')+' / '+(sub.C?fx(sub.C.mean_rmse_pct,2):'—'))),
    num(pr?sg(pr.diff,2)+' ± '+fx(pr.se_cluster,2)+' ('+fx(pr.ratio,2)+'×)':'—'),num(fx(b.hours,2))];}
T.brec=function(h){const rows=[];
  [['E1','device1','sequential',[60,50,40,30,20]],['E2','device2','sequential',[40,30,20]],['E3','device1','snapshot',[60,40,30,20]],['E4','device2','snapshot',[30]]].forEach(function(e){e[3].forEach(function(x){const s=set(e[1],e[2],x);if(!s)return;const lab=e[0]+' · '+DEVN[e[1]]+(e[2]==='snapshot'?' snapshot':'')+' · '+x+' dB';
    if(e[0]==='E1'&&D.b_vpred[String(x)])rows.push(bRow(lab,s,D.b_vpred[String(x)],e[0]));rows.push(bRow(lab,s,s.B,e[0]));});});
  tbl(h,['setting · prior','B RMSE mean-of-32 / single (%)','B CRPS','B cov50 / cov90','B single log HF','L / D / C on the same cubes (snapshot: L exact / L demosaic / D / C)','B − D (snapshot: − L demosaic) ± cl. SE (ratio)','GPU-h'],rows);};
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
  const t3=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">ratio to L: Device 1 '+fx(c.ratio_dev1,3)+' · Device 2 '+fx(c.ratio_dev2,3)+' (limit '+fx(th.T3_RATIO*c.ratio_dev1,3)+') · PH5 '+(c.ph5==null?'not run':fx(c.ph5,3)+' (report)')+'</span>',cls:''};};
  let t3p=null;const s1=set('device1','sequential',30),s2=set('device2','sequential',30);if(bOk(s1)&&bOk(s2)){const r1=s1.B.paired.rmse_vs_L.ratio,r2=s2.B.paired.rmse_vs_L.ratio;t3p={v:prev(r2<=th.T3_RATIO*r1)+' <span class="num">ratio to L: Device 1 '+fx(r1,3)+' · Device 2 '+fx(r2,3)+' (limit '+fx(th.T3_RATIO*r1,3)+')</span>',cls:''};}
  rows.push(['<b>T3</b> · 30 dB','(B or C / L) on Device 2 ≤ '+fx(th.T3_RATIO,2)+' × the same on Device 1',t3(V.T3&&V.T3.C)||pend(),bClause(t3(V.T3&&V.T3.B),t3p)]);
  const t4=function(c){if(!c)return null;return {v:pill(c.passed)+' <span class="num">ratio '+fx(c.ratio,3)+' · 2-SE bound '+fx(c.ratio_2se,3)+' (limit '+fx(th.T4_RATIO,2)+')'+(c.D?'; D: '+(c.D.passed?'pass':'fail')+' at '+fx(c.D.ratio,3):'')+'</span>',cls:''};};
  const s4=set('device1','snapshot',30);rows.push(['<b>T4</b> · snapshot 30 dB','RMSE ≤ '+fx(th.T4_RATIO,2)+' × demosaic-then-MMSE (2 cl. SE)',t4(V.T4&&V.T4.C)||pend(),bClause(t4(V.T4&&V.T4.B),bOk(s4)?previewT1(s4.B.paired.rmse_vs_L_demosaic,th.T4_RATIO):null)]);
  const sc=V.sanity||{};const fp=sc.floor_pass||{};const mo=sc.monotone||{};
  const seqC=E1S.map(function(x){return fx(rmse(set('device1','sequential',x),'C'),3);}).join(' / '),seqB=E1S.map(function(x){return fx(rmse(set('device1','sequential',x),'B'),3);}).join(' / ');
  const sanCell=function(line,seq){const ok=mo[line]&&fp[line];return {v:pill(ok)+' <span class="num">monotone '+(mo[line]?'yes':'no')+' ('+seq+' % at 60→20 dB) · floor '+(fp[line]?'pass':'fail')+' ('+fx(rmse(set('device1','sequential',60),line),3)+' vs '+fx(sc.floor_reference_rmse_pct,3)+' %)</span>'+(line==='B'&&!mo.B?'<br><span style="font-size:13px;color:var(--muted)">prior-limited: flat within noise, +'+fx(F0.B_e1_rmse['30']-F0.B_e1_rmse['40'],2)+' from 40 to 30 dB</span>':''),cls:''};};
  rows.push(['<b>sanity</b>','RMSE non-increasing in SNR; at 60 dB ≤ '+fx(sc.floor_reference_rmse_pct,3)+' % + 2 SE',sanCell('C',seqC),sanCell('B',seqB)]);
  const pc=function(line){const p=passes(line);return num('<b>'+p.length+' of 5</b> ('+(joinAnd(p)||'none')+')');};
  rows.push(['<b>clauses passed</b>','T1 once (all five SNRs), T2 per SNR, T3, T4: at most 5; tie → lower paired CRPS − V at 20 dB',pc('C'),V.b_valid?pc('B'):bInvalid()]);
  const mlB=function(){if(!V.b_valid)return bInvalid();return num(V.main_line==='B'?'<b>B</b> (tie-break: '+tieInfo(3).short+')':'—');};
  rows.push(['<b>main line</b>','',num(!V.b_valid?pend():(V.main_line==='C'?'<b>C</b>':'—')),mlB()]);
  tbl(h,['clause · setting','rule','line C (200 cubes)','line B (50 cubes)'],rows,function(i){return i>=rows.length-2;});};
T.runs=function(h){const rows=D.runs.map(function(r){return [r.run+(r.failed?' <span class="pill no">failed prior</span>':''),r.role,(r.prediction_type||'v')+' · '+(r.loss_type||'')+(r.snr_gamma&&r.prediction_type!=='direct'?' · min-SNR γ '+r.snr_gamma:''),num(r.epochs+' / '+r.planned),num(fx(r.sec_per_epoch,0)),num(fx(r.hours,2)),num(String(r.selected)),num(fx(r.val_at_sel,5))];});
  D.pilots.forEach(function(p){rows.push([p.run,'prior pilot',(p.prediction_type||'v')+' · '+(p.loss_type||'')+(p.snr_gamma!=null?' · min-SNR γ '+p.snr_gamma:''),num(p.epochs+' / '+p.planned),num(fx(p.sec_per_epoch,0)),num(fx(p.hours,2)),num('—'),num('—')]);});
  tbl(h,['run','role','parameterisation · loss','epochs','s / epoch','GPU-h','selected epoch','monitoring loss there'],rows);};

/* ---------- slots ---------- */
const SL={};
function bList(keys){return keys.map(function(k){const s=S[k];return (s.exp+' '+DEVN[s.device]+(s.mode==='snapshot'?' snapshot':'')+' '+s.snr+' dB');}).join(', ');}
SL.normcheck=function(h){const sc=D.stage0.shape_check;h.textContent='The normalisation check '+(D.stage0.normalisation_check_passed?'passed':'FAILED')+': with the raw (un-normalised) response the curve keeps its shape within the 10 % rule (largest deviation '+fx(100*sc.sequential.max_dev,1)+' % sequential, '+fx(100*sc.snapshot.max_dev,1)+' % snapshot), so the normalised R is the paper’s sensor and the raw curve (dashed) goes to the appendix.';};
SL.angles=function(h){const a=D.stage0.principal_angles;h.innerHTML='<b>Principal angles between the sensors’ column spaces.</b> Device 1 and Device 2 share a '+F0.ang_d1d2_shared+'-dimensional subspace: only '+F0.ang_d1d2_nonzero+' of the '+a.device1_device2.n+' angles is non-zero ('+fx(F0.ang_d1d2_max,1)+'°). The transfer test is therefore about how the two sensors weight directions, not about different subspaces. PH5’s response has rank '+F0.ph5_rank+', and every angle to Device 1’s space is at most '+fx(F0.ang_d1ph5_max,1)+'°, so PH5 measures a '+F0.ph5_rank+'-dimensional part of what Device 1 measures, with far worse conditioning.';};
SL.t1marks=function(h){const th=V.thresholds||{};h.innerHTML='<span><b>T1</b> (limit '+fx(th.T1_RATIO,2)+'):</span>'+[60,50,40,30,20].map(function(x){const c=V.T1&&V.T1.C&&V.T1.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T1.B&&V.T1.B[vk(x)];
    let b;if(vb)b=pill(vb.passed,'B '+fx(vb.ratio,2));else if(bOk(s))b='<span class="pill wn">B '+fx(s.B.paired.rmse_vs_D.ratio,2)+' (preview)</span>';else b=pend('B pending');
    return '<span>'+x+' dB '+(c?pill(c.passed,'C '+fx(c.ratio,2)):pend('C'))+' '+b+'</span>';}).join('');};
SL.t2marks=function(h){h.innerHTML=T2S.map(function(x){const c=V.T2&&V.T2.C&&V.T2.C[vk(x)];const s=set('device1','sequential',x);const vb=V.b_valid&&V.T2.B&&V.T2.B[vk(x)];
  return '<span><b>T2</b> '+x+' dB '+(c?pill(c.passed,'C '+(c.passed?'PASS':'FAIL')):pend('C'))+' '+(vb?pill(vb.passed,'B '+(vb.passed?'PASS':'FAIL')):(bOk(s)?'<span class="pill wn">B preview</span>':pend('B pending')))+'</span>';}).join('');};
SL.d2D=function(h){h.textContent=[40,30,20].map(function(x){return fx(-F0['device2_D_vs_L_'+x],0)+' %';}).join(' / ')+' below the MMSE at 40 / 30 / 20 dB (on Device 1: '+[40,30,20].map(function(x){return fx(F0['D_below_L_'+x],0)+' %';}).join(' / ')+').';};
SL.toytext=function(h){const t=D.toy;if(!t){h.textContent='The toy record was not found at build time.';return;}const r=t.rows.filter(function(r){return r.C_kind==='isotropic'&&r.method==='pigdm'&&r.n_steps===t.steps_for_B;});
  h.textContent='Before Stage 0, ΠGDM had to reproduce the exact Gaussian posterior on an isotropic toy with the real Device 1 operator (1 000 draws per cell, 30 dB). At '+t.steps_for_B+' steps it meets every gate in both modes ('+r.map(function(q){return q.mode+': mean excess '+fx(q.rmse_excess,4)+', variance ratio '+fx(q.var_ratio,3)+', cov90 '+fx(q.cov90,3);}).join('; ')+'), so B runs at '+t.steps_for_B+' steps as pre-registered. On the correlated HFD covariance (report only) the isotropic approximation costs about 4 % in RMSE and under-disperses in sequential mode. '+((passes('B').indexOf('T2 at 30 dB')<0&&passes('B').indexOf('T2 at 20 dB')<0)?'B’s coverage did fail T2 (section 3), so that approximation is the first explanation to test.':'That would be the first explanation to test if B’s calibration clauses failed.');};
/* verdict helpers: the verdict's own counting rule (sd_verdict._counts) */
const E1S=[60,50,40,30,20];
function passes(line){const out=[];const t1=V.T1&&V.T1[line]||{};if(E1S.every(function(x){return t1[vk(x)]&&t1[vk(x)].passed;}))out.push('T1');
  T2S.forEach(function(x){const c=V.T2&&V.T2[line]&&V.T2[line][vk(x)];if(c&&c.passed)out.push('T2 at '+x+' dB');});
  if(V.T3&&V.T3[line]&&V.T3[line].passed)out.push('T3');if(V.T4&&V.T4[line]&&V.T4[line].passed)out.push('T4');return out;}
function t2v(line,x){return V.T2&&V.T2[line]&&V.T2[line][vk(x)];}
/* B's verdict numbers are shown only while the verdict's B clauses are current (V.b_valid), as in the clause table */
function bInvalid(){return pend('B: verdict not current');}
function bInvalidWhy(){return 'the verdict <code>'+ST.verdict_file+'</code> read B records that are no longer current'+(V.b_invalid_reason?' ('+V.b_invalid_reason+')':'');}
/* the tie-break as the verdict computed it: on the cubes the tied lines share (V.tie_break), else each line's own T2 mean */
function tieInfo(n){const t=V.tie_break,tb=t2v('B',20),tc=t2v('C',20);
  if(t&&t.lines&&t.lines.B&&t.lines.C){const b=t.lines.B,c=t.lines.C;
    return {common:true,winner:t.winner,txt:'on the '+t.n_common+' cubes both lines share, B '+sg(b.mean_common,n)+' ± '+fx(b.se_cluster_common,n)+' against C '+sg(c.mean_common,n)+' ± '+fx(c.se_cluster_common,n)+' (clustered SE; on their own cubes B '+sg(b.mean_own,n)+' on '+b.n_own+', C '+sg(c.mean_own,n)+' on '+c.n_own+')',
      short:'B '+sg(b.mean_common,n)+' vs C '+sg(c.mean_common,n)+' on '+t.n_common+' common cubes'};}
  return {common:false,winner:V.main_line,txt:'B '+sg(tb.crps_vs_V.mean,n)+' on its 50 cubes against C '+sg(tc.crps_vs_V.mean,n)+' on the 200',
    short:sg(tb.crps_vs_V.mean,n)+' vs C '+sg(tc.crps_vs_V.mean,n)};}
function joinAnd(a){return a.length<2?a.join(''):a.slice(0,-1).join(', ')+' and '+a[a.length-1];}
function bestSnrs(m){const b=F0.e1_best_common||{};return E1S.filter(function(x){return b[String(x)]===m;}).sort(function(a,b){return b-a;});}
function rng(a){return a.length?(a.length>1?a[0]+'–'+a[a.length-1]:String(a[0]))+' dB':'no SNR';}
SL.status=function(h){const p=ST.pending;h.innerHTML=p.length?'<b>Not yet in the records at build time:</b> '+p.join(', ')+'. The cells that need them show “pending”.':'';};
SL.outcome=function(h){const cB=passes('B'),cC=passes('C');const tb=t2v('B',20);const ml=V.main_line;
  const cC40=[40,30,20].map(function(x){return met(set('device1','sequential',x),'C','cov90');});
  const bD=bestSnrs('D'),bB=bestSnrs('B'),bC=bestSnrs('C');
  const t2ok=function(line){return T2S.some(function(x){const c=t2v(line,x);return c&&c.passed;});};
  const demo=F0.e3_D_30!=null&&F0.Ldem_30!=null&&F0.e3_D_30<F0.Ldem_30;
  const ph5None=F0.ph5_D_worse_min!=null&&F0.ph5_C_ratio_min!=null&&F0.ph5_D_worse_min>0&&F0.ph5_C_ratio_min>1;
  const dItem='<li><b>Regressor D:</b> the best point estimate at high SNR ('+rng(bD)+' on the shared cubes) and the only method that demosaics: '+fx(F0.e3_D_30,2)+' % in snapshot at 30 dB against '+fx(F0.Ldem_30,2)+' % for demosaic-then-MMSE.</li>';
  const cItem='<li><b>C, conditional diffusion:</b> near-nominal coverage on its own sensor (cov90 '+fx(Math.min.apply(null,cC40),3)+'–'+fx(Math.max.apply(null,cC40),3)+' at 40–20 dB), and the lowest CRPS of any method whose coverage is near nominal at ≤ 40 dB ('+fx(F0.C_crps_gain_lo_min,0)+'–'+fx(F0.C_crps_gain_lo_max,0)+' % below L and V). It never matches the regressor’s mean.</li>';
  const dOpen='When the measurement is informative — high SNR'+(bD.length?' ('+rng(bD)+')':'')+', or snapshot mode on the sensor the networks were trained on — the one-call regressor D is the best point estimate'+(demo?' and the only method that demosaics':'')+'. ';
  const cOpen=bC.length?'The conditional diffusion C is the best point estimate at '+rng(bC)+'. ':(t2ok('C')&&!t2ok('B')?'The conditional diffusion C gives the best uncertainty of the calibrated methods on its own sensor (the only line to pass the uncertainty clause T2) but never the best mean. ':'The conditional diffusion C is never the best mean; its uncertainty is compared in section 3. ');
  const ph5Txt=ph5None?'Nothing learned on Device 1 transfers to PH5 (B was not run there).':'PH5’s transfer results are in section 4.';
  if(!V.b_valid){h.innerHTML='<p><b>No single method wins everywhere.</b> '+dOpen+cOpen+ph5Txt+'</p><ul>'+dItem+cItem+'</ul><p><b>Pre-registered verdict:</b> '+bInvalid()+' C passes '+cC.length+' ('+(joinAnd(cC)||'none')+') out of 5; '+bInvalidWhy()+', so B’s clauses, the pass counts and the main line wait for the regenerated verdict.</p>';return;}
  const r30=V.T1.B&&V.T1.B['30.0'],r20=V.T1.B&&V.T1.B['20.0'];const c30=t2v('B',30);const ti=tieInfo(2);
  const d2=set('device2','sequential',30);const d2ok=bOk(d2)&&d2.B.on_subset&&d2.B.on_subset.D;
  const bTrans=d2ok&&d2.B.means.mean_rmse_pct<d2.B.on_subset.D.rmse_pct;
  const bOpen=bB.length?'When noise dominates ('+rng(bB)+') the guided prior B becomes the best point estimate'+(bTrans?', and it is the only learned method that transfers to a second sensor without retraining':'')+'. ':'The guided prior B is the best point estimate at no SNR on the shared cubes. ';
  const tie=cB.length===cC.length?'the tie is broken by the lower paired CRPS − CRPS<sub>V</sub> at 20 dB ('+ti.txt+'), so the main line is <b>'+(ti.winner||ml||'—')+'</b>':'so the main line is <b>'+(ml||'—')+'</b>';
  h.innerHTML='<p><b>No single method wins everywhere.</b> '+dOpen+bOpen+cOpen+ph5Txt+'</p>'+
   '<ul>'+dItem+
   '<li><b>B, guided prior:</b> the best point estimate at '+rng(bB)+' (mean-of-32 '+fx(r30&&r30.ratio,3)+' / '+fx(r20&&r20.ratio,3)+' × the regressor at 30 / 20 dB on the shared cubes), and zero-shot on Device 2 ('+(d2ok?fx(d2.B.means.mean_rmse_pct,2)+' % at 30 dB against the regressor’s '+fx(d2.B.on_subset.D.rmse_pct,2)+' %':pend())+'), but over-confident everywhere.</li>'+
   cItem+'</ul>'+
   '<p><b>Pre-registered verdict</b> (<code>'+ST.verdict_file+'</code>)<b>:</b> B passes '+cB.length+' of 5 clauses ('+(joinAnd(cB)||'none')+'), C '+cC.length+' ('+(joinAnd(cC)||'none')+'); '+tie+
   (ml==='B'?' — won through the accuracy of its mean, not through calibration: its samples are over-confident (cov90 '+fx(c30&&c30.cov90,3)+' / '+fx(tb&&tb.cov90,3)+' at 30 / 20 dB against 0.90)':'')+'. Both lines are reported in full below.</p>';};
SL.kpiB=function(h){if(!V.b_valid){h.innerHTML=bInvalid();return;}const r=V.T1.B&&V.T1.B['20.0'];h.textContent=r?fx(r.ratio,3)+' ×':'—';};
SL.kpiBsub=function(h){if(!V.b_valid){h.textContent='the verdict’s B clauses are not current';return;}const r=V.T1.B&&V.T1.B['30.0'];h.textContent=r?(fx(r.ratio,3)+' × at 30 dB; T1 passes at 30 and 20 dB'):'';};
SL.countBC=function(h){if(!V.b_valid){h.innerHTML='B '+pend('not current')+' · C '+passes('C').length;return;}h.textContent='B '+passes('B').length+' · C '+passes('C').length;};
SL.countsub=function(h){if(!V.b_valid){h.textContent='C: '+(joinAnd(passes('C'))||'none')+' · B and the main line wait for the regenerated verdict';return;}
  h.textContent='B: '+(joinAnd(passes('B'))||'none')+' · C: '+(joinAnd(passes('C'))||'none')+' · main line '+(V.main_line||'—')+' by tie-break';};
SL.tbRuling=function(h){const t=V.tie_break;if(!t||!V.b_valid)return;   /* keeps the typed fallback sentence */
  h.textContent='The tie-break compares B and C on the '+t.n_common+' cubes both records share (paired CRPS − CRPS_V at 20 dB, clustered SE); each line’s own-cube value is kept beside it in the verdict.';};
SL.e1lead=function(h){const bD=bestSnrs('D'),bB=bestSnrs('B');h.textContent='The paper’s main figure. On the cubes B and the others share, the regressor D is the most accurate method at '+rng(bD)+' and B’s mean-of-32 at '+rng(bB)+'. B’s error barely moves with SNR, so it is limited by its prior rather than by the noise. Clause T1 asks whether the better diffusion line’s mean-of-32 RMSE stays within 5 % of the regressor’s, by 2 clustered SE.';};
SL.dbest=function(h){const c=V.T1.C;const bB=bestSnrs('B');
  h.innerHTML='<b>No single method is the best point estimate.</b> D is the most accurate method at '+rng(bestSnrs('D'))+' and halves the MMSE error at 20 dB ('+fx(F0.D_20,2)+' against '+fx(F0.L_20,2)+' %). B is the most accurate at '+rng(bB)+' on its 50 cubes ('+bB.map(function(x){return fx(V.T1.B[vk(x)].ratio,3)+' × D at '+x+' dB';}).join(', ')+'). C’s mean-of-32 never beats D. The spec predicted that C would pass T1 at every SNR, and it failed at every SNR: about '+fx(100*(c['60.0'].ratio-1),0)+' % behind at the high end, and '+fx(100*(c['30.0'].ratio-1),0)+'–'+fx(100*(c['20.0'].ratio-1),0)+' % from 30 dB down.';};
SL.t1text=function(h){const c=V.T1.C,b=V.T1.B;const bp=E1S.filter(function(x){return b[vk(x)]&&b[vk(x)].passed;});
  h.innerHTML='C fails T1 at all five SNRs; its ratio to D is '+E1S.map(function(x){return fx(c[vk(x)].ratio,3);}).join(' / ')+' at 60 / 50 / 40 / 30 / 20 dB. At the high-SNR end part of the gap is the 1/K Monte-Carlo term of a 32-sample mean, and the one-call C1 is closer ('+fx(F0.C1_60,3)+' % against D’s '+fx(F0.D_60,3)+' % at 60 dB). B passes at '+(bp.join(' and ')||'no SNR')+' dB, but T1 is counted once over all five SNRs, so neither line passes T1. B’s 2-SE bound at 30 dB is '+fx(b['30.0'].ratio_2se,3)+', a narrow margin from one training seed.';};
SL.e1crpscap=function(h){const b30=t2v('B',30),b20=t2v('B',20);
  h.innerHTML='At 60 dB the three 200-cube methods are level. From 40 dB down C’s CRPS is '+fx(F0.C_crps_gain_lo_min,0)+'–'+fx(F0.C_crps_gain_lo_max,0)+' % below both the MMSE posterior’s and the variance head’s. This is the opposite of Phase 2a, where the variance head won on CRPS. B, on its 50 cubes, has the lowest CRPS at 30 and 20 dB (ratio to L '+fx(b30.crps_vs_L.ratio_of_means,2)+' and '+fx(b20.crps_vs_L.ratio_of_means,2)+'). That comes from its accurate mean, not from its spread (section 3).';};
SL.e1B=function(h){const m=F0.B_e1_rmse;const mono=V.sanity&&V.sanity.monotone;
  h.innerHTML='<b>B is nearly flat with SNR.</b> Its mean-of-32 is '+E1S.map(function(x){return fx(m[String(x)],3);}).join(' / ')+' % at 60 / 50 / 40 / 30 / 20 dB. At the noise floor it is far above the MMSE (it fails the floor sanity check), and its error even rises by '+fx(m['30']-m['40'],2)+' from 40 to 30 dB, which fails the pre-registered monotonicity check'+(mono&&mono.B===false?'':' (verdict: passed)')+'. The prior, not the measurement noise, sets its error. That is why it wins only where the noise dominates everyone else’s error.';};
SL.t2text=function(h){const a=t2v('C',30),b=t2v('C',20),ba=t2v('B',30),bb=t2v('B',20);const band=V.thresholds.T2_COV90;
  h.innerHTML='L, V and C on the 200 gate cubes, B on its 50 (paired on the shared files). <b>C</b> '+(a.passed?'passes':'fails')+' T2 at 30 dB: CRPS − L = '+sg(a.crps_vs_L.mean,3)+' ± '+fx(a.crps_vs_L.se_cluster,3)+', − V = '+sg(a.crps_vs_V.mean,3)+' ± '+fx(a.crps_vs_V.se_cluster,3)+' (clustered), cov90 '+fx(a.cov90,3)+'. At 20 dB both differences are clear ('+sg(b.crps_vs_L.mean,3)+' ± '+fx(b.crps_vs_L.se_cluster,3)+' and '+sg(b.crps_vs_V.mean,3)+' ± '+fx(b.crps_vs_V.se_cluster,3)+'), but cov90 = '+fx(b.cov90,3)+(b.cov90_passed?' is inside the band':' lies '+fx(b.cov90-band[1],3)+' above the band')+', so it '+(b.passed?'passes':'fails on coverage only')+'. '+
   '<b>B</b> has the lowest CRPS at both SNRs (− V = '+sg(ba.crps_vs_V.mean,2)+' ± '+fx(ba.crps_vs_V.se_cluster,2)+' and '+sg(bb.crps_vs_V.mean,2)+' ± '+fx(bb.crps_vs_V.se_cluster,2)+'). It still fails T2 at both, on coverage (cov90 '+fx(ba.cov90,3)+' / '+fx(bb.cov90,3)+'). CRPS rewards an accurate mean more than it punishes a spread that is too narrow, so an over-confident sampler with a good mean can score the lowest CRPS.';};
SL.covtext=function(h){const c20=met(set('device1','sequential',20),'C','cov50'),c30=met(set('device1','sequential',30),'C','cov50');const B30=set('device1','sequential',30).B.means,B20=set('device1','sequential',20).B.means;
  h.textContent='C is slightly over-dispersed, and more so as the SNR falls (cov50 '+fx(c30,3)+' at 30 dB, '+fx(c20,3)+' at 20 dB, against 0.50 nominal). The variance head is the best calibrated of the four. B is badly over-confident: its 50 % interval covers '+fx(B30.cov50,2)+' / '+fx(B20.cov50,2)+' and its 90 % interval '+fx(B30.cov90,2)+' / '+fx(B20.cov90,2)+' at 30 / 20 dB.';};
SL.t3text=function(h){const c=V.T3.C,b=V.T3.B,th=V.thresholds;
  h.innerHTML='L, D and C on the 200 gate cubes, B on its 50. T3 at 30 dB: <b>B '+(b&&b.passed?'passes':'fails')+'</b>. Its RMSE ratio to the MMSE is '+fx(b.ratio_dev1,3)+' on Device 1 and '+fx(b.ratio_dev2,3)+' on Device 2, against the limit '+fx(th.T3_RATIO*b.ratio_dev1,3)+'. <b>C '+(c.passed?'passes':'fails')+'</b>: its ratio is '+fx(c.ratio_dev1,3)+' and '+fx(c.ratio_dev2,3)+', against the limit '+fx(th.T3_RATIO*c.ratio_dev1,3)+'. On PH5 (report-only) C’s ratio is '+fx(c.ph5,3)+'.';};
SL.ph5V=function(h){h.textContent=[40,30,20].map(function(x){const s=set('ph5','sequential',x);return fx(met(s,'V','crps'),2)+' vs '+fx(met(s,'L','crps'),2);}).join(', ')+' at 40 / 30 / 20 dB';};
SL.e2B=function(h){const xs=[40,30,20];
  h.innerHTML='<b>Device 2: B transfers zero-shot and '+(F0.B_best_dev2_all?'is the most accurate method at every SNR':'keeps most of its accuracy')+', but stays over-confident.</b> On its 50 cubes it scores '+xs.map(function(x){const b=set('device2','sequential',x).B;return fx(b.means.mean_rmse_pct,2)+' % against L '+fx(b.on_subset.L.mean_rmse_pct,2)+' and D '+fx(b.on_subset.D.rmse_pct,2)+' at '+x+' dB';}).join('; ')+'. Its cov90 is '+xs.map(function(x){return fx(set('device2','sequential',x).B.means.cov90,2);}).join(' / ')+'. B was never trained on any sensor, so transfer costs it nothing; its error is set by the prior.';};
SL.t4text=function(h){const c=V.T4.C,b=V.T4.B;
  h.textContent='L exact, L demosaic, D and C on the 200 gate cubes, B on its 50. The dashed segment at 30 dB is T4’s target. T4 fails for both lines. C: ratio '+fx(c.ratio,3)+' (2-SE bound '+fx(c.ratio_2se,3)+'), and its mean-of-32 ('+fx(F0.e3_C_30,2)+' %) is worse than demosaic-then-MMSE ('+fx(F0.Ldem_30,2)+' %). B: ratio '+fx(b.ratio,3)+' (2-SE bound '+fx(b.ratio_2se,3)+'). The regressor passes the same clause by a wide margin, at '+fx(F0.e3_D_30,2)+' % (ratio '+fx(c.D.ratio,3)+'), and it is below the baseline at every SNR.';};
SL.e3B=function(h){const xs=[60,40,30,20].filter(function(x){return bOk(set('device1','snapshot',x));});
  const r=xs.map(function(x){return set('device1','snapshot',x).B.paired.rmse_vs_L_demosaic.ratio;});
  h.innerHTML='<b>B cannot demosaic.</b> With the masked operator, zero-shot, it lands at the demosaic baseline: ratio '+fx(Math.min.apply(null,r),2)+'–'+fx(Math.max.apply(null,r),2)+' to demosaic-then-MMSE on its 50 cubes at '+xs.join(' / ')+' dB. Its spread is still too narrow (cov90 '+xs.map(function(x){return fx(set('device1','snapshot',x).B.means.cov90,2);}).join(' / ')+'). A spectral–spatial prior learned on full cubes does not supply the cross-pixel interpolation that the regressor learned from snapshot training cubes.';};
SL.e4=function(h){const s=set('device2','snapshot',30);if(!s||!s.methods.L){h.innerHTML='<b>E4 (Device 2, snapshot, 30 dB, report-only):</b> '+pend();return;}
  const bR=(bOk(s)&&s.B.paired&&s.B.paired.rmse_vs_L_demosaic)?s.B.paired.rmse_vs_L_demosaic.ratio:null;const bNear=bR!=null&&bR<1.15;
  const d1=set('device1','snapshot',30);const d1R=(d1&&bOk(d1)&&d1.B.paired&&d1.B.paired.rmse_vs_L_demosaic)?d1.B.paired.rmse_vs_L_demosaic.ratio:null;
  h.innerHTML='<b>E4, Device 2 in snapshot mode at 30 dB (report-only):</b> '+(F0.e4_D_over_dem>1&&F0.e4_C_over_dem>1?'none of the conditional networks trained on Device 1 transfers to the other sensor’s mosaic'+(bNear?'; the prior-guided B does, to the level of the classical pipeline':'')+'. ':'')+'Demosaic-then-MMSE gives '+fx(F0.e4_Ldem,2)+' % (exact per-pixel posterior '+fx(F0.e4_Lexact,2)+' %). D gives '+fx(F0.e4_D,2)+' % ('+fx(F0.e4_D_over_dem,2)+' × the baseline), C '+fx(F0.e4_C,2)+' % ('+fx(F0.e4_C_over_dem,2)+' ×) and C1 '+fx(F0.e4_C1,2)+' %. B: '+(bOk(s)?fx(s.B.means.mean_rmse_pct,2)+' % on its 50 cubes (ratio '+fx(bR,3)+' to the baseline on those cubes'+(d1R!=null?'; on Device 1 it was '+fx(d1R,3):'')+'), CRPS '+fx(s.B.means.crps,2)+', cov90 '+fx(s.B.means.cov90,2)+(bNear&&F0.e4_D_over_dem>1?' — its only sensor knowledge is the guidance operator, so it is the one learned method that transfers to the new mosaic, without beating the classical pipeline':''):pend())+'.'+(F0.e4_D_over_dem>1?' The regressor’s demosaicing is specific to the sensor it was trained on.':'')+'';};
SL.probetext=function(h){const P=D.b_prior_probe;const g=function(k){return P.series.filter(function(s){return s.key===k;})[0].err;};const i299=P.t.indexOf(299);const v=g('vpred'),bl=g('blind'),x0=g('x0'),g0=g('g0'),tt=g('tt');
  const gap=P.t.map(function(t,i){return Math.abs(v[i]-bl[i]);});const lo=P.t.filter(function(t){return t<=50;}).map(function(t){return gap[P.t.indexOf(t)];});const mid=P.t.filter(function(t){return t>=100&&t<=200;}).map(function(t){return gap[P.t.indexOf(t)];});
  const pg=Math.max.apply(null,P.t.filter(function(t){return t<=200;}).map(function(t){const i=P.t.indexOf(t);return Math.max(Math.abs(g0[i]-bl[i]),Math.abs(tt[i]-bl[i]));}));
  const worse=P.t.filter(function(t,i){return v[i]>P.mean_cube;});
  h.textContent='The v-prediction prior (epoch 100) follows the input-blind predictor E[v | t]: the two curves agree to within '+fx(Math.max.apply(null,lo),1)+' at t = 25 / 50 and within '+fx(Math.min.apply(null,mid),1)+'–'+fx(Math.max.apply(null,mid),1)+' points from t = 100 to 200 (the two pilots within '+fx(pg,1)+'). It is worse than predicting the mean cube ('+fx(P.mean_cube,2)+' %) at t = '+worse.join(', ')+'. Neither uniform loss weighting nor a time-token path moves the curve. With x₀-prediction the low-t target is close to the identity, which forces the network to read x_t. The retrained prior’s error is '+fx(x0[0],1)+' % at t = '+P.t[0]+' and '+fx(x0[i299],1)+' % at t = 299, where the v-prediction prior had '+fx(v[i299],1)+' % and the blind predictor '+fx(bl[i299],1)+' %; it grows monotonically and stays below the mean cube at every t. Unguided DDIM-50 samples saturated '+fx(P.saturation_pct.vpred,1)+' % of their elements with the v-prediction prior and '+fx(P.saturation_pct.x0,1)+' % with the x₀-prediction prior.';};
SL.x0rule=function(h){const P=D.b_prior_probe;const g=function(k){return P.series.filter(function(s){return s.key===k;})[0].err;};const v=g('vpred'),p=g('x0pilot');
  const idx=P.t.map(function(t,i){return i;}).filter(function(i){return P.t[i]<=300;});const drops=idx.map(function(i){return 1-p[i]/v[i];});const ok=idx.filter(function(i,j){return drops[j]>=0.3;}).map(function(i){return P.t[i];});const bad=idx.filter(function(i,j){return drops[j]<0.3;});
  h.innerHTML='<b>B’s prior retrained with x₀-prediction (an amendment, not pre-registered).</b> This amends spec §5.5’s “v-prediction” for the prior only (section 6). The decision rule was set after the first prior failed and before the pilot ran: the pilot’s curve must be monotone and at least 30 % below the failed prior’s at t ≤ 300. It met that at t = '+ok[0]+'–'+ok[ok.length-1]+(bad.length?'; at t = '+bad.map(function(i){return P.t[i]+' the drop is '+fx(100*(1-p[i]/v[i]),0)+' %';}).join(', '):'')+'. C stays v-prediction. The harness builds B’s sampler from the checkpoint’s own prediction type. B’s verdict therefore scores an amended line.';};
SL.bnow=function(h){const keys=ST.b_have;const under=keys.filter(function(k){return S[k].B.means.cov90<0.85;});
  const worse=keys.filter(function(k){const p=S[k].B.paired;const q=S[k].mode==='snapshot'?p.rmse_vs_L_demosaic:p.rmse_vs_D;return q&&q.diff-2*q.se_cluster>0;});
  const vk_=Object.keys(D.b_vpred).sort(function(a,b){return b-a;});
  h.innerHTML='<b>Where B stands.</b> With the x₀-prediction prior the samples are no longer degenerate. The single-sample RMSE is '+fx(Math.min.apply(null,keys.map(function(k){return S[k].B.means.single_rmse_pct;})),1)+'–'+fx(Math.max.apply(null,keys.map(function(k){return S[k].B.means.single_rmse_pct;})),1)+' % over the '+keys.length+' settings, against the v-prediction prior’s '+vk_.map(function(k){return fx(D.b_vpred[k].means.single_rmse_pct,1)+' % at '+k+' dB';}).join(' and ')+'. The mean-of-32 is worse than the reference method (D; in snapshot mode, demosaic-then-MMSE) by more than 2 clustered SE in '+worse.length+' of '+keys.length+' settings. The samples are under-dispersed (cov90 below 0.85) in '+under.length+' of '+keys.length+'.';};
SL.mainline=function(h){const cB=passes('B'),cC=passes('C'),tb=t2v('B',20),tc=t2v('C',20);
  const rule='The verdict counts T1 once (it passes only if all five SNRs pass), T2 once per SNR, then T3 and T4: at most 5 passes per line. ';
  if(!V.b_valid){h.innerHTML=rule+'C passes '+cC.length+' ('+(joinAnd(cC)||'none')+'). B: '+bInvalid()+' — '+bInvalidWhy()+', so B’s count and the main line wait for the regenerated verdict. Both lines are reported in full.';return;}
  const t=V.tie_break;let tie;
  if(t&&t.lines&&t.lines.B&&t.lines.C){const b=t.lines.B,c=t.lines.C,w=t.winner,l=w==='B'?'C':'B';
    tie='On the '+t.n_common+' cubes both lines share, '+w+'’s '+sg(t.lines[w].mean_common,3)+' ± '+fx(t.lines[w].se_cluster_common,3)+' is below '+l+'’s '+sg(t.lines[l].mean_common,3)+' ± '+fx(t.lines[l].se_cluster_common,3)+' (clustered SE; on their own cubes B '+sg(b.mean_own,3)+' on '+b.n_own+' and C '+sg(c.mean_own,3)+' on '+c.n_own+'), so the main line is <b>'+V.main_line+'</b>.';}
  else tie='B’s '+sg(tb.crps_vs_V.mean,3)+' (50 cubes) is below C’s '+sg(tc.crps_vs_V.mean,3)+' (200 cubes), so the main line is <b>'+V.main_line+'</b>.';
  h.innerHTML=rule+'B passes '+cB.length+' ('+(joinAnd(cB)||'none')+') and C passes '+cC.length+' ('+(joinAnd(cC)||'none')+'). A tie goes to the line with the lower paired CRPS − CRPS<sub>V</sub> at 20 dB, whether or not it passes T2 there. '+tie+' B wins on the accuracy of its mean; its coverage fails T2 at both SNRs. Both lines are reported in full.';};
SL.verdictnote=function(h){h.innerHTML='Verdict <code>'+ST.verdict_file+'</code>, created '+(V.created||'—')+'. B cells are on B’s 50 cubes, paired with the 200-cube references on the shared files (at least '+V.min_common+' common files required).';};
SL.budget=function(h){const b=D.budget,e=b.eval_hours;const ev=e.main+e.C+e.B+e.B_vpred+(e.B_stale||0);   /* B_stale: non-x0 B records still in eval/ (not counted in e.B) */const tot=b.train_hours+b.pilot_hours+ev;
  h.innerHTML='<b>Budget.</b> Training took '+fx(b.train_hours,1)+' GPU-h for the five runs (the failed v-prediction prior included), plus '+fx(b.pilot_hours,1)+' GPU-h for the three prior pilots. Evaluation took '+fx(e.main,2)+' GPU-h for the L / D / V records, '+fx(e.C,1)+' for C, '+fx(e.B_vpred,1)+' for the archived v-prediction B records (E1 60 / 50 dB and the mis-launched E4 cell) and '+fx(e.B,1)+' for the '+b.b_records+' x₀-prior B records'+(e.B_stale?' ('+fx(e.B_stale,1)+' more for stale non-x₀ B records not yet archived)':'')+' (E4’s L / D / V and C records are counted in those totals). A B record costs ≈ '+fx(b.b_hours_mean,2)+' GPU-h, against the spec’s 12 GPU-h cap for all of B. Total: ≈ '+fx(tot,0)+' GPU-h on RTX A4500s.';};
/* horizontal-scroll cue for charts and tables that overflow their card (phones) */
function hints(){document.querySelectorAll('.scroll,.tbl').forEach(function(h){const over=h.scrollWidth>h.clientWidth+4;let n=h.nextElementSibling;const has=n&&n.classList&&n.classList.contains('hint');
  if(over&&!has){const d=document.createElement('p');d.className='hint';d.textContent='scroll sideways for the full '+(h.classList.contains('tbl')?'table':'chart')+' →';h.parentNode.insertBefore(d,h.nextSibling);}else if(!over&&has)n.remove();});}

/* ---------- data-bound values ---------- */
function resolve(path){return path.split('.').reduce(function(o,k){return o==null?undefined:o[k];},D);}
document.querySelectorAll('[data-v]').forEach(function(h){const sp=h.dataset.v.split(':');const v=resolve(sp[0]);const f=sp[1];
  if(v==null){h.innerHTML=pend();return;}
  if(f==='e')h.textContent=sci(v);else if(f!=null&&typeof v==='number')h.textContent=Number(v).toFixed(+f);else h.textContent=String(v);});
document.querySelectorAll('[data-fig]').forEach(function(h){const f=F[h.dataset.fig];if(f)f(h);});
document.querySelectorAll('[data-chart]').forEach(function(h){const f=CH[h.dataset.chart];if(f)f(h);});
document.querySelectorAll('[data-table]').forEach(function(h){const f=T[h.dataset.table];if(f)f(h);});
document.querySelectorAll('[data-slot]').forEach(function(h){const f=SL[h.dataset.slot];if(f)f(h);});
hints();window.addEventListener('resize',hints);
document.querySelectorAll('details.fold').forEach(function(d){d.addEventListener('toggle',hints);});
