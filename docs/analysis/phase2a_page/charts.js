/* ---------- Phase 2a figures, charts and tables (helpers el, tx, box, arr, line, mk, rrect, cpath, hover, dataTable, clear come from the shared block) ---------- */
function fx(v,n){return v==null?'—':Number(v).toFixed(n);}
function sg(v,n){return v==null?'—':(v>0?'+':'')+Number(v).toFixed(n);}
function pill(ok,txt){return '<span class="pill '+(ok?'ok':'no')+'">'+(txt||(ok?'PASS':'FAIL'))+'</span>';}
function seg(attr,draw){const bs=document.querySelectorAll('[data-'+attr+']');bs.forEach(function(b){b.addEventListener('click',function(){bs.forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');draw(b.getAttribute('data-'+attr));});});}
function tbl(host,head,rows,hiRow){const w=document.createElement('table');w.className='t';let h='<thead><tr>'+head.map(function(x){return '<th>'+x+'</th>';}).join('')+'</tr></thead><tbody>';rows.forEach(function(r,i){h+='<tr'+(hiRow&&hiRow(i)?' class="hi"':'')+'>'+r.map(function(c){return typeof c==='object'&&c!==null?'<td class="'+(c.cls||'')+'">'+c.v+'</td>':'<td>'+c+'</td>';}).join('')+'</tr>';});w.innerHTML=h+'</tbody>';host.appendChild(w);}
function num(v){return {v:v,cls:'num'};}
const RUN={d8:D.runs.d8,k64:D.runs.d8k64,d4:D.runs.d4,ddpm:D.runs.d8ddpm};
const CC={P022:'--s2',P023:'--accent',P024:'--s3'};

const F={};
F.pipeline=function(h){
  const s=mk(h,960,246,760);
  tx(s,10,22,'ONE CENTRE','tag');
  box(s,10,34,210,66,['Low-res sensor image y_d','1 pixel per d × d block'],'box');
  arr(s,[[224,67],[262,67]]);
  box(s,266,34,236,66,['One-call regressor R','pre-registered epoch'],'box box-acc');
  arr(s,[[506,67],[544,67]],'ar-acc');
  box(s,548,34,190,66,['Centre x̂_R','RMSE 4.23 % at d = 8'],'box box-acc');
  tx(s,10,140,'TWO UNCERTAINTY MODELS AROUND IT','tag');
  box(s,10,152,350,72,['Residual diffusion','32 samples x̂_R + σ_R r_k (DDIM-10)'],'box box-s2');
  box(s,388,152,350,72,['Variance head','N(x̂_R, σ_R² σ²) with σ per pixel and band'],'box box-s3');
  arr(s,[[600,104],[185,148]],'ar-s2');arr(s,[[660,104],[563,148]]);
  tx(s,770,22,'THE TEST','tag');
  box(s,770,34,180,190,['200 gate cubes','CRPS · coverage','AUSE · texture','6 pre-registered','clauses at d = 8','d = 4 report only'],'box');
  arr(s,[[742,188],[766,188]]);
  tx(s,10,240,'Training centres are cross-fitted: each training cube is centred by the half-data regressor that never saw it.','sm');
};

const CH={};
CH.crps=function(h){
  const host=h;
  function rowsFor(k){const m=RUN[k].means;const g=D.constgauss[k];const best=Object.keys(g).filter(function(c){return !isNaN(+c);}).reduce(function(a,c){return g[c]<g[a]?c:a;});
    const rows=[['variance head','--accent',m.varhead.crps,'one Gaussian width per pixel and band'],['fixed-width Gaussian (c = '+best+')','--de',g[best],'N(x̂_R, ('+best+' σ_R)²), c chosen on the gate'],['residual diffusion','--s2',m.diffusion.crps,'32 DDIM-10 samples around x̂_R']];
    if(k==='d4'&&m.b)rows.push(['B, linear-centred diffusion','--s3',m.b.crps,'Phase 1a model around the linear estimate x̂₀']);
    rows.push(['regressor ensemble ('+(k==='d4'?'3':'2')+' members)','--ink-2',m.ensemble.crps,'Gaussian from the members’ mean and spread']);return rows;}
  function draw(k){clear(host);const rows=rowsFor(k);const W=720,rowH=40,m={l:240,r:80,t:14,b:34};const H=m.t+m.b+rowH*rows.length;const svg=mk(host,W,H,560);
    const max=Math.max.apply(null,rows.map(function(r){return r[2];}))*1.12;const pw=W-m.l-m.r;const X=function(v){return m.l+v/max*pw;};
    const step=max>1.5?0.5:0.2;for(let t=0;t<=max+1e-9;t+=step){line(svg,X(t),m.t-4,X(t),H-m.b,'grid');tx(svg,X(t),H-m.b+16,t.toFixed(1),'tick','middle');}
    tx(svg,m.l+pw,H-m.b+30,'mean CRPS, % of range (200 cubes)','axt','end');
    rows.forEach(function(r,i){const yc=m.t+rowH*i+rowH/2;tx(svg,m.l-12,yc+4,r[0],'bs','end');el('path',{d:rrect(X(0),yc-10,X(r[2])-X(0),20,4),style:'fill:var('+r[1]+')'},svg);tx(svg,X(r[2])+8,yc+4,r[2].toFixed(4),'vlbl','start');
      const hit=el('rect',{x:m.l,y:yc-rowH/2,width:pw+m.r,height:rowH,fill:'transparent',class:'barhit'},svg);hover(hit,function(){return '<div><b style="margin:0;padding:0">'+r[0]+'</b></div><div class="sub">CRPS '+r[2].toFixed(4)+' % · '+r[3]+'</div>';});});
    dataTable(host,['model','mean CRPS %','what it is'],rows.map(function(r){return [r[0],r[2].toFixed(4),r[3]];}));}
  draw('d8');seg('crps',draw);
};
CH.scatter=function(h){
  const host=h;
  function draw(k){clear(host);const sc=RUN[k].scatter;const W=720,H=420,m={l:58,r:24,t:24,b:44};const svg=mk(host,W,H,560);const pw=W-m.l-m.r,ph=H-m.t-m.b;
    const all=sc.diffusion.concat(sc.varhead);const lo=Math.floor(Math.min.apply(null,all)*4)/4,hi=Math.ceil(Math.max.apply(null,all)*4)/4;
    const X=function(v){return m.l+(v-lo)/(hi-lo)*pw;},Y=function(v){return m.t+ph-(v-lo)/(hi-lo)*ph;};
    const st=(hi-lo)>3?1:0.5;for(let t=Math.ceil(lo/st)*st;t<=hi+1e-9;t+=st){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,t.toFixed(1),'tick','end');line(svg,X(t),m.t,X(t),m.t+ph,'grid');tx(svg,X(t),m.t+ph+18,t.toFixed(1),'tick','middle');}
    line(svg,m.l,m.t+ph,m.l+pw,m.t+ph,'axis');line(svg,X(lo),Y(lo),X(hi),Y(hi),'diag');
    tx(svg,4,14,'diffusion CRPS %','axt');tx(svg,m.l+pw,m.t+ph+38,'variance head CRPS % (same cube)','axt','end');
    let worse=0;sc.diffusion.forEach(function(v,i){const u=sc.varhead[i];if(v>u)worse++;const p=el('circle',{cx:X(u),cy:Y(v),r:4,class:'pt',style:'fill:var('+CC[sc.classes[i]]+')'},svg);
      hover(p,function(){return '<div><b style="margin:0;padding:0">'+sc.classes[i]+' · cube '+(i+1)+'</b></div><div class="sub">variance head '+u.toFixed(3)+' · diffusion '+v.toFixed(3)+' ('+sg(v-u,3)+')</div>';});});
    tx(svg,m.l+10,m.t+14,'diffusion worse on '+worse+' of '+sc.diffusion.length+' cubes','sm');
    const lg=[['P022 (seen class)','--s2'],['P023','--accent'],['P024','--s3']];lg.forEach(function(q,i){el('circle',{cx:m.l+pw-150,cy:m.t+ph-58+i*18,r:4,class:'pt',style:'fill:var('+q[1]+')'},svg);tx(svg,m.l+pw-140,m.t+ph-54+i*18,q[0],'sm');});}
  draw('d8');seg('sc',draw);
};
function logY(lo,hi,top,ph){const a=Math.log10(lo),b=Math.log10(hi);return function(v){return top+ph-(Math.log10(Math.max(v,lo))-a)/(b-a)*ph;};}
CH.phase=function(h){
  const host=h;
  function draw(k){clear(host);const p=D.phases[k];const n=p.err.length;const idx=p.err.map(function(v,i){return i;}).sort(function(a,b){return p.err[a]-p.err[b];});
    const W=720,H=330,m={l:56,r:20,t:20,b:44};const svg=mk(host,W,H,560);const pw=W-m.l-m.r,ph=H-m.t-m.b;const lo=k==='d8'?0.2:0.1,hi=k==='d8'?10:5;const Y=logY(lo,hi,m.t,ph);const X=function(i){return m.l+i/(n-1)*pw;};
    (k==='d8'?[0.2,0.5,1,2,5,10]:[0.1,0.2,0.5,1,2,5]).forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,String(t),'tick','end');});
    line(svg,m.l,m.t+ph,m.l+pw,m.t+ph,'axis');tx(svg,4,12,'% of range (log scale)','axt');tx(svg,m.l+pw,m.t+ph+36,'sensor phase, sorted by error (first = measured pixel)','axt','end');
    [1,Math.round(n/4),Math.round(n/2),Math.round(3*n/4),n].forEach(function(t){tx(svg,X(t-1),m.t+ph+18,String(t),'tick','middle');});
    const ser=[['RMS error of x̂_R','--ink-2',p.err],['diffusion spread','--s2',p.sd],['variance head width','--accent',p.vh]];
    ser.forEach(function(s){const d=idx.map(function(j,i){return (i?'L':'M')+X(i).toFixed(1)+' '+Y(s[2][j]).toFixed(1);}).join(' ');el('path',{d:d,class:'ln',style:'stroke:var('+s[1]+')'},svg);
      idx.forEach(function(j,i){if(n<=16||i===0||i===n-1)el('circle',{cx:X(i),cy:Y(s[2][j]),r:3,class:'dot',style:'fill:var('+s[1]+')'},svg);});});
    const hit=el('rect',{x:m.l,y:m.t,width:pw,height:ph,class:'hit'},svg);
    hit.addEventListener('pointermove',function(e){const q=svgPt(svg,e);const i=Math.max(0,Math.min(n-1,Math.round((q.x-m.l)/pw*(n-1))));const j=idx[i];const r=Math.floor(j/(k==='d8'?8:4)),c=j%(k==='d8'?8:4);
      showTip(e,'<div><b style="margin:0;padding:0">phase ('+r+', '+c+')'+(j===0?' · measured pixel':'')+'</b></div>'+ser.map(function(s){return '<div class="tr"><i style="background:var('+s[1]+')"></i>'+s[0]+'<b>'+s[2][j].toFixed(3)+' %</b></div>';}).join(''));});
    hit.addEventListener('pointerleave',hideTip);
    dataTable(host,['phase (row, col)','error %','diffusion spread %','variance head %'],idx.map(function(j){const w=k==='d8'?8:4;return ['('+Math.floor(j/w)+', '+(j%w)+')',p.err[j].toFixed(3),p.sd[j].toFixed(3),p.vh[j].toFixed(3)];}));}
  draw('d8');seg('ph',draw);
};
CH.cov=function(h){
  const host=h;
  function draw(key){clear(host);const k=key.split(':')[0],lv=key.split(':')[1];const rows=D.deciles[k];const W=720,H=320,m={l:56,r:20,t:20,b:44};const svg=mk(host,W,H,560);const pw=W-m.l-m.r,ph=H-m.t-m.b;
    const y0=lv==='50'?0.2:0.6,y1=lv==='50'?1:1;const Y=function(v){return m.t+ph-(v-y0)/(y1-y0)*ph;};const X=function(i){return m.l+(i+0.5)/10*pw;};
    const band=lv==='50'?[0.45,0.55]:[0.85,0.95];el('rect',{x:m.l,y:Y(band[1]),width:pw,height:Y(band[0])-Y(band[1]),style:'fill:var(--good-bg)'},svg);
    line(svg,m.l,Y(+lv/100),m.l+pw,Y(+lv/100),'refacc');tx(svg,m.l+pw-4,Y(+lv/100)-6,'nominal '+(+lv/100).toFixed(2),'reflbl','end');
    for(let t=y0;t<=y1+1e-9;t+=0.1){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,t.toFixed(1),'tick','end');}
    line(svg,m.l,m.t+ph,m.l+pw,m.t+ph,'axis');for(let i=0;i<10;i++)tx(svg,X(i),m.t+ph+18,String(i+1),'tick','middle');
    tx(svg,4,12,'coverage of the central '+lv+' % interval','axt');tx(svg,m.l+pw,m.t+ph+36,'decile of the variance head’s predicted width (1 = calmest)','axt','end');
    const ser=[['residual diffusion','--s2','d'+lv],['variance head','--accent','v'+lv]];
    ser.forEach(function(s){const d=rows.map(function(r,i){return (i?'L':'M')+X(i).toFixed(1)+' '+Y(r[s[2]]).toFixed(1);}).join(' ');el('path',{d:d,class:'ln',style:'stroke:var('+s[1]+')'},svg);
      rows.forEach(function(r,i){const c=el('circle',{cx:X(i),cy:Y(r[s[2]]),r:4,class:'dot',style:'fill:var('+s[1]+')'},svg);hover(c,function(){return '<div><b style="margin:0;padding:0">'+s[0]+' · decile '+(i+1)+'</b></div><div class="sub">coverage '+r[s[2]].toFixed(3)+' · error RMS '+(100*r.er).toFixed(2)+' % · diffusion sd '+(100*r.sd).toFixed(2)+' % · variance head σ '+(100*r.sv).toFixed(2)+' %</div>';});});});
    dataTable(host,['decile','diffusion','variance head','error RMS %','diffusion sd %','variance head σ %'],rows.map(function(r,i){return [i+1,r['d'+lv].toFixed(3),r['v'+lv].toFixed(3),(100*r.er).toFixed(2),(100*r.sd).toFixed(2),(100*r.sv).toFixed(2)];}));}
  draw('d8:50');seg('cv',draw);
};
CH.psd=function(h){
  const P=D.psd_d8;const n=P.P_res_true.length;const W=520,H=300,m={l:52,r:28,t:20,b:44};const svg=mk(h,W,H,380);const pw=W-m.l-m.r,ph=H-m.t-m.b;const Y=logY(0.01,100,m.t,ph);const X=function(i){return m.l+i/(n-1)*pw;};
  [0.01,0.1,1,10,100].forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,String(t),'tick','end');});
  line(svg,m.l,m.t+ph,m.l+pw,m.t+ph,'axis');[0,8,16,24,32].forEach(function(t){tx(svg,X(t),m.t+ph+18,String(t),'tick','middle');});
  tx(svg,4,12,'power (log scale)','axt');tx(svg,m.l+pw,m.t+ph+36,'spatial frequency radius','axt','end');
  line(svg,X(12.5),m.t,X(12.5),m.t+ph,'ref');tx(svg,X(12.5)+6,m.t+12,'12.5','reflbl','start');
  [['true residual','--ink-2',P.P_res_true],['sample deviation','--s2',P.P_res_samp]].forEach(function(s){el('path',{d:s[2].map(function(v,i){return (i?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1);}).join(' '),class:'ln',style:'stroke:var('+s[1]+')'},svg);});
  dataTable(h,['radius','true residual','sample deviation','coherence'],P.P_res_true.map(function(v,i){return [i,v.toFixed(4),P.P_res_samp[i].toFixed(4),P.coh[i].toFixed(3)];}));
};
CH.probe=function(h){
  const R=D.probe_d4;lineChart(h,{w:520,h:300,ml:46,mr:34,minW:380,x:[0,1000],y:[0,1.2],xTicks:[0,200,400,600,800,1000],yTicks:[0,0.2,0.4,0.6,0.8,1,1.2],yFmt:function(v){return v.toFixed(1);},
    yTitle:'x₀ RMS error (r units)',xTitle:'timestep t',vrefs:[{x:399,label:'t = 399'}],tipX:function(v){return 't = '+v;},tipY:function(v){return v.toFixed(3);},
    series:[{name:'network',c:'--s2',data:R.map(function(r){return [r.t,r.net];}),marks:true},{name:'spectral filter',c:'--accent',data:R.map(function(r){return [r.t,r.spec];}),marks:true},{name:'no denoising',c:'--muted',dash:true,data:R.map(function(r){return [r.t,r.iid];})}],
    table:{head:['t','network','spectral filter','position-aware filter','no denoising','variance-head filter'],rows:R.map(function(r){return [r.t,r.net.toFixed(3),r.spec.toFixed(3),r.wpos.toFixed(3),r.iid.toFixed(3),r.vhw.toFixed(3)];})}});
};
function groupedBars(h,o){const W=720,H=o.h||320,m={l:58,r:16,t:22,b:56};const svg=mk(h,W,H,560);const pw=W-m.l-m.r,ph=H-m.t-m.b;const Y=function(v){return m.t+ph-(v-o.y[0])/(o.y[1]-o.y[0])*ph;};
  o.ticks.forEach(function(t){line(svg,m.l,Y(t),m.l+pw,Y(t),'grid');tx(svg,m.l-8,Y(t)+4,o.fmt(t),'tick','end');});line(svg,m.l,Y(o.base),m.l+pw,Y(o.base),'axis');tx(svg,4,12,o.yTitle,'axt');
  const band=pw/o.groups.length;const nb=o.series.length;const bw=Math.min(46,band/(nb+1.6));
  o.groups.forEach(function(g,gi){const cx=m.l+band*gi+band/2;tx(svg,cx,H-m.b+22,g.label,'bs','middle');
    o.series.forEach(function(s,si){const v=g.values[si];if(v==null)return;const x=cx+(si-(nb-1)/2)*(bw+5)-bw/2;const y=Y(Math.max(v,o.base)),y2=Y(Math.min(v,o.base));
      el('path',{d:v>=o.base?cpath(x,Y(o.base),bw,Y(o.base)-Y(v),4):rrect(x,Y(o.base),bw,Y(v)-Y(o.base),0),style:'fill:var('+s.c+')'},svg);
      tx(svg,x+bw/2,v>=o.base?y-6:y2+14,o.vfmt(v),'vlbl','middle');
      const hit=el('rect',{x:x,y:m.t,width:bw+5,height:ph,fill:'transparent',class:'barhit'},svg);hover(hit,function(){return '<div><b style="margin:0;padding:0">'+g.label+' · '+s.name+'</b></div><div class="sub">'+o.vfmt(v)+(s.note?' · '+s.note:'')+'</div>';});});});
  const lg=document.createElement('div');lg.className='legend';lg.innerHTML=o.series.map(function(s){return '<span><i class="sq" style="background:var('+s.c+')"></i>'+s.name+'</span>';}).join('');h.parentNode.insertBefore(lg,h);
  dataTable(h,['stride'].concat(o.series.map(function(s){return s.name;})),o.groups.map(function(g){return [g.label].concat(g.values.map(function(v){return v==null?'—':o.vfmt(v);}));}));}
CH.p2=function(h){
  const g=['d8','d4'].map(function(k){const m=RUN[k].means,P=D.P2[k==='d8'?'8':'4'];return {label:k==='d8'?'d = 8':'d = 4',values:[m.regressor.rmse_pct,m.baseline.rmse_pct,m.diffusion.single_rmse_pct,P.calibrated_single_rmse_sqrt2_reg]};});
  groupedBars(h,{y:[0,7],ticks:[0,1,2,3,4,5,6,7],base:0,fmt:function(v){return v.toFixed(0);},vfmt:function(v){return v.toFixed(2);},yTitle:'mean RMSE % (200 cubes)',groups:g,
    series:[{name:'regressor x̂_R',c:'--accent'},{name:'linear baseline',c:'--de'},{name:'one diffusion sample',c:'--s2',note:'observed'},{name:'calibrated sample (√2 × regressor)',c:'--warn',note:'what any calibrated sampler would give'}]});
};
CH.hf=function(h){
  const g=['d8','d4'].map(function(k){const m=RUN[k].means;return {label:k==='d8'?'d = 8':'d = 4',values:[m.baseline.hf,m.regressor.hf,m.diffusion.mean_hf,m.diffusion.single_hf]};});
  groupedBars(h,{h:330,y:[-3.6,3.2],ticks:[-3,-2,-1,0,1,2,3],base:0,fmt:function(v){return (v>0?'+':'')+v.toFixed(0);},vfmt:function(v){return sg(v,2);},yTitle:'log HF energy ratio (0 = truth)',groups:g,
    series:[{name:'linear baseline',c:'--de'},{name:'regressor x̂_R',c:'--accent'},{name:'mean of 32 samples',c:'--s3'},{name:'one diffusion sample',c:'--s2'}]});
};

/* ---------- tables ---------- */
const T={};
T.clauses=function(h){tbl(h,['clause','asks','rule (d = 8)','could a calibrated sampler pass?'],[
  ['<b>C1</b> calibration','the diffusion’s CRPS beats the variance head’s','mean(CRPS<sub>diff</sub> − CRPS<sub>vh</sub>) + 2 SE<sub>cl</sub> &lt; 0','<span class="pill ok">yes</span>'],
  ['<b>C2</b> calibration','the empirical intervals cover their nominal share','0.85 ≤ cov90 ≤ 0.95 and 0.45 ≤ cov50 ≤ 0.55','<span class="pill ok">yes</span> strict band'],
  ['<b>P1</b> perceptual','a single sample is less blurred than the regressor','mean(|hf<sub>single</sub>| − |hf<sub>reg</sub>|) + 2 SE<sub>cl</sub> &lt; 0','<span class="pill wn">passes by artefact</span>'],
  ['<b>P2</b> perceptual','a single sample still beats the linear baseline','mean(RMSE<sub>single</sub> − RMSE<sub>base</sub>) + 2 SE<sub>cl</sub> &lt; 0','<span class="pill no">no</span> √2 argument'],
  ['<b>S1</b> sanity','the mean of K samples stays on the regressor','|mean(RMSE<sub>meanK</sub> − RMSE<sub>reg</sub>)| ≤ 2 SE<sub>cl</sub>','<span class="pill no">no</span> needs K ≈ 256'],
  ['<b>S2</b> validity','swapping the condition makes the sample mean worse','mean(swap − true) − 2 SE &gt; 0 on 32 cubes','<span class="pill no">no</span> no power']]);};
T.runs=function(h){tbl(h,['stride','model','selected (rule)','monitoring value','GPU time'],[
  ['d = 8','half-1 regressor (4516 files)',num('epoch 15 of 40'),num('0.4857'),num('1.85 h')],
  ['d = 8','half-2 regressor (4484 files)',num('epoch 13 of 40'),num('0.4847'),num('1.83 h')],
  ['d = 8','centre cache (9000 + 1000 cubes)',num('σ_R 0.0893'),num('σ_full / σ_R 0.971'),num('0.1 h')],
  ['d = 8','residual diffusion',num('epoch 100 (fixed)'),num('0.1178 → 0.1030 (ep 5 → 100)'),num('8.97 h')],
  ['d = 8','variance head',num('epoch 10 of 40'),num('NLL −0.2391'),num('3.58 h')],
  ['d = 8','second-seed regressor',num('epoch 12 of 100'),num('0.4817'),num('9.05 h')],
  ['d = 4','half-1 regressor',num('epoch 12 of 40'),num('0.4310'),num('1.97 h')],
  ['d = 4','half-2 regressor',num('epoch 20 of 40'),num('0.4316'),num('1.91 h')],
  ['d = 4','centre cache',num('σ_R 0.0406'),num('σ_full / σ_R 1.004'),num('0.1 h')],
  ['d = 4','residual diffusion',num('epoch 100 (fixed)'),num('0.1225 → 0.1107 (ep 5 → 100)'),num('8.33 h')],
  ['d = 4','variance head',num('epoch 12 of 40'),num('NLL −0.2016'),num('3.99 h')]]);};
function clauseCell(c,k){if(!c)return num('—');if(k==='C2')return {v:pill(c.passed)+' <span class="num">cov90 '+fx(c.cov90,3)+' · cov50 '+fx(c.cov50,3)+'</span>',cls:''};const se=c.se_cluster!=null?c.se_cluster:c.se;return {v:pill(c.passed)+' <span class="num">'+sg(c.mean,4)+' ± '+fx(se,4)+'</span>',cls:''};}
T.verdicts=function(h){const ks=['C1','C2','P1','P2','S1','S2'];const cols=[['d8','d = 8 · K = 32 · the verdict'],['k64','d = 8 · K = 64 · fix round'],['d4','d = 4 · K = 32 · report only']];
  const rows=ks.map(function(k){return ['<b>'+k+'</b>'].concat(cols.map(function(c){return clauseCell(RUN[c[0]]&&RUN[c[0]].clauses[k],k);}));});
  ['calibration_pass','perceptual_pass','wins'].forEach(function(f){rows.push(['<b>'+f.replace('_pass','').replace('wins','diffusion wins')+'</b>'].concat(cols.map(function(c){return {v:pill(RUN[c[0]].clauses[f],RUN[c[0]].clauses[f]?'PASS':'FAIL'),cls:''};})));});
  tbl(h,['clause'].concat(cols.map(function(c){return c[1];})),rows,function(i){return i>=6;});
  const p=document.createElement('p');p.className='lead';p.style.fontSize='14px';p.textContent='Values are the clause statistic ± its cluster SE (S2: iid SE over the 32 y-swap cubes). The d = 4 column applies the same rules without a gate.';h.appendChild(p);};
T.fix=function(h){const a=RUN.d8.clauses,b=RUN.k64.clauses;const pred={C1:'≈ +0.17, unchanged',C2:'cov50 ≈ 0.67',P1:'unchanged, passes as artefact',P2:'≈ +1.17',S1:'≈ +0.023, still FAIL',S2:'still underpowered'};
  const held={C1:Math.abs(b.C1.mean-0.17)<0.02&&!b.C1.passed,C2:Math.abs(b.C2.cov50-0.67)<0.03&&!b.C2.passed,P1:b.P1.passed&&Math.abs(b.P1.mean-a.P1.mean)<1e-6,P2:Math.abs(b.P2.mean-1.17)<0.03&&!b.P2.passed,S1:Math.abs(b.S1.mean-0.023)<0.006&&!b.S1.passed,S2:!b.S2.passed};
  tbl(h,['clause','K = 32 (the verdict)','prediction for K = 64, written first','K = 64 observed','held?'],['C1','C2','P1','P2','S1','S2'].map(function(k){return ['<b>'+k+'</b>',clauseCell(a[k],k),pred[k],clauseCell(b[k],k),{v:held[k]?'<span class="pill ok">held</span>':'<span class="pill no">missed</span>',cls:''}];}));};
T.dose=function(h){const d8=D.dose.d8,d4=D.dose.d4,se=D.dose_se,pp=D.dose_paired;function pm2(v,s,n,u){return num(sg(v,n)+(s!=null?' ± '+fx(s,n):'')+(u||''));}
  tbl(h,['measure','d = 8','d = 4','d = 4 − d = 8 (paired)'],[
    ['CRPS gap, diffusion − variance head (% of range)',num(sg(d8.crps_gap_abs,3)+' ± '+fx(se.crps_gap_abs.d8,3)),num(sg(d4.crps_gap_abs,3)+' ± '+fx(se.crps_gap_abs.d4,3)),pm2(pp.crps_gap_abs_d4_minus_d8.mean,pp.crps_gap_abs_d4_minus_d8.se_cluster,3)],
    ['CRPS gap relative to the variance head (ratio of means)',num(fx(100*d8.crps_rel_ratio_of_means,1)+' % ± '+fx(100*se.crps_rel_ratio_of_means.d8,1)),num(fx(100*d4.crps_rel_ratio_of_means,1)+' % ± '+fx(100*se.crps_rel_ratio_of_means.d4,1)),num(sg(100*pp.crps_gap_rel_d4_minus_d8.mean,1)+' ± '+fx(100*pp.crps_gap_rel_d4_minus_d8.se_cluster_delta,1)+' pp')],
    ['CRPS gap relative, mean of per-cube ratios',num(fx(100*d8.crps_rel_mean_percube,1)+' % ± '+fx(100*se.crps_rel_mean_percube.d8,1)),num(fx(100*d4.crps_rel_mean_percube,1)+' % ± '+fx(100*se.crps_rel_mean_percube.d4,1)),num(sg(100*pp.crps_gap_rel_percube_d4_minus_d8.mean,1)+' ± '+fx(100*pp.crps_gap_rel_percube_d4_minus_d8.se_cluster,1)+' pp')],
    ['diffusion − best fixed-width Gaussian (CRPS)',num(sg(se.diff_minus_homo05.d8_mean,3)+' ± '+fx(se.diff_minus_homo05.d8,3)),num(sg(se.diff_minus_homo05.d4_mean,3)+' ± '+fx(se.diff_minus_homo05.d4,3)),num('—')],
    ['AUSE gap, diffusion − variance head',num(sg(d8.ause_gap_abs,3)+' ± '+fx(se.ause_gap_abs.d8,3)),num(sg(d4.ause_gap_abs,3)+' ± '+fx(se.ause_gap_abs.d4,3)),pm2(pp.ause_gap_abs_d4_minus_d8.mean,pp.ause_gap_abs_d4_minus_d8.se_cluster,3)],
    ['error-ranking skill, diffusion / variance head',num(fx(100*d8.ause_skill_d,0)+' % / '+fx(100*d8.ause_skill_v,0)+' %'),num(fx(100*d4.ause_skill_d,0)+' % / '+fx(100*d4.ause_skill_v,0)+' %'),num('—')],
    ['spread slope on error, diffusion / variance head',num(fx(D.slope['8'].diffusion,2)+' / '+fx(D.slope['8'].varhead,2)),num(fx(D.slope['4'].diffusion,2)+' / '+fx(D.slope['4'].varhead,2)),num('—')],
    ['cov50, diffusion / variance head',num(fx(d8.cov50_d,3)+' / '+fx(d8.cov50_v,3)),num(fx(d4.cov50_d,3)+' / '+fx(d4.cov50_v,3)),pm2(pp.cov50_diff_d4_minus_d8.mean,pp.cov50_diff_d4_minus_d8.se_cluster,3,' (diffusion)')],
    ['cov90, diffusion / variance head',num(fx(d8.cov90_d,3)+' / '+fx(d8.cov90_v,3)),num(fx(d4.cov90_d,3)+' / '+fx(d4.cov90_v,3)),pm2(pp.cov90_diff_d4_minus_d8.mean,pp.cov90_diff_d4_minus_d8.se_cluster,3,' (diffusion)')],
    ['single-sample log HF ratio',num(sg(d8.hf_single,2)),num(sg(d4.hf_single,2)),pm2(pp.hf_single_d4_minus_d8.mean,pp.hf_single_d4_minus_d8.se_cluster,3)],
    ['single-sample gradient-W1 penalty over the regressor',num(sg(100*d8.w1_pen_rel_ratio,0)+' % ± '+fx(100*se.w1_pen_rel_ratio.d8,0)),num(sg(100*d4.w1_pen_rel_ratio,0)+' % ± '+fx(100*se.w1_pen_rel_ratio.d4,0)),num(sg(100*pp.w1_gap_rel_d4_minus_d8.mean,1)+' ± '+fx(100*pp.w1_gap_rel_d4_minus_d8.se_cluster_delta,1)+' pp')],
    ['P1 statistic',num(sg(d8.p1,3)+' ± '+fx(se.p1.d8,3)),num(sg(d4.p1,3)+' ± '+fx(se.p1.d4,3)),pm2(pp.P1_abs_d4_minus_d8.mean,pp.P1_abs_d4_minus_d8.se_cluster,3)]]);};
T.compute=function(h){const rows=[['Stage 0 harness check (d = 4 assets)','0.55','22 Oct'],['d = 8 half regressors, 2 × 40 epochs','3.67','22 Oct'],['d = 8 residual diffusion, 100 epochs','8.97','22–23 Oct'],['d = 8 variance head, 40 epochs','3.58','22–23 Oct'],['d = 8 second-seed regressor, 100 epochs','9.05','23 Oct'],['centre caches, both strides','0.20','22–23 Oct'],['d = 8 verdict evaluation (K = 32)','1.53','23 Oct'],['d = 8 fix round, K = 64','2.48','23–24 Oct'],['d = 4 half regressors','3.88','23 Oct'],['d = 4 residual diffusion','8.33','23–24 Oct'],['d = 4 variance head','3.99','23 Oct'],['d = 4 contrast evaluation','2.09','24 Oct']];
  let tot=rows.reduce(function(a,r){return a+(+r[1]);},0);
  if(RUN.ddpm&&D.ddpm_hours){rows.push(['d = 8 DDPM-1000 reference row (K = 8)',fx(D.ddpm_hours,2),'24 Oct']);tot+=D.ddpm_hours;}
  tbl(h,['run','GPU-hours (RTX A4500)','when'],rows.map(function(r){return [r[0],num(r[1]),r[2]];}).concat([['<b>total</b>',num('<b>'+tot.toFixed(1)+'</b>'),'']]));
  const p=document.createElement('p');p.style.fontSize='14px';p.style.color='var(--muted)';p.textContent=(RUN.ddpm?'':'The total leaves out the DDPM-1000 reference row, still running. ')+'Verification ran on CPU: 6 agents for the d = 8 verdict and 2 for the d = 4 contrast, with a synthesis each.';h.appendChild(p);};

/* ---------- slots ---------- */
function verdictText(c){const L=[];const w=function(k,body){L.push(k+': '+(c[k].passed?'PASS':'FAIL')+'  '+body+'   ['+c[k].rule+']');};
  w('C1',sg(c.C1.mean,4)+' ± '+fx(c.C1.se,4)+' (cluster '+fx(c.C1.se_cluster,4)+')');w('C2','cov90 '+fx(c.C2.cov90,3)+' cov50 '+fx(c.C2.cov50,3));w('P1',sg(c.P1.mean,4)+' ± '+fx(c.P1.se,4)+' (cluster '+fx(c.P1.se_cluster,4)+')');
  w('P2',sg(c.P2.mean,4)+' ± '+fx(c.P2.se,4)+' (cluster '+fx(c.P2.se_cluster,4)+')');w('S1',sg(c.S1.mean,4)+' ± '+fx(c.S1.se,4)+' (cluster '+fx(c.S1.se_cluster,4)+')');w('S2',sg(c.S2.mean,4)+' ± '+fx(c.S2.se,4));
  L.push('calibration '+(c.calibration_pass?'PASS':'FAIL')+'; perceptual '+(c.perceptual_pass?'PASS':'FAIL')+'; diffusion wins (a): '+(c.wins?'True':'False'));return L.join('\n');}
const S={};
S.verdict=function(h){h.textContent=verdictText(RUN.d8.clauses);};
S.ddpm=function(h){const r=RUN.ddpm;if(!r){h.innerHTML='<b>DDPM-1000 reference row:</b> still running at the time of writing; it is reported without a gate because the verdict refuses $K&lt;19$.';return;}
  const m=r.means;h.innerHTML='<b>DDPM-1000 reference row</b> (report only, $K=8$, the 50-cube DDPM subset): diffusion CRPS '+fx(m.diffusion.crps,4)+' against the variance head’s '+fx(m.varhead.crps,4)+' on the same cubes; cov50 '+fx(m.diffusion.cov50,3)+', cov90 '+fx(m.diffusion.cov90,3)+', AUSE '+fx(m.diffusion.ause,3)+'; single-sample log HF ratio '+sg(m.diffusion.single_hf,2)+'.';};

document.querySelectorAll('[data-fig]').forEach(function(h){const f=F[h.dataset.fig];if(f)f(h);});
document.querySelectorAll('[data-chart]').forEach(function(h){const f=CH[h.dataset.chart];if(f)f(h);});
document.querySelectorAll('[data-table]').forEach(function(h){const f=T[h.dataset.table];if(f)f(h);});
document.querySelectorAll('[data-slot]').forEach(function(h){const f=S[h.dataset.slot];if(f)f(h);});
