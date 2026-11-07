#!/usr/bin/env python3
"""Live web dashboard for the market-qx signal bot (multi-pair, multi-timeframe).

The user picks a pair and a timeframe, then presses Analyze. The bot computes
the indicator set on the stored candles for that exact (pair, timeframe) and
returns an explainable CALL / PUT / NEUTRAL verdict.

A single websocket connection streams every pair on every timeframe; closed
candles are persisted to SQLite (candle_store.py) so history accumulates across
restarts. Stdlib only — no Flask/etc.

Usage
-----
    .venv/bin/python dashboard.py --port 12000
    .venv/bin/python dashboard.py --assets EURUSD_otc,BTCUSD_otc --port 12000
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from analysis import analyze
from candle_store import CandleStore, MultiFeed, PERIODS, TIMEFRAMES, asset_catalog
from journal import Journal, settle_open, summary as journal_summary

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>market-qx signal bot</title>
<style>
  :root{--bg:#0b0f14;--panel:#131a23;--line:#1f2a37;--txt:#e6edf3;--mut:#8b98a5;
        --up:#26a69a;--down:#ef5350;--acc:#4c8dff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{display:flex;align-items:center;gap:14px;padding:12px 20px;
         border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
  .sym{font-size:18px;font-weight:600}
  .price{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--mut);display:inline-block}
  .dot.on{background:var(--up);box-shadow:0 0 8px var(--up)}
  .mut{color:var(--mut)}
  .wrap{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px;max-width:1500px;margin:0 auto}
  @media(max-width:960px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
  .card h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
  canvas{width:100%;height:360px;display:block}
  label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:10px 0 4px}
  select,button{width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--line);
        background:#0f151d;color:var(--txt);font-size:14px}
  button{background:var(--acc);border:none;font-weight:700;cursor:pointer;margin-top:14px}
  button:disabled{opacity:.5;cursor:default}
  .verdict{margin-top:14px;border-radius:10px;padding:14px;text-align:center;background:#0f151d;border:1px solid var(--line)}
  .verdict .big{font-size:30px;font-weight:800;letter-spacing:.02em}
  .verdict.call .big{color:var(--up)} .verdict.put .big{color:var(--down)}
  .verdict.neutral .big{color:var(--mut)}
  .reasons{margin:10px 0 0;padding:0;list-style:none;font-size:13px}
  .reasons li{padding:4px 0;border-bottom:1px solid var(--line)}
  .reasons li:last-child{border:none}
  .bull{color:var(--up)} .bear{color:var(--down)}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
  td{padding:5px 6px;border-bottom:1px solid var(--line)}
  td:first-child{color:var(--mut)} td:last-child{text-align:right;font-variant-numeric:tabular-nums}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat{background:#0f151d;border:1px solid var(--line);border-radius:8px;padding:8px}
  .stat b{display:block;font-size:17px}
  .bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:10px}
  .bar i{display:block;height:100%} .bar .b{background:var(--up)} .bar .s{background:var(--down)}
  .row{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center}
  .sbtn{background:#0f151d;border:1px solid var(--line);color:var(--txt);font-weight:600;
        cursor:pointer;padding:9px 10px;border-radius:8px;margin-top:0;width:auto}
  .sbtn:hover{border-color:var(--acc)}
  #scanList{margin-top:10px;max-height:420px;overflow:auto}
  .sig{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;
       padding:8px 6px;border-bottom:1px solid var(--line);font-size:13px}
  .sig:last-child{border:none}
  .sig .tag{font-weight:800;font-size:11px;padding:3px 7px;border-radius:5px;letter-spacing:.03em}
  .sig .tag.CALL{background:rgba(38,166,154,.16);color:var(--up)}
  .sig .tag.PUT{background:rgba(239,83,80,.16);color:var(--down)}
  .sig .tag.NEUTRAL{background:#0f151d;color:var(--mut)}
  .sig .m{font-variant-numeric:tabular-nums}
  .sig .why{color:var(--mut);font-size:11px;margin-top:2px}
  .sig .when{font-size:11px;margin-top:3px;font-variant-numeric:tabular-nums}
  .sig .when b{color:var(--acc)}
</style></head><body>
<header>
  <span class="dot" id="dot"></span>
  <span class="sym" id="sym">—</span>
  <span class="price" id="price">—</span>
  <span class="mut" id="meta"></span>
  <span class="mut" id="updated" style="margin-left:auto"></span>
</header>
<div class="wrap">
  <div>
    <div class="card" style="margin-bottom:16px">
      <h3>Select market</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div><label>Pair</label><select id="pair"></select></div>
        <div><label>Timeframe</label><select id="tf"></select></div>
      </div>
      <div style="margin-top:10px"><label>Engine</label>
        <select id="engine">
          <option value="rule">Rule (indicator votes)</option>
          <option value="smart">Smart (regime + volatility)</option>
        </select>
      </div>
      <button id="go">Analyze</button>
      <div id="hint" class="mut" style="margin-top:8px;font-size:12px"></div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <h3>Scan all pairs</h3>
      <div class="row">
        <div>
          <label>Timeframes</label>
          <select id="scanTf">
            <option value="5">5s (all pairs)</option>
            <option value="15">15s</option>
            <option value="60">1m</option>
            <option value="5,15,60">5s + 15s + 1m</option>
            <option value="all">All timeframes</option>
          </select>
        </div>
        <button class="sbtn" id="scanGo">Scan now</button>
      </div>
      <div id="scanHint" class="mut" style="margin-top:8px;font-size:12px">
        runs the current engine over every stored pair at once</div>
      <div id="scanList"></div>
    </div>
    <div class="card"><h3>Candles <span id="ctf" class="mut"></span></h3><canvas id="chart"></canvas></div>
  </div>
  <div class="card" style="margin-top:16px">
    <h3>Track record <span class="mut" style="font-size:12px">— every signal this bot has logged, settled on the real close</span></h3>
    <div id="jStats" class="grid" style="margin-bottom:10px"></div>
    <div id="jVerdict" class="mut" style="font-size:13px;margin-bottom:8px"></div>
    <div id="jPairs"></div>
  </div>
  <div>
    <div class="card" style="margin-bottom:16px">
      <h3>Signal</h3>
      <div class="verdict neutral" id="verdict">
        <div class="big" id="vbig">—</div>
        <div class="mut" id="vsub">pick a pair and press Analyze</div>
        <div id="vetas" style="margin-top:10px;font-size:13px;font-variant-numeric:tabular-nums"></div>
      </div>
      <div class="bar"><i class="b" id="barB" style="width:50%"></i><i class="s" id="barS" style="width:50%"></i></div>
      <ul class="reasons" id="reasons"></ul>
    </div>
    <div class="card" style="margin-bottom:16px"><h3>Indicators</h3>
      <table id="ind"></table></div>
    <div class="card"><h3>Feed</h3><div class="grid" id="stats"></div></div>
  </div>
</div>
<script>
let catalog=[], tfs=[], cur=null, lastAnalysis=null, autoN=0, autoScan=0, scanning=false;
function fmt(t){return new Date(t*1000).toLocaleTimeString();}
function clock(ts){return new Date(ts*1000).toLocaleTimeString();}
function cd(secs){
  if(secs==null) return '—';
  if(secs<=0) return 'now';
  if(secs<60) return 'in '+secs+'s';
  return 'in '+Math.floor(secs/60)+'m'+String(secs%60).padStart(2,'0')+'s';
}
async function boot(){
  const r=await fetch('/api/catalog'); const j=await r.json();
  if(j.loading){document.getElementById('hint').textContent='connecting to market feed…';setTimeout(boot,2000);return;}
  catalog=j.groups; tfs=j.timeframes;
  const pairSel=document.getElementById('pair');
  catalog.forEach(g=>{const og=document.createElement('optgroup');og.label=g.type;
    g.symbols.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;og.appendChild(o);});
    pairSel.appendChild(og);});
  pairSel.value=j.default_asset;
  const tfSel=document.getElementById('tf');
  tfs.forEach(t=>{const o=document.createElement('option');o.value=t.seconds;o.textContent=t.label;
    tfSel.appendChild(o);});
  tfSel.value=String(j.default_period);
  cur={asset:pairSel.value, period:tfSel.value};
  document.getElementById('go').onclick=analyzeNow;
  document.getElementById('scanGo').onclick=scanAll;
  tick();
  scanAll();
}
async function scanAll(){
  if(scanning) return;
  scanning=true;
  const tfs=document.getElementById('scanTf').value;
  const eng=document.getElementById('engine').value;
  const btn=document.getElementById('scanGo'); btn.disabled=true;
  document.getElementById('scanHint').textContent='scanning every pair…';
  const t0=performance.now();
  try{
    const r=await fetch('/api/scan?periods='+encodeURIComponent(tfs)+'&engine='+eng);
    const j=await r.json();
    const ms=Math.round(performance.now()-t0);
    const scope=tfs==='all'?'all timeframes':tfs.replace(/,/g,'s, ')+'s';
    document.getElementById('scanHint').textContent=
      j.scanned+' series · '+j.ready+' ready · '+j.signal_count+' signals · '+ms+' ms ('+scope+')';
    const el=document.getElementById('scanList');
    if(!j.signals.length){el.innerHTML='<div class="mut" style="padding:10px 0">no signal right now — most candles are noise</div>';return;}
    el.innerHTML=j.signals.map(s=>{
      const strength=(j.engine==='smart')
        ? 'conf '+s.confidence.toFixed(2)+(s.regime?(' · '+s.regime):'')
        : 'score '+s.score+' · bull '+s.bull+' / bear '+s.bear;
      const why=(s.reasons||[]).slice(0,2).join(' · ');
      const when=(s.entry_at&&s.expiry_at)
        ? 'entry <b>'+clock(s.entry_at)+'</b> <span data-in="'+s.entry_at+'">('+cd(s.entry_in)+')</span>'
          +' → expiry <b>'+clock(s.expiry_at)+'</b> <span data-in="'+s.expiry_at+'">('+cd(s.expires_in)+')</span>'
        : '';
      return '<div class="sig"><span class="tag '+s.action+'">'+s.action+'</span>'
        +'<span>'+s.asset+' <span class="mut">'+s.label+'</span>'
        +'<div class="why">'+why+'</div>'
        +'<div class="when">'+when+'</div></span>'
        +'<span class="m mut">'+strength+'</span></div>';
    }).join('');
  }catch(e){
    document.getElementById('scanHint').textContent='scan failed — retrying';
  }finally{
    btn.disabled=false; scanning=false;
  }
}
async function analyzeNow(){
  const a=document.getElementById('pair').value, p=document.getElementById('tf').value;
  const eng=document.getElementById('engine').value;
  cur={asset:a,period:p};
  document.getElementById('go').disabled=true;
  document.getElementById('hint').textContent='analyzing…';
  const r=await fetch('/api/analyze?asset='+encodeURIComponent(a)+'&period='+p+'&engine='+eng);
  lastAnalysis=await r.json(); renderAnalysis(lastAnalysis);
  document.getElementById('go').disabled=false;
}
function renderAnalysis(A){
  const v=document.getElementById('verdict');
  v.className='verdict '+(A.action==='CALL'?'call':A.action==='PUT'?'put':'neutral');
  document.getElementById('vbig').textContent=A.action==='NEUTRAL'?'NO TRADE':A.action;
  let sub;
  if(A.engine==='smart'){
    sub=A.action!=='NEUTRAL'
      ? (A.asset+' · '+A.period+'s · '+A.regime+' · confidence '+A.confidence.toFixed(2))
      : (A.enough? (A.asset+' · '+A.period+'s · '+A.note) : A.note);
  }else{
    sub=A.enough? (A.asset+' · '+A.period+'s · score '+A.score+' (bull '+A.bull+' / bear '+A.bear+')')
                : A.note;
  }
  document.getElementById('vsub').textContent=sub;
  const eta=document.getElementById('vetas');
  if(A.entry_at && A.expiry_at && A.action!=='NEUTRAL'){
    eta.innerHTML='<div>entry <b>'+clock(A.entry_at)+'</b> <span data-in="'+A.entry_at+'">('+cd(A.entry_in)+')</span>'
      +' &nbsp;→&nbsp; expiry <b>'+clock(A.expiry_at)+'</b> <span data-in="'+A.expiry_at+'">('+cd(A.expires_in)+')</span></div>'
      +'<div class="mut" style="font-size:11px;margin-top:4px">enter at the open of the next '
      +(A.expiry_candles||1)+'-candle window; expiry '+(A.expiry_candles||1)
      +' candle'+((A.expiry_candles||1)!==1?'s':'')+' later</div>';
  }else{
    eta.innerHTML='';
  }
  const tot=Math.max(1,(A.bull||0)+(A.bear||0));
  document.getElementById('barB').style.width=((A.bull||0)/tot*100)+'%';
  document.getElementById('barS').style.width=((A.bear||0)/tot*100)+'%';
  document.getElementById('reasons').innerHTML=(A.reasons||[]).map(x=>{
    const cls=/cross|uptrend|oversold|above/i.test(x)?'bull':'bear';
    return '<li class="'+cls+'">'+x+'</li>';}).join('')||'<li class="mut">no reasons</li>';
  const t=document.getElementById('ind'); const V=A.values||{};
  t.innerHTML=Object.keys(V).filter(k=>k!=='last_candle_time').map(k=>
    '<tr><td>'+k.replace(/_/g,' ')+'</td><td>'+(V[k]===null?'—':V[k])+'</td></tr>').join('');
  document.getElementById('hint').textContent=
    A.enough? 'candles used: '+A.candles : (A.note||'');
}
async function tick(){
  try{
    const a=cur.asset,p=parseInt(cur.period);
    const r=await fetch('/api/state?asset='+encodeURIComponent(a)+'&period='+p);
    const s=await r.json();
    document.getElementById('dot').className='dot'+(s.connected?' on':'');
    document.getElementById('updated').textContent='updated '+fmt(s.now);
    document.getElementById('stats').innerHTML=
      st('assets',s.assets)+st('timeframes',s.timeframes)
      +st('candles stored',s.stored.toLocaleString())+st('ticks/s',s.tps);
    if(cur){
      const a=cur.asset,p=parseInt(cur.period);
      document.getElementById('sym').textContent=a+' · '+p+'s';
      const pr=s.prices[a];
      document.getElementById('price').textContent=pr!=null?pr:'—';
      const cc=(s.candle_counts||{})[a+'|'+p];
      document.getElementById('meta').textContent=cc?('candles '+cc):'collecting…';
      document.getElementById('ctf').textContent=a+' '+p+'s';
      draw((s.series||{})[a+'|'+p], (s.forming||{})[a+'|'+p]);
      if(lastAnalysis && !lastAnalysis.enough){
        const need=52, have=cc||0, pct=Math.min(100,have/need*100);
        const eta=Math.max(0,(need-have)*p);
        document.getElementById('hint').innerHTML=
          'collecting candles: <b>'+have+' / '+need+'</b>'
          +'<div style="height:6px;background:#0f151d;border-radius:3px;margin-top:6px;overflow:hidden">'
          +'<div style="height:100%;width:'+pct+'%;background:#4c8dff"></div></div>'
          +'<span class="mut">signal ready in ~'+Math.ceil(eta/60)+' min (no public history; builds live)</span>';
      }
      // auto-refresh: re-analyze once enough candles exist, then keep it fresh
      if(lastAnalysis && !lastAnalysis.enough && (cc||0)>=52){ analyzeNow(); }
      else if(lastAnalysis && lastAnalysis.enough){
        autoN++; if(autoN>=7){autoN=0; analyzeNow();}
      }
      // keep the all-pairs scan fresh too (~every 20s)
      autoScan++; if(autoScan>=13){autoScan=0; scanAll();}
    }
  }catch(e){document.getElementById('dot').className='dot';}
  setTimeout(tick,1500);
}
function st(k,v){return '<div class="stat"><span class="mut">'+k+'</span><b>'+v+'</b></div>';}
async function loadJournal(){
  try{
    const r=await fetch('/api/journal'); const j=await r.json();
    const o=j.overall||{n:0,hit:0,lo:0,hi:0}, d=j.decisive||{n:0,hit:0,lo:0,hi:0};
    const pc=x=>(x*100).toFixed(1)+'%';
    document.getElementById('jStats').innerHTML=
      st('settled',j.settled||0)+
      st('win / loss',(j.wins||0)+' / '+(j.losses||0))+
      st('flat ties',j.flats||0)+
      st('open',j.open||0)+
      st('hit-rate',pc(o.hit||0))+
      st('decisive',pc(d.hit||0))+
      st('breakeven',pc(j.breakeven||0.5405));
    let v='';
    if(!j.settled){v='No signals settled yet — the bot needs time to log and settle them. Check back later.';}
    else if(j.verdict==='above'){v='🟢 Hit-rate is statistically ABOVE the payout breakeven.';}
    else if(j.verdict==='below'){v='🔴 Hit-rate is statistically BELOW the payout breakeven — trading this loses money.';}
    else {v='⚪ Not yet distinguishable from breakeven (needs many more settled signals).';}
    if(j.settled && o.n) v+='  95% CI ['+pc(o.lo)+', '+pc(o.hi)+'] on n='+o.n+'.';
    document.getElementById('jVerdict').textContent=v;
    const pairs=(j.pairs||[]).filter(p=>p.n>=10).slice(0,12);
    document.getElementById('jPairs').innerHTML = pairs.length ? pairs.map(p=>
      '<div class="mut" style="font-size:12px;font-variant-numeric:tabular-nums">'+
      p.asset+' '+(p.period>=60?(p.period/60)+'m':p.period+'s')+
      ' — '+p.wins+'/'+p.n+' ('+pc(p.hit)+')</div>').join('')
      : '<div class="mut" style="font-size:12px">no pair has 10+ settled signals yet</div>';
  }catch(e){}
}
setInterval(loadJournal,10000);
function tickCountdowns(){
  const now=Math.floor(Date.now()/1000);
  document.querySelectorAll('[data-in]').forEach(el=>{
    const t=parseInt(el.getAttribute('data-in'));
    el.textContent='('+cd(t-now)+')';
  });
}
setInterval(tickCountdowns,1000);
function draw(cs,form){
  const cv=document.getElementById('chart');
  const dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight;
  cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  if(!cs||cs.length<2){g.fillStyle='#8b98a5';g.font='14px sans-serif';
    g.fillText('collecting candles…',16,H/2);return;}
  cs=cs.slice(-70); const all=form?cs.concat([form]):cs;
  const pad=8, plotW=W-pad*2, plotH=H-pad*2;
  let hi=Math.max(...all.map(c=>c[3])), lo=Math.min(...all.map(c=>c[4]));
  const rng=(hi-lo)||1; hi+=rng*0.05; lo-=rng*0.05;
  const y=v=>pad+plotH*(1-(v-lo)/(hi-lo)), cw=plotW/cs.length;
  g.strokeStyle='#1f2a37'; g.lineWidth=1;
  for(let i=0;i<=4;i++){const yy=pad+plotH*i/4;g.beginPath();g.moveTo(pad,yy);g.lineTo(W-pad,yy);g.stroke();}
  g.fillStyle='#8b98a5'; g.font='11px monospace';
  g.fillText(hi.toFixed(5),pad+2,pad+10); g.fillText(lo.toFixed(5),pad+2,H-pad-2);
  all.forEach((c,i)=>{
    const up=c[2]>=c[1], col=up?'#26a69a':'#ef5350';
    const x=pad+cw*i+cw/2, bw=Math.max(1,cw*0.62);
    g.strokeStyle=col; g.fillStyle=col; g.lineWidth=1;
    g.beginPath(); g.moveTo(x,y(c[3])); g.lineTo(x,y(c[4])); g.stroke();
    const yo=y(c[1]), yc=y(c[2]);
    g.fillRect(x-bw/2, Math.min(yo,yc), bw, Math.max(1,Math.abs(yc-yo)));
  });
}
boot(); window.addEventListener('resize',()=>{});
</script></body></html>"""


class Hub:
    """Shared live state behind the HTTP handlers."""

    def __init__(self, assets, periods, store, wait_instruments=75.0, journal=None):
        self.store = store
        self.periods = list(periods)
        self.assets = list(assets)
        self.lock = threading.Lock()
        self._tick_count = 0
        self._last_n = 0
        self._tps = 0.0
        self.feed: Optional[MultiFeed] = None
        self.instruments: dict[str, dict] = {}
        self.ready = threading.Event()
        self.error = ""
        self._wait = wait_instruments
        self.journal = journal
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        try:
            self.feed = MultiFeed(self.assets, self.periods, self.store,
                                  on_closed=self._on_closed, on_tick=self._on_tick,
                                  wait_instruments=self._wait)
            self.assets = self.feed.assets
            self.instruments = self.feed.instruments
            self.ready.set()
            threading.Thread(target=self._tps_loop, daemon=True).start()
            if self.journal is not None:
                threading.Thread(target=self._journal_loop, daemon=True).start()
        except Exception as exc:  # surface, don't hang the server
            self.error = f"{type(exc).__name__}: {exc}"
            print("feed boot failed:", self.error, flush=True)

    def _journal_loop(self):
        """Record live signals and settle expired ones, in the background."""
        from scan import scan_all
        while True:
            time.sleep(30)
            try:
                now = int(time.time())
                settle_open(self.journal, self.store, now)
                rep = scan_all(self.store, self.periods, "rule", top=0)
                for row in rep["signals"]:
                    cs = self.store.load(row["asset"], row["period"], limit=2)
                    row["price"] = cs[-1][2] if cs else 0.0
                    self.journal.add(row, "rule", now)
            except Exception as exc:  # never let the journal kill the feed
                print("journal loop error:", exc, flush=True)

    def _on_closed(self, asset, period, candle):
        pass  # hook for future alerts

    def _on_tick(self, asset, ts, price):
        with self.lock:
            self._tick_count += 1

    def _tps_loop(self):
        while True:
            time.sleep(5)
            with self.lock:
                n = self._tick_count
            self._tps = (n - self._last_n) / 5.0
            self._last_n = n

    def catalog(self):
        if not self.ready.is_set():
            return {"groups": [], "timeframes": [{"seconds": s, "label": l} for s, l in TIMEFRAMES],
                    "default_asset": "", "default_period": self.periods[0], "loading": True}
        return {"groups": asset_catalog(self.instruments, self.assets),
                "timeframes": [{"seconds": s, "label": l} for s, l in TIMEFRAMES],
                "default_asset": self._default_asset(), "default_period": self.periods[0],
                "loading": False}

    def _default_asset(self):
        for want in ("EURUSD_otc", "EURUSD"):
            if want in self.assets:
                return want
        return self.assets[0] if self.assets else ""

    def state(self, series_asset=None, series_period=None):
        with self.lock:
            tps = round(self._tps, 1)
        if not self.ready.is_set():
            return {"now": time.time(), "connected": False, "loading": True,
                    "error": self.error, "assets": len(self.assets),
                    "timeframes": len(self.periods), "stored": self.store.total,
                    "tps": 0.0, "prices": {}, "candle_counts": {},
                    "series": {}, "forming": {}}
        prices = dict(self.feed.last_price)
        counts, series, forming = {}, {}, {}
        if series_asset is not None:
            key = f"{series_asset}|{series_period}"
            cs = self.feed.candles(series_asset, series_period)
            counts[key] = len(cs)
            series[key] = cs[-70:]
            forming[key] = self.feed.forming(series_asset, series_period)
        return {"now": time.time(), "connected": self.feed.connected, "loading": False,
                "assets": len(self.assets), "timeframes": len(self.periods),
                "stored": self.store.total, "tps": tps, "prices": prices,
                "candle_counts": counts, "series": series, "forming": forming}


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/scan":
            q = parse_qs(u.query)
            spec = (q.get("periods") or ["5"])[0]
            engine = (q.get("engine") or ["rule"])[0]
            try:
                top = int((q.get("top") or ["60"])[0])
            except ValueError:
                top = 60
            from scan import parse_periods, scan_all
            rep = scan_all(self.hub.store, parse_periods(spec), engine, top=top)
            return self._json(rep)
        if u.path == "/api/journal":
            q = parse_qs(u.query)
            asset = (q.get("asset") or [""])[0]
            engine = (q.get("engine") or ["rule"])[0]
            try:
                payout = float((q.get("payout") or ["0.85"])[0])
            except ValueError:
                payout = 0.85
            if self.hub.journal is None:
                return self._json({"logged": 0, "settled": 0, "open": 0,
                                   "verdict": "disabled", "pairs": []})
            return self._json(journal_summary(self.hub.journal, asset, engine, payout))
        if u.path == "/api/catalog":
            return self._json(self.hub.catalog())
        if u.path == "/api/analyze":
            q = parse_qs(u.query)
            default = self.hub.assets[0] if self.hub.assets else ""
            asset = (q.get("asset") or [default])[0]
            try:
                period = int((q.get("period") or [self.hub.periods[0]])[0])
            except ValueError:
                period = self.hub.periods[0]
            candles = self.hub.feed.candles(asset, period) if self.hub.feed else []
            engine = (q.get("engine") or ["rule"])[0]
            from scan import entry_plan
            try:
                expiry = int((q.get("expiry") or ["1"])[0])
            except ValueError:
                expiry = 1
            res = (analyze_smart(asset, period, candles) if engine == "smart"
                   else analyze(asset, period, candles))
            out = res.__dict__
            if res.enough and candles:
                out.update(entry_plan(period, candles[-1][0], time.time(), expiry))
            return self._json(out)
        if u.path == "/api/state":
            q = parse_qs(u.query)
            sa = (q.get("asset") or [None])[0]
            try:
                sp = int((q.get("period") or [0])[0]) or None
            except ValueError:
                sp = None
            return self._json(self.hub.state(sa, sp))
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Multi-pair signal dashboard")
    ap.add_argument("--assets", default="", help="comma list; default = all instruments")
    ap.add_argument("--periods", default="", help="comma list of seconds; default = all")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 12000)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--db", default=os.environ.get("CANDLE_DB", "data/candles.db"))
    ap.add_argument("--journal", default=os.environ.get("JOURNAL_DB", "data/journal.db"),
                    help="signal journal database ('' to disable)")
    args = ap.parse_args()

    store = CandleStore(args.db)
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    periods = [int(p) for p in args.periods.split(",") if p.strip()] or PERIODS

    journal = Journal(args.journal) if args.journal else None
    hub = Hub(assets, periods, store, journal=journal)
    Handler.hub = hub
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  "
          f"({'all' if not assets else len(assets)} pairs x {len(periods)} timeframes)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
