"""
ALPHA vs BETA — the question XENO forced us to ask.
Every setup hovers near the base short-rate. So: do our setups actually
SELECT better moments, or is it just "short gold in a falling market"?

3 angles:
A) NAIVE DIRECTIONAL BENCHMARK
   Just short EVERY regime-eligible H1 bar (1:1 ATR). This is pure "beta".
   Any real edge must beat this per-trade EV.

B) MONTE-CARLO PERMUTATION TEST
   The flagship has n trades. Draw n RANDOM short entries from the SAME
   eligible pool (same regime/time conditions), 5000 times, build the EV
   distribution. Where does the flagship's actual EV fall?
   percentile >= 95% => real selection skill (alpha). Near 50% => beta.
   Do the same for the XENO short and for hour10.

C) CONDITIONAL BASE-RATE
   Within the eligible pool, what is the unconditional short win-rate?
   The setup's lift = setup_WR - pool_WR.
No look-ahead anywhere. SPREAD=0.3.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

def load(p):
    df=pd.read_csv(p); df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
    return df
def atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a
def swings_w(h,l,n,w):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl
def regime(c, slope_k):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if s>0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values; c=h1['c'].values
a=atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
ma50=pd.Series(c).rolling(50).mean().values
reg=regime(c,50)
hr=h1['Date'].dt.hour.values

def short_trade(i, horizon=24):
    """1:1 ATR short at next open. Return pnl (None if no resolution/invalid)."""
    if i+1>=n or np.isnan(a[i]) or a[i]<=0: return None
    entry=o[i+1]; rng=a[i]; tp=entry-rng; slv=entry+rng
    for k in range(i+1, min(n, i+1+horizon)):
        if h[k]>=slv: return -rng-SPREAD
        if l[k]<=tp:  return  rng-SPREAD
    return None

def evstat(idxs):
    pnls=[short_trade(i) for i in idxs]
    pnls=[p for p in pnls if p is not None]
    if not pnls: return None
    wr=100*np.mean([p>0 for p in pnls])
    return dict(wr=wr, ev=np.mean(pnls), n=len(pnls), tot=sum(pnls))

# eligible pool = bars where a short would be "allowed" by the regime filter
pool_sloping=[i for i in range(n) if reg[i] in ('UP','DOWN') and short_trade(i) is not None]
pool_down   =[i for i in range(n) if reg[i]=='DOWN'         and short_trade(i) is not None]

# ---- flagship signals (3bar sweep) ----
sh,sl=swings_w(h,l,n,1); shi=np.where(sh)[0]
sweep=[]
for i in range(2,n):
    ps=shi[shi<i-1]
    if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]]: sweep.append(i)
flagship=[i for i in sweep if reg[i] in ('UP','DOWN')]
flagship_prem=[i for i in flagship if not np.isnan(ma50[i]) and c[i]>ma50[i]]
hour10=[i for i in range(n) if hr[i]==10 and reg[i] in ('UP','DOWN')]

def perm_test(signal_idxs, pool, label, iters=5000):
    s=evstat(signal_idxs)
    if s is None or s['n']<10:
        print(f"{label}: n too few"); return
    # pool stats
    ps=evstat(pool)
    nsig=s['n']
    pool_valid=[i for i in pool]  # already validated
    rng=np.random.default_rng(42)
    dist=[]
    for _ in range(iters):
        samp=rng.choice(pool_valid, size=nsig, replace=False)
        pnls=[short_trade(i) for i in samp]
        pnls=[p for p in pnls if p is not None]
        dist.append(np.mean(pnls))
    dist=np.array(dist)
    pct=100*np.mean(dist < s['ev'])         # percentile of actual EV
    pval=np.mean(dist >= s['ev'])           # one-sided p: random >= actual
    print(f"\n{label}")
    print(f"   setup     : WR={s['wr']:.0f}% EV={s['ev']:+.2f} n={s['n']} tot={s['tot']:+.0f}")
    print(f"   pool(base): WR={ps['wr']:.0f}% EV={ps['ev']:+.2f} n={ps['n']}  (unconditional short in same regime)")
    print(f"   lift      : WR {s['wr']-ps['wr']:+.1f}pts  EV {s['ev']-ps['ev']:+.2f}")
    print(f"   random-EV dist: mean={dist.mean():+.2f} sd={dist.std():.2f} [5%={np.percentile(dist,5):+.2f}, 95%={np.percentile(dist,95):+.2f}]")
    print(f"   >>> flagship EV is at {pct:.1f}th percentile of random-same-pool   p(random>=actual)={pval:.3f}")
    verdict = "ALPHA (real selection skill)" if pct>=95 else ("weak edge" if pct>=80 else "BETA (indistinguishable from random short in regime)")
    print(f"   >>> VERDICT: {verdict}")

print("#"*92)
print("# ALPHA vs BETA — is it skill, or just shorting a falling market?")
print("#"*92)

print("\n--- NAIVE DIRECTIONAL BENCHMARK (short EVERY eligible bar) ---")
print(f"  SLOPING pool : {evstat(pool_sloping)}")
print(f"  DOWN pool    : {evstat(pool_down)}")
print("  ^ This is pure beta. ALL setups must beat the per-trade EV of this to matter.")

perm_test(flagship,      pool_sloping, "FLAGSHIP (sweep 3bar + RANGE-excl)  vs random-short-in-SLOPING")
perm_test(flagship_prem, pool_sloping, "FLAGSHIP + MA50上 (premium)         vs random-short-in-SLOPING")
perm_test(hour10,        pool_sloping, "hour10 short (RANGE-excl)           vs random-short-in-SLOPING")

# XENO on M15 -------------------------------------------------------------
print("\n" + "#"*92)
print("# Same permutation test for XENO short on M15")
print("#"*92)
m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv"); n15=len(m15)
o15=m15['o'].values;h15=m15['h'].values;l15=m15['l'].values;c15=m15['c'].values
a15=atr(h15,l15,c15)
ma200_15=pd.Series(c15).rolling(200).mean().values
reg15=regime(c15,200)
ts15=m15['Date'].values
# H1 trend (HH/HL) mapped
sh1,sl1=swings_w(h,l,n,1); shi1=np.where(sh1)[0]; sli1=np.where(sl1)[0]
h1_trend=np.array(['NA']*n,dtype=object)
for i in range(n):
    psh=shi1[shi1<i]; psl=sli1[sli1<i]
    if len(psh)<2 or len(psl)<2: continue
    hh=h[psh[-1]]>h[psh[-2]]; hl=l[psl[-1]]>l[psl[-2]]
    lh=h[psh[-1]]<h[psh[-2]]; ll=l[psl[-1]]<l[psl[-2]]
    if hh and hl: h1_trend[i]='UP'
    elif lh and ll: h1_trend[i]='DOWN'
    elif hh or hl: h1_trend[i]='WEAK_UP'
    elif lh or ll: h1_trend[i]='WEAK_DOWN'
h1_ts=h1['Date'].values
def h1_state(ts):
    idx=np.searchsorted(h1_ts, ts, side='right')-1
    if idx<0: return 'NA','NA'
    mb='UP' if (not np.isnan(ma200[idx]) and c[idx]>ma200[idx]) else 'DOWN'
    return h1_trend[idx], mb

def short_trade15(i, horizon=48):
    if i+1>=n15 or np.isnan(a15[i]) or a15[i]<=0: return None
    entry=o15[i+1]; rng=a15[i]; tp=entry-rng; slv=entry+rng
    for k in range(i+1, min(n15, i+1+horizon)):
        if h15[k]>=slv: return -rng-SPREAD
        if l15[k]<=tp:  return  rng-SPREAD
    return None

sh15,_=swings_w(h15,l15,n15,1); shi15=np.where(sh15)[0]
xeno=[]
for i in range(5,n15):
    if np.isnan(a15[i]) or a15[i]<=0: continue
    tr,mb=h1_state(ts15[i])
    if tr in ('DOWN','WEAK_DOWN') and mb=='DOWN':
        ps=shi15[shi15<i-1]
        if len(ps) and h15[i]>h15[ps[-1]] and c15[i]<h15[ps[-1]]: xeno.append(i)

# pool for XENO = M15 bars where H1 short-bias holds (regime context), short-valid
pool_xeno=[]
for i in range(5,n15):
    tr,mb=h1_state(ts15[i])
    if tr in ('DOWN','WEAK_DOWN') and mb=='DOWN' and short_trade15(i) is not None:
        pool_xeno.append(i)

def perm_test15(signal_idxs, pool, label, iters=5000):
    pnls_s=[short_trade15(i) for i in signal_idxs]; pnls_s=[p for p in pnls_s if p is not None]
    if len(pnls_s)<10: print(f"{label}: n few"); return
    ev=np.mean(pnls_s); wr=100*np.mean([p>0 for p in pnls_s])
    pnls_p=[short_trade15(i) for i in pool]; pnls_p=[p for p in pnls_p if p is not None]
    pev=np.mean(pnls_p); pwr=100*np.mean([p>0 for p in pnls_p])
    rng=np.random.default_rng(42); dist=[]
    for _ in range(iters):
        samp=rng.choice(pool, size=len(pnls_s), replace=False)
        pp=[short_trade15(i) for i in samp]; pp=[p for p in pp if p is not None]
        dist.append(np.mean(pp))
    dist=np.array(dist); pct=100*np.mean(dist<ev)
    print(f"\n{label}")
    print(f"   setup : WR={wr:.0f}% EV={ev:+.2f} n={len(pnls_s)}")
    print(f"   pool  : WR={pwr:.0f}% EV={pev:+.2f} n={len(pnls_p)}  (unconditional short when H1 short-bias)")
    print(f"   lift  : WR {wr-pwr:+.1f}pts  EV {ev-pev:+.2f}")
    print(f"   >>> at {pct:.1f}th percentile  verdict: {'ALPHA' if pct>=95 else ('weak' if pct>=80 else 'BETA')}")

perm_test15(xeno, pool_xeno, "XENO short (decline-origin) vs random-short-when-H1-short-bias")
