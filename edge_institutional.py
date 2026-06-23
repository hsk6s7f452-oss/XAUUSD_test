"""
INSTITUTIONAL / ALGO EDGE TEST
Concepts: VWAP deviation, PDH/PDL sweep, Asia range fake-break,
          round numbers (xx00/xx50), volume climax/exhaustion.
All vs beta benchmark EV+1.25 (short every sloping H1 bar).
1:1 ATR, SPREAD=0.3, no look-ahead, half-split stability.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3
BETA_EV=1.25   # benchmark: naive directional short in sloping regime

def load(p):
    df=pd.read_csv(p)
    df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
    return df

def atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a

def regime(c, slope_k=50):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

# ── load H1 ──
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values
c=h1['c'].values; v=h1['v'].values
at=atr(h,l,c); h1['atr']=at
reg=regime(c)
hr=h1['Date'].dt.hour.values
dy=h1['Date'].dt.date.values
mon=h1['Date'].dt.strftime('%Y-%m').values

def trade(i, side=-1, horizon=24):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]
    tp=entry+rng*side; slv=entry-rng*side
    for k in range(i+1, min(n,i+1+horizon)):
        if side<0:
            if h[k]>=slv: return -rng-SPREAD
            if l[k]<=tp:  return  rng-SPREAD
        else:
            if l[k]<=slv: return -rng-SPREAD
            if h[k]>=tp:  return  rng-SPREAD
    return None

def report(label, idxs, side=-1, min_n=10):
    pnls=[trade(i,side) for i in idxs]; pnls=[p for p in pnls if p is not None]
    if len(pnls)<min_n:
        print(f"  {label:<54} n={len(pnls)} (too few)"); return
    wr=100*np.mean([p>0 for p in pnls]); ev=np.mean(pnls); tot=sum(pnls)
    mcl=cur=0
    for p in pnls:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    months=len(set(mon[i] for i in idxs)); tpm=len(pnls)/max(months,1)
    mid=n//2
    h1p=[p for p,i in zip(pnls,idxs) if i<mid]
    h2p=[p for p,i in zip(pnls,idxs) if i>=mid]
    def hw(x): return f"{100*np.mean([p>0 for p in x]):.0f}%/{len(x)}" if len(x)>=8 else "-"
    stable=""
    if len(h1p)>=8 and len(h2p)>=8:
        stable="✅STABLE" if np.mean(h1p)>0 and np.mean(h2p)>0 else "❌unstable"
    beat="🔥BEATS_BETA" if ev>BETA_EV else ""
    print(f"  {label:<54} WR={wr:4.0f}% EV={ev:+5.2f} n={len(pnls):4d} MCL={mcl} ~{tpm:.0f}/mo  half={hw(h1p)},{hw(h2p)}  {stable} {beat}")

print("="*100)
print("# INSTITUTIONAL / ALGO EDGES  (beta benchmark: EV+1.25)")
print("="*100)

# ══════════════════════════════════════════
# 1. DAILY VWAP deviation
# ══════════════════════════════════════════
print("\n### 1. VWAP — daily anchor, deviation fade short/long ###")

# daily VWAP (cum vol-price / cum vol, reset each calendar day)
vwap=np.full(n,np.nan)
days=sorted(set(dy))
for d in days:
    mask=np.where(dy==d)[0]
    tp_=(h[mask]+l[mask]+c[mask])/3
    cv=np.cumsum(v[mask]*tp_); cvol=np.cumsum(v[mask])
    vwap[mask]=cv/cvol

for dev_thr, label in [(1.0,'≥1ATR above VWAP short'), (1.5,'≥1.5ATR above VWAP short'),
                        (2.0,'≥2ATR above VWAP short'), (0.5,'≤0.5ATR above VWAP short')]:
    sigs=[i for i in range(1,n)
          if not np.isnan(vwap[i]) and not np.isnan(at[i]) and at[i]>0
          and (c[i]-vwap[i])/at[i]>=dev_thr and reg[i] in ('UP','DOWN')]
    report(label, sigs, -1)

# VWAP support bounce long
for dev_thr, label in [(1.0,'≥1ATR below VWAP long'), (1.5,'≥1.5ATR below VWAP long')]:
    sigs=[i for i in range(1,n)
          if not np.isnan(vwap[i]) and not np.isnan(at[i]) and at[i]>0
          and (vwap[i]-c[i])/at[i]>=dev_thr and reg[i] in ('UP','DOWN')]
    report(label, sigs, +1)

# Price crosses above VWAP (momentum long) in UP regime
cross_up=[i for i in range(1,n)
          if not np.isnan(vwap[i]) and not np.isnan(vwap[i-1])
          and c[i-1]<vwap[i-1] and c[i]>vwap[i] and reg[i]=='UP']
report("Cross above VWAP long (UP regime)", cross_up, +1)
cross_dn=[i for i in range(1,n)
          if not np.isnan(vwap[i]) and not np.isnan(vwap[i-1])
          and c[i-1]>vwap[i-1] and c[i]<vwap[i] and reg[i]=='DOWN']
report("Cross below VWAP short (DOWN regime)", cross_dn, -1)

# ══════════════════════════════════════════
# 2. PDH / PDL  (Previous Day High/Low)
# ══════════════════════════════════════════
print("\n### 2. PDH/PDL — previous day high/low sweep & reject ###")

# precompute daily high/low
day_hl={}
for d in days:
    mask=np.where(dy==d)[0]
    day_hl[d]=(h[mask].max(), l[mask].min())

pdh=np.full(n,np.nan); pdl=np.full(n,np.nan)
for i in range(n):
    d=dy[i]
    idx=days.index(d)
    if idx>0:
        prev=days[idx-1]
        if prev in day_hl:
            pdh[i]=day_hl[prev][0]
            pdl[i]=day_hl[prev][1]

# PDH sweep short: bar wicks above PDH and closes below it
pdh_sweep=[i for i in range(1,n)
           if not np.isnan(pdh[i]) and not np.isnan(at[i]) and at[i]>0
           and h[i]>pdh[i] and c[i]<pdh[i] and reg[i] in ('UP','DOWN')]
report("PDH sweep → short (close back below)", pdh_sweep, -1)

# PDH sweep short, DOWN only
pdh_sweep_dn=[i for i in pdh_sweep if reg[i]=='DOWN']
report("PDH sweep → short (DOWN regime)", pdh_sweep_dn, -1)

# PDL sweep long
pdl_sweep=[i for i in range(1,n)
           if not np.isnan(pdl[i]) and not np.isnan(at[i]) and at[i]>0
           and l[i]<pdl[i] and c[i]>pdl[i] and reg[i] in ('UP','DOWN')]
report("PDL sweep → long (close back above)", pdl_sweep, +1)

# PDH rejection (price approaches within 0.3ATR but doesn't break)
pdh_reject=[i for i in range(1,n)
            if not np.isnan(pdh[i]) and not np.isnan(at[i]) and at[i]>0
            and abs(h[i]-pdh[i])<0.3*at[i] and c[i]<pdh[i] and reg[i]=='DOWN']
report("PDH rejection (near but hold below) → short", pdh_reject, -1)

# ══════════════════════════════════════════
# 3. ASIA SESSION RANGE  (00:00-07:00 UTC)
# ══════════════════════════════════════════
print("\n### 3. Asia range — London fake-break & breakout ###")

asia_range={}
for d in days:
    mask=np.where((dy==d) & (hr>=0) & (hr<7))[0]
    if len(mask)>=3:
        asia_range[d]=(h[mask].max(), l[mask].min())

# London open (07:00-10:00): fake-break of Asia high → short
asia_fake_hi=[]; asia_fake_lo=[]
for i in range(n):
    if hr[i] not in (7,8,9,10): continue
    d=dy[i]
    if d not in asia_range: continue
    ah,al=asia_range[d]
    if not np.isnan(at[i]) and at[i]>0:
        # fake break high: spike above asia high, close back inside
        if h[i]>ah and c[i]<ah and reg[i] in ('UP','DOWN'):
            asia_fake_hi.append(i)
        # fake break low: spike below asia low, close back inside
        if l[i]<al and c[i]>al and reg[i] in ('UP','DOWN'):
            asia_fake_lo.append(i)

report("Asia high fake-break London → short", asia_fake_hi, -1)
report("Asia low fake-break London → long",   asia_fake_lo, +1)

# Clean breakout of Asia high (close above, momentum)
asia_break_hi=[i for i in range(n)
               if hr[i] in (7,8,9) and dy[i] in asia_range
               and not np.isnan(at[i]) and at[i]>0
               and c[i]>asia_range[dy[i]][0]
               and reg[i]=='UP']
report("Asia high clean break → long (UP)", asia_break_hi, +1)
asia_break_lo=[i for i in range(n)
               if hr[i] in (7,8,9) and dy[i] in asia_range
               and not np.isnan(at[i]) and at[i]>0
               and c[i]<asia_range[dy[i]][1]
               and reg[i]=='DOWN']
report("Asia low clean break → short (DOWN)", asia_break_lo, -1)

# ══════════════════════════════════════════
# 4. ROUND NUMBERS  (xx00, xx50)
# ══════════════════════════════════════════
print("\n### 4. Round numbers (xx00 / xx50) — rejection ###")

def nearest_round(price, step):
    return round(price/step)*step

# Upper wick touches round level and closes below
rn00_reject=[]; rn50_reject=[]
for i in range(1,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.25*at[i]
    # find nearest xx00 above midpoint of bar
    mid=(h[i]+l[i])/2
    nr00=nearest_round(mid,100)
    nr50=nearest_round(mid,50)
    if abs(h[i]-nr00)<tol and c[i]<nr00 and reg[i] in ('UP','DOWN'):
        rn00_reject.append(i)
    if abs(h[i]-nr50)<tol and c[i]<nr50 and reg[i] in ('UP','DOWN') and (nr50%100)!=0:
        rn50_reject.append(i)

report("Round xx00 wick-reject → short", rn00_reject, -1)
report("Round xx50 wick-reject → short", rn50_reject, -1)

# Support bounce at round number (price approaches from above, wick below, close above)
rn00_support=[]; rn50_support=[]
for i in range(1,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.25*at[i]
    mid=(h[i]+l[i])/2
    nr00=nearest_round(mid,100)
    nr50=nearest_round(mid,50)
    if abs(l[i]-nr00)<tol and c[i]>nr00 and reg[i] in ('UP','DOWN'):
        rn00_support.append(i)
    if abs(l[i]-nr50)<tol and c[i]>nr50 and reg[i] in ('UP','DOWN') and (nr50%100)!=0:
        rn50_support.append(i)

report("Round xx00 support bounce → long", rn00_support, +1)
report("Round xx50 support bounce → long", rn50_support, +1)

# ══════════════════════════════════════════
# 5. VOLUME CLIMAX / EXHAUSTION
# ══════════════════════════════════════════
print("\n### 5. Volume climax & exhaustion ###")

# rolling z-score of volume
vol_ma=pd.Series(v).rolling(20).mean().values
vol_sd=pd.Series(v).rolling(20).std().values
vol_z=(v-vol_ma)/np.where(vol_sd>0,vol_sd,np.nan)

# Volume spike (z>2) on a bearish bar → selling climax → long fade
# Volume spike on bullish bar → buying climax → short fade
vol_climax_bear=[i for i in range(1,n)
                 if not np.isnan(vol_z[i]) and vol_z[i]>2
                 and c[i]<o[i]  # bearish bar (selling pressure)
                 and reg[i] in ('UP','DOWN')]
report("Vol climax (z>2) bearish bar → long (exhaustion)", vol_climax_bear, +1)

vol_climax_bull=[i for i in range(1,n)
                 if not np.isnan(vol_z[i]) and vol_z[i]>2
                 and c[i]>o[i]  # bullish bar
                 and reg[i] in ('UP','DOWN')]
report("Vol climax (z>2) bullish bar → short (exhaustion)", vol_climax_bull, -1)

# Higher threshold
for z_thr in [2.5, 3.0]:
    s=[i for i in range(1,n)
       if not np.isnan(vol_z[i]) and vol_z[i]>z_thr
       and c[i]>o[i] and reg[i] in ('UP','DOWN')]
    report(f"Vol climax (z>{z_thr}) bullish → short", s, -1)

# Vol spike (z>2) on DOWN-sweep candle (combo with flagship)
from itertools import chain
sh2=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh2[i]=True
shi2=np.where(sh2)[0]
sweep_vol=[i for i in range(2,n)
           if not np.isnan(vol_z[i]) and vol_z[i]>1.5
           and reg[i] in ('UP','DOWN')
           and not np.isnan(at[i]) and at[i]>0]
psh_last={} # precompute for speed
sweep_vol_final=[]
for i in sweep_vol:
    ps=shi2[shi2<i-1]
    if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]]:
        sweep_vol_final.append(i)
report("Flagship sweep + vol z>1.5 (big-money sweep) → short", sweep_vol_final, -1)

# ══════════════════════════════════════════
# 6. ORDER BLOCK COMBO: PDH + Volume
# ══════════════════════════════════════════
print("\n### 6. PDH/PDL + volume combo ###")

pdh_vol=[i for i in pdh_sweep
         if not np.isnan(vol_z[i]) and vol_z[i]>1.0]
report("PDH sweep + vol z>1.0 → short", pdh_vol, -1)

pdh_vol2=[i for i in pdh_sweep
          if not np.isnan(vol_z[i]) and vol_z[i]>1.5]
report("PDH sweep + vol z>1.5 → short", pdh_vol2, -1)

# ══════════════════════════════════════════
# 7. SUMMARY vs BETA
# ══════════════════════════════════════════
print(f"\n{'='*100}")
print(f"# BETA benchmark: EV+{BETA_EV:.2f}  (short every SLOPING H1 bar naively)")
print(f"# 🔥BEATS_BETA = EV > {BETA_EV:.2f}.  ✅STABLE = both halves positive EV.")
print(f"{'='*100}")
