"""
XENO METHOD — Statistical test
Multi-TF rules:
  Direction (1H): cutting higher (HH+HL) → long bias; cutting lower (LH+LL) → short bias
  MA200 filter: price above → long bias, below → short bias
  Conflict: if 1H direction contradicts MA200 → skip
  Cancel rule: if 1H breaks recent high → cancel short bias (and vice versa)
  Both 1H + 15m cutting same way → skip counter entries

Entry (M15/M5 level):
  1. Decline origin (下落起点) = last swing high that broke down (for shorts)
  2. Pullback high (戻り高値) = prior swing high in current downtrend move

Exit: 1:1 ATR, horizon 48 bars (M15).
No look-ahead. SPREAD=0.3. Half-split stability.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
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

def swings(h,l,n,w=1):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

def h1_trend_at(ts_query, h1_ts, h1_trend):
    """Get H1 trend for a given M15 timestamp (no look-ahead)."""
    idx=np.searchsorted(h1_ts, ts_query, side='right')-1
    if idx<0: return 'NA'
    return h1_trend[idx]

# =====================================================================
# Compute causal H1 trend: HH+HL = UP, LH+LL = DOWN, else NEUTRAL
# Uses confirmed swing highs/lows (look-back only)
# =====================================================================
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
n1=len(h1)
o1=h1['o'].values; h1h=h1['h'].values; h1l=h1['l'].values; h1c=h1['c'].values
a1=atr(h1h,h1l,h1c); h1['atr']=a1
ma200_h1=pd.Series(h1c).rolling(200).mean().values

sh1,sl1=swings(h1h,h1l,n1,w=1)
shi1=np.where(sh1)[0]; sli1=np.where(sl1)[0]

# Causal trend: at each bar, look at last 2 confirmed swing highs and lows
h1_trend=np.array(['NA']*n1, dtype=object)
for i in range(n1):
    past_sh=shi1[shi1<i]
    past_sl=sli1[sli1<i]
    if len(past_sh)<2 or len(past_sl)<2: continue
    # last two swing highs and lows
    sh_a=h1h[past_sh[-2]]; sh_b=h1h[past_sh[-1]]
    sl_a=h1l[past_sl[-2]]; sl_b=h1l[past_sl[-1]]
    hh=sh_b>sh_a; hl=sl_b>sl_a
    lh=sh_b<sh_a; ll=sl_b<sl_a
    if hh and hl: h1_trend[i]='UP'
    elif lh and ll: h1_trend[i]='DOWN'
    elif hh or hl: h1_trend[i]='WEAK_UP'
    elif lh or ll: h1_trend[i]='WEAK_DOWN'
    else: h1_trend[i]='NEUTRAL'

# MA200 H1 bias
h1_ma_bias=np.array(['NA']*n1, dtype=object)
for i in range(n1):
    if np.isnan(ma200_h1[i]): continue
    h1_ma_bias[i]='UP' if h1c[i]>ma200_h1[i] else 'DOWN'

h1_ts=h1['Date'].values

# =====================================================================
# M15: compute local swing highs/lows for entry points
# =====================================================================
m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv")
n15=len(m15)
o15=m15['o'].values; h15=m15['h'].values; l15=m15['l'].values; c15=m15['c'].values
a15=atr(h15,l15,c15); m15['atr']=a15
ma200_m15=pd.Series(c15).rolling(200).mean().values
ts15=m15['Date'].values
mon15=m15['Date'].dt.strftime('%Y-%m').values
sh15,sl15=swings(h15,l15,n15,w=1)
shi15=np.where(sh15)[0]
sli15=np.where(sl15)[0]

def get_h1_state(i):
    """Get H1 trend and MA bias for M15 bar i (no look-ahead)."""
    ts=ts15[i]
    idx=np.searchsorted(h1_ts, ts, side='right')-1
    if idx<0: return 'NA','NA'
    return h1_trend[idx], h1_ma_bias[idx]

# =====================================================================
# XENO SHORT entries
# Conditions:
#   1. H1 trend = DOWN or WEAK_DOWN
#   2. H1 MA200 bias = DOWN (price below MA200)
#   3. M15: swing-high swept (sweep of prior swing high = price > prior SH then closes below) ← decline origin
#      OR: M15 pullback high (price rallied to prior swing high zone, rejection)
# Cancel: if H1 trend = UP or WEAK_UP (recent high broken)
# =====================================================================

signals_short=[]  # (bar_idx, entry_type)
signals_long=[]

for i in range(5, n15):
    if np.isnan(a15[i]) or a15[i]<=0: continue
    trend, ma_bias = get_h1_state(i)
    if trend=='NA' or ma_bias=='NA': continue

    # SHORT bias conditions
    short_ok = (trend in ('DOWN','WEAK_DOWN')) and (ma_bias=='DOWN')
    long_ok   = (trend in ('UP','WEAK_UP'))   and (ma_bias=='UP')

    if short_ok:
        # Entry type 1: decline origin = sweep of prior swing high (wick above + close below)
        past_sh=shi15[shi15<i-1]
        if len(past_sh):
            lv=h15[past_sh[-1]]
            if h15[i]>lv and c15[i]<lv:
                signals_short.append((i,'decline_origin'))
                continue
        # Entry type 2: pullback high = price rallied to near prior swing high (within 0.3ATR) and closed below
        if len(past_sh):
            lv=h15[past_sh[-1]]
            if abs(h15[i]-lv)<0.3*a15[i] and c15[i]<lv:
                signals_short.append((i,'pullback_high'))
                continue

    if long_ok:
        past_sl=sli15[sli15<i-1]
        if len(past_sl):
            lv=l15[past_sl[-1]]
            if l15[i]<lv and c15[i]>lv:
                signals_long.append((i,'decline_origin'))
                continue
        if len(past_sl):
            lv=l15[past_sl[-1]]
            if abs(l15[i]-lv)<0.3*a15[i] and c15[i]>lv:
                signals_long.append((i,'pullback_high'))
                continue

def trade(i, side, horizon=48):
    if i+1>=n15 or np.isnan(a15[i]) or a15[i]<=0: return None
    entry=o15[i+1]; rng=a15[i]
    tp=entry+rng*side; slv=entry-rng*side
    for k in range(i+1, min(n15, i+1+horizon)):
        if side<0:
            if h15[k]>=slv: return (False,-rng-SPREAD)
            if l15[k]<=tp: return (True,rng-SPREAD)
        else:
            if l15[k]<=slv: return (False,-rng-SPREAD)
            if h15[k]>=tp: return (True,rng-SPREAD)
    return None

def report(label, sig_list, side):
    tr=[(trade(i,side),i,et) for i,et in sig_list]
    tr=[(x,i,et) for x,i,et in tr if x]
    if len(tr)<5:
        print(f"  {label:<52} n={len(tr)} too few"); return
    res=[x[0] for x in tr]; idxs=[x[1] for x in tr]
    wr=100*np.mean([r[0] for r in res]); ev=np.mean([r[1] for r in res])
    tot=sum(r[1] for r in res)
    mcl=cur=0
    for r in res:
        if not r[0]: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    months=len(set(mon15[i] for i in idxs)); tpm=len(res)/max(months,1)
    mid=n15//2
    h1_=[r for r,i,_ in zip(res,idxs,[0]*len(idxs)) if i<mid]
    h2_=[r for r,i,_ in zip(res,idxs,[0]*len(idxs)) if i>=mid]
    def hw(x): return f"{100*np.mean([r[0] for r in x]):.0f}%/{len(x)}" if len(x)>=5 else "-"
    stable=""
    if len(h1_)>=5 and len(h2_)>=5:
        stable="STABLE" if np.mean([r[1] for r in h1_])>0 and np.mean([r[1] for r in h2_])>0 else "unstable"
    print(f"  {label:<52} WR={wr:.0f}% EV={ev:+.2f} n={len(res):4d} tot={tot:+.0f} MCL={mcl} ~{tpm:.0f}/mo  half={hw(h1_)},{hw(h2_)}  {stable}")

    # monthly
    for m in sorted(set(mon15[i] for i in idxs)):
        sub_r=[r for r,i,_ in zip(res,idxs,[0]*len(idxs)) if mon15[i]==m]
        if len(sub_r)>=3:
            print(f"     {m}: WR={100*np.mean([r[0] for r in sub_r]):.0f}% n={len(sub_r)}")

print("#"*90)
print("# XENO METHOD — M15 entry, H1 direction filter + MA200")
print("#"*90)

print("\n=== SHORT (H1 DOWN + MA200 DOWN + M15 sweep/pullback) ===")
report("ALL short signals", signals_short, -1)
do_sig=[(i,et) for i,et in signals_short if et=='decline_origin']
pb_sig=[(i,et) for i,et in signals_short if et=='pullback_high']
report("Short: decline origin only (下落起点)", do_sig, -1)
report("Short: pullback high only (戻り高値)", pb_sig, -1)

print("\n=== LONG (H1 UP + MA200 UP + M15 sweep/pullback) ===")
report("ALL long signals", signals_long, +1)
do_long=[(i,et) for i,et in signals_long if et=='decline_origin']
pb_long=[(i,et) for i,et in signals_long if et=='pullback_high']
report("Long: decline origin only", do_long, +1)
report("Long: pullback high only", pb_long, +1)

# Relaxed: WEAK trend also qualifies
signals_short2=[]; signals_long2=[]
for i in range(5, n15):
    if np.isnan(a15[i]) or a15[i]<=0: continue
    trend, ma_bias = get_h1_state(i)
    if trend=='NA' or ma_bias=='NA': continue
    short_ok = (trend in ('DOWN','WEAK_DOWN'))  # MA200 not required
    long_ok   = (trend in ('UP','WEAK_UP'))
    if short_ok:
        past_sh=shi15[shi15<i-1]
        if len(past_sh):
            lv=h15[past_sh[-1]]
            if h15[i]>lv and c15[i]<lv: signals_short2.append((i,'decline_origin')); continue
            if abs(h15[i]-lv)<0.3*a15[i] and c15[i]<lv: signals_short2.append((i,'pullback_high')); continue
    if long_ok:
        past_sl=sli15[sli15<i-1]
        if len(past_sl):
            lv=l15[past_sl[-1]]
            if l15[i]<lv and c15[i]>lv: signals_long2.append((i,'decline_origin')); continue
            if abs(l15[i]-lv)<0.3*a15[i] and c15[i]>lv: signals_long2.append((i,'pullback_high')); continue

print("\n=== RELAXED (H1 trend only, no MA200 requirement) ===")
report("Short relaxed", signals_short2, -1)
report("Long relaxed", signals_long2, +1)

print("\n=== H1 TREND DISTRIBUTION ===")
for t in ['UP','WEAK_UP','NEUTRAL','WEAK_DOWN','DOWN','NA']:
    cnt=np.sum(h1_trend==t)
    print(f"  {t:<12}: {cnt} bars ({100*cnt/n1:.0f}%)")
print(f"\n  Total SHORT signals found: {len(signals_short)} (strict) / {len(signals_short2)} (relaxed)")
print(f"  Total LONG  signals found: {len(signals_long)} (strict) / {len(signals_long2)} (relaxed)")
