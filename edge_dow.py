"""
DOW THEORY — statistical test on H1 XAUUSD
============================================
Pure Dow Theory:
  Trend = confirmed swing highs/lows (HH+HL=UP, LH+LL=DOWN, else NEUTRAL)
  Causal only — trend label uses swings confirmed BEFORE current bar.

4 strategies:
  A) Pullback to prior LH (in DOWN trend) → short
     "Price rallies to the last lower-high level, fade it"
  B) Breakout below prior LL (in DOWN trend) → short continuation
     "New low confirmed, trend continues"
  C) Trend reversal: first LH in an UP trend → short
     "Dow says trend broken, get short"
  D) Pure trend direction filter (HH+HL=skip short, LH+LL=short)
     applied to all bars (Dow as regime filter vs MA200)

Also test LONG mirror for completeness.
1:1 ATR + asymmetric SL3/TP10. SPREAD=0.3. Half-split stability.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

def load(p):
    df=pd.read_csv(p)
    df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
    return df

def calc_atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a

def swings(h,l,n,w=1):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values; c=h1['c'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
mon=h1['Date'].dt.strftime('%Y-%m').values

sh_arr,sl_arr=swings(h,l,n,w=1)
shi=np.where(sh_arr)[0]; sli=np.where(sl_arr)[0]

# ─────────────────────────────────────────────
# Causal Dow trend at each bar
# Uses only swing highs/lows confirmed before bar i
# ─────────────────────────────────────────────
dow_trend=np.array(['NA']*n, dtype=object)
# track last 2 swing highs and lows
for i in range(n):
    psh=shi[shi<i]; psl=sli[sli<i]
    if len(psh)<2 or len(psl)<2: continue
    # last 2 confirmed swing highs
    sh1_v=h[psh[-2]]; sh2_v=h[psh[-1]]   # sh2 is more recent
    sl1_v=l[psl[-2]]; sl2_v=l[psl[-1]]
    hh=(sh2_v>sh1_v); hl=(sl2_v>sl1_v)
    lh=(sh2_v<sh1_v); ll=(sl2_v<sl1_v)
    if hh and hl:   dow_trend[i]='UP'
    elif lh and ll: dow_trend[i]='DOWN'
    elif hh or hl:  dow_trend[i]='WEAK_UP'
    elif lh or ll:  dow_trend[i]='WEAK_DOWN'
    else:           dow_trend[i]='NEUTRAL'

# ─────────────────────────────────────────────
# Trade engine
# ─────────────────────────────────────────────
def trade_sym(i,side=-1,horizon=24):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]
    tp=entry+rng*side; sl=entry-rng*side
    for k in range(i+1,min(n,i+1+horizon)):
        if side<0:
            if h[k]>=sl: return -rng-SPREAD
            if l[k]<=tp: return  rng-SPREAD
        else:
            if l[k]<=sl: return -rng-SPREAD
            if h[k]>=tp: return  rng-SPREAD
    return None

def trade_asym(i,side=-1,sl_pt=3,tp_pt=10,horizon=48):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]
    tp=entry+tp_pt*side; sl=entry-sl_pt*side
    for k in range(i+1,min(n,i+1+horizon)):
        if side<0:
            if h[k]>=sl: return -sl_pt-SPREAD
            if l[k]<=tp: return  tp_pt-SPREAD
        else:
            if l[k]<=sl: return -sl_pt-SPREAD
            if h[k]>=tp: return  tp_pt-SPREAD
    return None

def report(label, idxs, side=-1, asym=False, sl=3, tp=10, min_n=8):
    fn=(lambda i: trade_asym(i,side,sl,tp)) if asym else (lambda i: trade_sym(i,side))
    res=[(fn(i),i) for i in idxs]; res=[(p,i) for p,i in res if p is not None]
    if len(res)<min_n:
        print(f"  {label:<65} n={len(res)} (少)"); return None
    ps=[p for p,_ in res]; ids=[i for _,i in res]
    wr=100*np.mean([p>0 for p in ps]); ev=np.mean(ps); tot=sum(ps)
    mcl=cur=0
    for p in ps:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    mths=len(set(mon[i] for i in ids)); tpm=len(ps)/max(mths,1)
    mid=n//2
    h1p=[p for p,i in zip(ps,ids) if i<mid]
    h2p=[p for p,i in zip(ps,ids) if i>=mid]
    def hw(x): return f"{100*np.mean([p>0 for p in x]):.0f}%/{len(x)}" if len(x)>=6 else f"-/{len(x)}"
    stab=""
    if len(h1p)>=6 and len(h2p)>=6:
        stab="✅STABLE" if np.mean(h1p)>0 and np.mean(h2p)>0 else "❌unstable"
    beat="🔥" if (not asym and ev>1.25) or (asym and ev>0) else ""
    rr="[asym]" if asym else "[1:1] "
    print(f"  {rr} {label:<62} WR={wr:4.0f}% EV={ev:+6.2f} n={len(ps):4d} ~{tpm:.0f}/mo MCL={mcl}  {hw(h1p)},{hw(h2p)}  {stab}{beat}")
    return res

print("="*100)
print("# DOW THEORY — H1 XAUUSD  (1:1ATR & SL3/TP10両方)")
print("="*100)

# ダウトレンド分布確認
print("\n=== Dow Trend 分布 ===")
from collections import Counter
dist=Counter(dow_trend)
for k in ['UP','WEAK_UP','NEUTRAL','WEAK_DOWN','DOWN','NA']:
    pct=100*dist[k]/n
    print(f"  {k:<12}: {dist[k]:5d}本 ({pct:.0f}%)")

# ══════════════════════════════════════════
# A) プルバックエントリー
#    DOWN/WEAK_DOWN中に直前のLH水準へ戻り → ショート
# ══════════════════════════════════════════
print("\n### A. プルバックエントリー (LH水準への戻り→ショート) ###")
print("    ダウ下降中: 直前スイングハイ(=LH)に価格が戻ったら売る")

pullback_short=[]; pullback_tol_list=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    if dow_trend[i] not in ('DOWN','WEAK_DOWN'): continue
    psh=shi[shi<i]
    if len(psh)<2: continue
    lh_level=h[psh[-1]]   # most recent swing high (= Lower High in downtrend)
    tol=0.20*at[i]
    # price approached LH level (within tol) and still closed below it
    if abs(h[i]-lh_level)<=tol and c[i]<lh_level:
        pullback_short.append(i)

report("DOWN: LH水準プルバック→ショート", pullback_short, -1)
report("DOWN: LH水準プルバック→ショート", pullback_short, -1, asym=True)

# stricter: only pure DOWN (not WEAK)
pb_strict=[i for i in pullback_short if dow_trend[i]=='DOWN']
report("DOWN(純粋): LH水準プルバック→ショート", pb_strict, -1)
report("DOWN(純粋): LH水準プルバック [SL3/TP10]", pb_strict, -1, asym=True)

# with MA200 below (trend alignment)
pb_ma=[i for i in pullback_short if not np.isnan(ma200[i]) and c[i]<ma200[i]]
report("DOWN + MA200下: LH水準プルバック→ショート", pb_ma, -1)
report("DOWN + MA200下: LH水準プルバック [SL3/TP10]", pb_ma, -1, asym=True)

# ── mirror: UP trend pullback to HL → long ──
pullback_long=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    if dow_trend[i] not in ('UP','WEAK_UP'): continue
    psl=sli[sli<i]
    if len(psl)<2: continue
    hl_level=l[psl[-1]]
    tol=0.20*at[i]
    if abs(l[i]-hl_level)<=tol and c[i]>hl_level:
        pullback_long.append(i)

report("UP: HL水準プルバック→ロング", pullback_long, +1)
report("UP: HL水準プルバック [SL3/TP10]", pullback_long, +1, asym=True)

# ══════════════════════════════════════════
# B) ブレイクアウトエントリー
#    直前LL割れ→ショート継続 / HH超え→ロング継続
# ══════════════════════════════════════════
print("\n### B. ブレイクアウトエントリー (新安値/新高値で乗る) ###")
print("    ダウ理論: 新しいLLを作った=トレンド確認 → 継続ショート")

breakout_short=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    if dow_trend[i] not in ('DOWN','WEAK_DOWN'): continue
    psl=sli[sli<i]
    if not len(psl): continue
    prev_ll=l[psl[-1]]
    # current bar closes below prior LL (new LL confirmed → Dow breakdown)
    if c[i]<prev_ll and l[i]<prev_ll:
        breakout_short.append(i)

report("DOWN: 新LL確認→継続ショート", breakout_short, -1)
report("DOWN: 新LL確認→継続 [SL3/TP10]", breakout_short, -1, asym=True)

# with volume
vol_ma=pd.Series(h1['v'].values).rolling(20).mean().values
vol_sd2=pd.Series(h1['v'].values).rolling(20).std().values
vol_z=(h1['v'].values-vol_ma)/np.where(vol_sd2>0,vol_sd2,np.nan)
bo_vol=[i for i in breakout_short if not np.isnan(vol_z[i]) and vol_z[i]>0.5]
report("DOWN: 新LL + vol z>0.5 → 継続ショート", bo_vol, -1)
report("DOWN: 新LL + vol z>0.5 [SL3/TP10]", bo_vol, -1, asym=True)

breakout_long=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    if dow_trend[i] not in ('UP','WEAK_UP'): continue
    psh=shi[shi<i]
    if not len(psh): continue
    if c[i]>h[psh[-1]] and h[i]>h[psh[-1]]:
        breakout_long.append(i)

report("UP: 新HH確認→継続ロング", breakout_long, +1)
report("UP: 新HH確認→継続 [SL3/TP10]", breakout_long, +1, asym=True)

# ══════════════════════════════════════════
# C) トレンド転換シグナル
#    上昇中の初のLH → ショート (「ダウ転換」エントリー)
#    下降中の初のHL → ロング
# ══════════════════════════════════════════
print("\n### C. トレンド転換シグナル ###")
print("    上昇(UP)中に初めてLHが出た → 転換の初動でショート")

reversal_short=[]; reversal_long=[]
prev_trend='NA'
for i in range(n):
    if dow_trend[i]=='NA': prev_trend='NA'; continue
    # UP → WEAK_DOWN or DOWN (first sign of weakness)
    if prev_trend in ('UP','WEAK_UP') and dow_trend[i] in ('WEAK_DOWN','DOWN'):
        if not np.isnan(at[i]) and at[i]>0:
            reversal_short.append(i)
    # DOWN → WEAK_UP or UP
    if prev_trend in ('DOWN','WEAK_DOWN') and dow_trend[i] in ('WEAK_UP','UP'):
        if not np.isnan(at[i]) and at[i]>0:
            reversal_long.append(i)
    prev_trend=dow_trend[i]

report("UP→DOWN転換の初動→ショート", reversal_short, -1)
report("UP→DOWN転換の初動 [SL3/TP10]", reversal_short, -1, asym=True)
report("DOWN→UP転換の初動→ロング", reversal_long, +1)
report("DOWN→UP転換の初動 [SL3/TP10]", reversal_long, +1, asym=True)

# ══════════════════════════════════════════
# D) ダウトレンドフィルター (MA200の代わりに使う)
#    DOWN/WEAK_DOWN の時だけ全バーショート可 → ベースレート比較
# ══════════════════════════════════════════
print("\n### D. ダウトレンドフィルター vs MA200フィルター 比較 ###")
print("    「MA200が傾いてる」vs「ダウ理論でDOWN」どちらが優秀か")

# Dow filter baseline
dow_all=[i for i in range(n) if dow_trend[i] in ('DOWN','WEAK_DOWN')]
report("ダウDOWN中 毎足ショート(ベースレート)", dow_all, -1)

dow_strict=[i for i in range(n) if dow_trend[i]=='DOWN']
report("ダウ純DOWN中 毎足ショート(ベースレート)", dow_strict, -1)

# MA200 regime baseline (for comparison)
from numpy import array
slope_k=50; ma200_base=pd.Series(c).rolling(200).mean().values
reg_arr=array(['NA']*n,dtype=object)
for i in range(n):
    if i<200+slope_k or np.isnan(ma200_base[i]) or np.isnan(ma200_base[i-slope_k]): continue
    s=(ma200_base[i]-ma200_base[i-slope_k])/ma200_base[i-slope_k]
    if   s> 0.003 and c[i]>ma200_base[i]: reg_arr[i]='UP'
    elif s<-0.003 and c[i]<ma200_base[i]: reg_arr[i]='DOWN'
    else: reg_arr[i]='RANGE'
ma200_base_sigs=[i for i in range(n) if reg_arr[i] in ('UP','DOWN')]
report("MA200 SLOPING中 毎足ショート(比較用)", ma200_base_sigs, -1)

# ══════════════════════════════════════════
# E) 組み合わせ: ダウ × 旗艦スイープ × MA200
# ══════════════════════════════════════════
print("\n### E. ダウ理論 × 旗艦スイープ の組み合わせ ###")

# rebuild flagship
sh2=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh2[i]=True
shi2=np.where(sh2)[0]

sweep_dow=[]; sweep_dow_ma=[]
for i in range(2,n):
    ps=shi2[shi2<i-1]
    if not len(ps): continue
    if h[i]>h[ps[-1]] and c[i]<h[ps[-1]]:
        if dow_trend[i] in ('DOWN','WEAK_DOWN'):
            sweep_dow.append(i)
        if dow_trend[i] in ('DOWN','WEAK_DOWN') and not np.isnan(ma200[i]) and c[i]<ma200[i]:
            sweep_dow_ma.append(i)

report("旗艦スイープ × ダウDOWN", sweep_dow, -1)
report("旗艦スイープ × ダウDOWN [SL3/TP10]", sweep_dow, -1, asym=True)
report("旗艦スイープ × ダウDOWN × MA200下", sweep_dow_ma, -1)
report("旗艦スイープ × ダウDOWN × MA200下 [SL3/TP10]", sweep_dow_ma, -1, asym=True)

# ダウ vs MA200 フィルタ on sweep: which is better?
sweep_ma200=[i for i in range(2,n)
             if len(shi2[shi2<i-1]) and h[i]>h[shi2[shi2<i-1][-1]]
             and c[i]<h[shi2[shi2<i-1][-1]]
             and reg_arr[i] in ('UP','DOWN')]
report("旗艦スイープ × MA200 SLOPING(現行)", sweep_ma200, -1)

print("\n" + "="*100)
print("# ダウ理論まとめ判定基準:")
print("# プルバック・ブレイクアウト・転換の3戦略 × ショート/ロング × 1:1/SL3TP10")
print("="*100)
