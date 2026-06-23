"""
UNEXPLORED EDGES — 5pt+ focus, all untested angles
From chart observation + CSV:
  A) 曜日効果 (月〜金)
  B) ディスプレイスメント大足 (1本で2ATR+移動) → 半値戻しショート / 継続
  C) ボラティリティ局面 (ATRパーセンタイル高/低) × 既存エッジ
  D) スクイーズ→ブレイク (直近N本のATR収縮後の膨張)
  E) 連続同色足 (3本連続陰線→継続 or 逆張り)
  F) 前週高値/安値スイープ (PDH/PDL の週版)
  G) オープニングギャップ (週間ギャップ)
  H) 大足後の戻りショート (ディスプレイスメント後の最初のプルバック)
  I) FVG(フェアバリューギャップ) → 埋め後のショート継続
  J) 既存エッジ × 曜日フィルタ重ね合わせ

全て 1:1 ATR (horizon=24) + 非対称 SL3/TP10 の両方でテスト
SPREAD=0.3, no look-ahead, half-split stability
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

def regime(c, k=50):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+k or np.isnan(ma200[i]) or np.isnan(ma200[i-k]): continue
        s=(ma200[i]-ma200[i-k])/ma200[i-k]
        if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values; c=h1['c'].values; v=h1['v'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values
reg=regime(c)
hr=h1['Date'].dt.hour.values
dow=h1['Date'].dt.dayofweek.values   # 0=Mon 4=Fri
mon=h1['Date'].dt.strftime('%Y-%m').values
vol_ma=pd.Series(v).rolling(20).mean().values
vol_sd=pd.Series(v).rolling(20).std().values
vol_z=(v-vol_ma)/np.where(vol_sd>0,vol_sd,np.nan)

# ATR percentile (rolling 200-bar)
atr_pct=np.full(n,np.nan)
for i in range(200,n):
    window=at[i-200:i]
    window=window[~np.isnan(window)]
    if len(window)>0: atr_pct[i]=np.sum(window<=at[i])/len(window)

sh_arr=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh_arr[i]=True
shi=np.where(sh_arr)[0]
sl_arr=np.zeros(n,bool)
for i in range(1,n-1):
    if l[i]==l[i-1:i+2].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl_arr[i]=True
sli=np.where(sl_arr)[0]

# ── TRADE ENGINE ──
def trade_sym(i, side=-1, horizon=24):
    """1:1 ATR symmetric"""
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]
    tp=entry+rng*side; sl=entry-rng*side
    for k in range(i+1,min(n,i+1+horizon)):
        if side<0:
            if h[k]>= sl: return -rng-SPREAD
            if l[k] <= tp: return  rng-SPREAD
        else:
            if l[k] <= sl: return -rng-SPREAD
            if h[k] >= tp: return  rng-SPREAD
    return None

def trade_asym(i, side=-1, sl_pt=3, tp_pt=10, horizon=48):
    """Fixed-pt asymmetric SL/TP"""
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]
    tp=entry+tp_pt*side; sl=entry-sl_pt*side
    for k in range(i+1,min(n,i+1+horizon)):
        if side<0:
            if h[k]>= sl: return -sl_pt-SPREAD
            if l[k] <= tp: return  tp_pt-SPREAD
        else:
            if l[k] <= sl: return -sl_pt-SPREAD
            if h[k] >= tp: return  tp_pt-SPREAD
    return None

def report(label, idxs, side=-1, asym=False, sl=3, tp=10, min_n=10):
    fn = (lambda i: trade_asym(i,side,sl,tp)) if asym else (lambda i: trade_sym(i,side))
    pnls=[(fn(i),i) for i in idxs]; pnls=[(p,i) for p,i in pnls if p is not None]
    if len(pnls)<min_n:
        print(f"  {label:<62} n={len(pnls)} (少)")
        return
    ps=[p for p,_ in pnls]; ids=[i for _,i in pnls]
    wr=100*np.mean([p>0 for p in ps]); ev=np.mean(ps); tot=sum(ps)
    mcl=cur=0
    for p in ps:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    mths=len(set(mon[i] for i in ids)); tpm=len(ps)/max(mths,1)
    mid=n//2
    h1p=[p for p,i in zip(ps,ids) if i<mid]
    h2p=[p for p,i in zip(ps,ids) if i>=mid]
    def hw(x): return f"{100*np.mean([p>0 for p in x]):.0f}%/{len(x)}" if len(x)>=8 else f"-/{len(x)}"
    stab=""
    if len(h1p)>=8 and len(h2p)>=8:
        stab="✅STABLE" if np.mean(h1p)>0 and np.mean(h2p)>0 else "❌unstable"
    rr_tag="[asym SL{sl}/TP{tp}]" if asym else "[1:1ATR]"
    flag="🔥" if (ev>1.25 and not asym) or (ev>0 and asym) else ""
    print(f"  {label:<62} WR={wr:4.0f}% EV={ev:+5.2f} n={len(ps):4d} ~{tpm:.0f}/mo MCL={mcl}  {hw(h1p)},{hw(h2p)}  {stab}{flag}")

print("="*100)
print("# 未検証エッジ総当たり — 5pt+ focus (SL3/TP10 & 1:1 ATR)")
print("="*100)

# ══════════════════════════════════════════
# A) 曜日効果
# ══════════════════════════════════════════
print("\n### A. 曜日効果 (ショート, 1:1 ATR, SLOPING) ###")
dow_names=['月','火','水','木','金']
for d in range(5):
    sigs=[i for i in range(n) if dow[i]==d and reg[i] in ('UP','DOWN')]
    report(f"  {dow_names[d]}曜日 ショート", sigs, -1)

print("\n### A2. 曜日 × hour10 ショート ###")
for d in range(5):
    sigs=[i for i in range(n) if dow[i]==d and hr[i]==10 and reg[i] in ('UP','DOWN')]
    report(f"  {dow_names[d]}曜 × hour10", sigs, -1, min_n=5)

# ══════════════════════════════════════════
# B) ディスプレイスメント大足 → 半値戻しショート
# ══════════════════════════════════════════
print("\n### B. ディスプレイスメント大足 (1本 ≥ 2ATR の陰線) ###")
print("    チャートで見えた『大きな陰線→短い戻り→継続』パターンの数値化")

# B1: 直近に大陰線(≥2ATR body)があり、現在バーがその後の戻り → ショート
for thresh, label in [(1.5,'≥1.5ATR大陰線後の戻りショート'),
                       (2.0,'≥2.0ATR大陰線後の戻りショート'),
                       (2.5,'≥2.5ATR大陰線後の戻りショート')]:
    sigs=[]
    for i in range(5,n):
        if np.isnan(at[i]) or at[i]<=0: continue
        if reg[i] not in ('UP','DOWN'): continue
        # look back 1-5 bars for a displacement candle
        for j in range(i-1,max(0,i-6),-1):
            if np.isnan(at[j]) or at[j]<=0: continue
            body=o[j]-c[j]  # positive = bearish
            if body >= thresh*at[j]:  # big bearish bar
                # current bar is a pullback (higher close than prior bar)
                if c[i]>c[i-1] and c[i]<o[j]:  # retrace but still below displacement open
                    sigs.append(i)
                break
    report(label, list(set(sigs)), -1)
    report(label+" [SL3/TP10]", list(set(sigs)), -1, asym=True, sl=3, tp=10)

# B2: 大陰線の翌足を継続ショート (モメンタム)
for thresh in [2.0, 2.5]:
    sigs=[i for i in range(1,n)
          if not np.isnan(at[i-1]) and at[i-1]>0
          and (o[i-1]-c[i-1])>=thresh*at[i-1]   # prior big bear
          and reg[i] in ('UP','DOWN')]
    report(f"大陰線(≥{thresh}ATR)翌足 継続ショート", sigs, -1)
    report(f"大陰線(≥{thresh}ATR)翌足 [SL3/TP10]", sigs, -1, asym=True, sl=3, tp=10)

# ══════════════════════════════════════════
# C) ボラティリティ局面フィルタ
# ══════════════════════════════════════════
print("\n### C. ATRパーセンタイル × 既存エッジ ###")
print("    高ボラ(ATR上位25%) vs 低ボラ でエッジ強度が変わるか")

# rebuild flagship
sweep_base=[i for i in range(2,n)
            if len(shi[shi<i-1]) and h[i]>h[shi[shi<i-1][-1]]
            and c[i]<h[shi[shi<i-1][-1]] and reg[i] in ('UP','DOWN')]

for pct_thr, label in [(0.75,'高ボラ(ATR≥75%ile)'),(0.50,'中ボラ(ATR≥50%ile)'),(0.25,'低ボラ(ATR<25%ile)')]:
    if pct_thr==0.25:
        sigs=[i for i in sweep_base if not np.isnan(atr_pct[i]) and atr_pct[i]<0.25]
    else:
        sigs=[i for i in sweep_base if not np.isnan(atr_pct[i]) and atr_pct[i]>=pct_thr]
    report(f"旗艦スイープ × {label} ショート", sigs, -1)

# hour10 × high vol
h10_highvol=[i for i in range(n)
             if hr[i]==10 and reg[i] in ('UP','DOWN')
             and not np.isnan(atr_pct[i]) and atr_pct[i]>=0.75]
report("hour10 × 高ボラ(≥75%ile)", h10_highvol, -1)
h10_asym_hv=[i for i in range(n)
             if hr[i]==10 and reg[i] in ('UP','DOWN')
             and not np.isnan(atr_pct[i]) and atr_pct[i]>=0.75]
report("hour10 × 高ボラ [SL3/TP10]", h10_asym_hv, -1, asym=True, sl=3, tp=10)

# ══════════════════════════════════════════
# D) スクイーズ → ブレイク
# ══════════════════════════════════════════
print("\n### D. スクイーズ→ブレイク (ATR収縮後の拡大) ###")
print("    直近10本の平均ATRが過去50本の下位30% → 次の大足は追う")

squeeze_short=[]; squeeze_long=[]
for i in range(60,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    recent_atr=np.nanmean(at[max(0,i-10):i])
    hist_atr=at[max(0,i-50):i]; hist_atr=hist_atr[~np.isnan(hist_atr)]
    if len(hist_atr)<20: continue
    pct=np.sum(hist_atr<=recent_atr)/len(hist_atr)
    if pct>0.30: continue  # not squeezed
    # squeeze detected: now wait for breakout bar
    body=abs(c[i]-o[i])
    if body<0.5*at[i]: continue  # not a real breakout bar
    if c[i]<o[i] and reg[i] in ('UP','DOWN'):  # bearish breakout
        squeeze_short.append(i)
    elif c[i]>o[i] and reg[i]=='UP':
        squeeze_long.append(i)

report("スクイーズ後 大陰線ブレイク → ショート継続", squeeze_short, -1)
report("スクイーズ後 大陰線ブレイク [SL3/TP10]",    squeeze_short, -1, asym=True, sl=3, tp=10)
report("スクイーズ後 大陽線ブレイク → ロング継続",   squeeze_long,  +1)

# ══════════════════════════════════════════
# E) 連続同色足
# ══════════════════════════════════════════
print("\n### E. 連続同色足パターン ###")

for k in [2,3,4]:
    # k本連続陰線 → 継続ショート
    cont_bear=[i for i in range(k,n)
               if all(c[i-j]<o[i-j] for j in range(k))
               and reg[i] in ('UP','DOWN')]
    report(f"{k}本連続陰線 → 継続ショート [1:1]", cont_bear, -1)
    report(f"{k}本連続陰線 → 継続 [SL3/TP10]",   cont_bear, -1, asym=True, sl=3, tp=10)
    # k本連続陰線 → 逆張りロング (exhaustion)
    fade_bear=[i for i in cont_bear if reg[i]=='DOWN']  # DOWN中の連続陰線fad
    report(f"{k}本連続陰線 → 逆張りロング(DOWN)", fade_bear, +1, min_n=5)

# ══════════════════════════════════════════
# F) 前週高値/安値スイープ
# ══════════════════════════════════════════
print("\n### F. 前週高値/安値スイープ ###")

# weekly high/low
week_num=h1['Date'].dt.isocalendar().week.values
year_num=h1['Date'].dt.year.values
week_key=np.array([f"{y}W{w:02d}" for y,w in zip(year_num,week_num)])
unique_weeks=sorted(set(week_key))
week_hl={}
for wk in unique_weeks:
    mask=np.where(week_key==wk)[0]
    week_hl[wk]=(h[mask].max(), l[mask].min())

pwh=np.full(n,np.nan); pwl=np.full(n,np.nan)
for i in range(n):
    wk=week_key[i]
    idx=unique_weeks.index(wk)
    if idx>0:
        prev=unique_weeks[idx-1]
        if prev in week_hl:
            pwh[i]=week_hl[prev][0]
            pwl[i]=week_hl[prev][1]

pwh_sweep=[i for i in range(1,n)
           if not np.isnan(pwh[i]) and not np.isnan(at[i]) and at[i]>0
           and h[i]>pwh[i] and c[i]<pwh[i] and reg[i] in ('UP','DOWN')]
pwl_sweep=[i for i in range(1,n)
           if not np.isnan(pwl[i]) and not np.isnan(at[i]) and at[i]>0
           and l[i]<pwl[i] and c[i]>pwl[i] and reg[i] in ('UP','DOWN')]

report("前週高値スイープ → ショート [1:1]",    pwh_sweep, -1)
report("前週高値スイープ → ショート [SL3/TP10]",pwh_sweep, -1, asym=True, sl=3, tp=10)
report("前週安値スイープ → ロング",             pwl_sweep, +1)

# + vol
pwh_vol=[i for i in pwh_sweep if not np.isnan(vol_z[i]) and vol_z[i]>1.0]
report("前週高値スイープ + vol z>1 → ショート",pwh_vol, -1, min_n=5)

# ══════════════════════════════════════════
# G) 週間ギャップ (日曜オープン vs 金曜クローズ)
# ══════════════════════════════════════════
print("\n### G. 週間ギャップフェード ###")

gap_short=[]; gap_long=[]
for i in range(1,n):
    if dow[i]!=0 or hr[i]!=0: continue  # Monday 00:00 only
    if np.isnan(at[i]) or at[i]<=0: continue
    # find last Friday close
    j=i-1
    while j>=0 and not (dow[j]==4): j-=1
    if j<0: continue
    gap=o[i]-c[j]   # positive = gap up
    if abs(gap)<0.3*at[i]: continue  # no meaningful gap
    if gap>0 and reg[i] in ('UP','DOWN'): gap_short.append(i)  # gap up → fade short
    if gap<0 and reg[i]=='UP': gap_long.append(i)              # gap down → fade long

report("月曜ギャップアップ → フェードショート", gap_short, -1)
report("月曜ギャップダウン → フェードロング",   gap_long,  +1)

# ══════════════════════════════════════════
# H) FVG (3本チャートギャップ) → 埋め後の継続
# ══════════════════════════════════════════
print("\n### H. FVG(フェアバリューギャップ) → 埋め後継続ショート ###")
print("    チャートで多数確認。3本ギャップ>0.5pt → 価格が埋めに来たら再度ショート")

fvg_bear=[]   # bearish FVG: gap between bar i-2 low and bar i high (all 3 bars bearish direction)
for i in range(2,n):
    # bearish FVG: h[i] < l[i-2] (imbalance)
    gap_size=l[i-2]-h[i]
    if gap_size<0.5: continue
    if c[i-1]>o[i-1]: continue   # middle bar must be bearish
    fvg_bear.append((i, h[i], l[i-2]))   # FVG between h[i] and l[i-2]

# now find when price returns to fill the FVG (touches h[i] to l[i-2] zone)
fvg_retest=[]
for src, fvg_lo, fvg_hi in fvg_bear:
    for j in range(src+1, min(n, src+48)):
        if np.isnan(at[j]) or at[j]<=0: continue
        # price entered the FVG zone
        if h[j]>=fvg_lo and l[j]<=fvg_hi and reg[j] in ('UP','DOWN'):
            fvg_retest.append(j)
            break

report("FVG bearish 埋め後 継続ショート [1:1]",    fvg_retest, -1)
report("FVG bearish 埋め後 継続 [SL3/TP10]",       fvg_retest, -1, asym=True, sl=3, tp=10)
# with vol
fvg_vol=[i for i in fvg_retest if not np.isnan(vol_z[i]) and vol_z[i]>0.5]
report("FVG retest + vol z>0.5 [SL3/TP10]",        fvg_vol, -1, asym=True, sl=3, tp=10)

# ══════════════════════════════════════════
# I) 既存エッジ × 曜日フィルタ (月〜木 vs 金)
# ══════════════════════════════════════════
print("\n### I. 既存エッジ × 曜日フィルタ ###")

for dname, dval in [('月〜木(エネルギーあり)',list(range(4))),('金曜(値動き鈍化)',  [4])]:
    s=[i for i in sweep_base if dow[i] in dval]
    report(f"旗艦スイープ × {dname}", s, -1, min_n=5)
    h10=[i for i in range(n) if hr[i]==10 and reg[i] in ('UP','DOWN') and dow[i] in dval]
    report(f"hour10 × {dname}", h10, -1, min_n=5)

# ══════════════════════════════════════════
# J) 大足後の半値戻し (SL2/TP8 精密版)
# ══════════════════════════════════════════
print("\n### J. 大陰線後の半値戻しショート — RR詳細スキャン ###")
print("    5pt TP設定の最適RRを探す")

disp_pullback=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    if reg[i] not in ('UP','DOWN'): continue
    for j in range(i-1,max(0,i-5),-1):
        if np.isnan(at[j]) or at[j]<=0: continue
        if (o[j]-c[j])>=2.0*at[j]:   # big bear
            retrace=(c[i]-c[j])/(o[j]-c[j])
            if 0.3<=retrace<=0.7 and c[i]<o[j]:  # 30-70% retracement
                disp_pullback.append(i)
            break

for sl_,tp_ in [(2,5),(2,8),(2,10),(3,8),(3,10),(3,15),(2,6)]:
    report(f"大陰線半値戻し SL{sl_}/TP{tp_}", disp_pullback, -1, asym=True, sl=sl_, tp=tp_)

print("\n" + "="*100)
print("# 凡例: 🔥=ベータ超え  ✅=前後半両方プラス  SL3/TP10でEV>0=5pt狙い可能")
print("="*100)
