"""
TRENDLINE & HORIZONTAL S/R — comprehensive statistical test
=============================================================
Auto-detect:
  A) Downtrend line: connect last 2 confirmed swing highs (extrapolate forward)
  B) Uptrend line  : connect last 2 confirmed swing lows
  C) Horizontal S/R: all prior swing highs (resistance) and swing lows (support)

Events at each bar:
  1. TOUCH+REJECT  → fade trade (short at resistance, long at support)
  2. TOUCH+BREAK   → breakout trade (momentum, enter on break direction)

Combinations tested:
  - Regime filter (RANGE excluded)
  - MA200 side alignment
  - Volume confirmation (z-score)
  - Hour of day (killzone)
  - Multiple touches (tested-level is stronger)

Benchmark: random short in sloping regime = EV+1.25
1:1 ATR, horizon 24 bars (H1), SPREAD=0.3
No look-ahead.
"""
import pandas as pd, numpy as np, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3
BETA_EV=1.25

def load(p):
    df=pd.read_csv(p)
    df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
    return df

def calc_atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a

def calc_regime(c, slope_k=50):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

def swings(h,l,n,w=1):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

# ──────────────────────────────────────────
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values
c=h1['c'].values; v=h1['v'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values
reg=calc_regime(c)
hr=h1['Date'].dt.hour.values
mon=h1['Date'].dt.strftime('%Y-%m').values
vol_ma=pd.Series(v).rolling(20).mean().values
vol_sd=pd.Series(v).rolling(20).std().values
vol_z=(v-vol_ma)/np.where(vol_sd>0,vol_sd,np.nan)

sh_arr,sl_arr=swings(h,l,n,w=1)
shi=np.where(sh_arr)[0]; sli=np.where(sl_arr)[0]

# ──────────────────────────────────────────
# TRADE ENGINE
# ──────────────────────────────────────────
def trade_short(i, horizon=24):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]; tp=entry-rng; slv=entry+rng
    for k in range(i+1,min(n,i+1+horizon)):
        if h[k]>=slv: return -rng-SPREAD
        if l[k]<=tp:  return  rng-SPREAD
    return None

def trade_long(i, horizon=24):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]; tp=entry+rng; slv=entry-rng
    for k in range(i+1,min(n,i+1+horizon)):
        if l[k]<=slv: return -rng-SPREAD
        if h[k]>=tp:  return  rng-SPREAD
    return None

def report(label, pnls_idxs, min_n=10):
    items=[(p,i) for p,i in pnls_idxs if p is not None]
    if len(items)<min_n:
        print(f"  {label:<60} n={len(items)} (too few)"); return None
    pnls=[p for p,_ in items]; idxs=[i for _,i in items]
    wr=100*np.mean([p>0 for p in pnls]); ev=np.mean(pnls); tot=sum(pnls)
    mcl=cur=0
    for p in pnls:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    months=len(set(mon[i] for i in idxs)); tpm=len(pnls)/max(months,1)
    mid=n//2
    h1p=[p for p,i in items if i<mid]; h2p=[p for p,i in items if i>=mid]
    def hw(x): return f"{100*np.mean([p>0 for p in x]):.0f}%/{len(x)}" if len(x)>=8 else f"-/{len(x)}"
    stable=""
    if len(h1p)>=8 and len(h2p)>=8:
        stable="✅STABLE" if np.mean(h1p)>0 and np.mean(h2p)>0 else "❌unstable"
    beat="🔥BEAT_BETA" if ev>BETA_EV else ""
    print(f"  {label:<60} WR={wr:4.0f}% EV={ev:+5.2f} n={len(pnls):4d} MCL={mcl} ~{tpm:.0f}/mo  {hw(h1p)},{hw(h2p)}  {stable}{beat}")
    return items

# ══════════════════════════════════════════════════════════════════════
# A) HORIZONTAL S/R  (swing high = resistance, swing low = support)
# ══════════════════════════════════════════════════════════════════════
print("="*100)
print("# TRENDLINE & HORIZONTAL S/R  (H1, 1:1 ATR, horizon 24)")
print("="*100)
print("\n### A. HORIZONTAL LINES — swing high (resistance) & swing low (support) ###")
print("    Logic: prior swing high level → future price touches it → reject or break")

TOL=0.20   # fraction of ATR = tolerance zone width

# For each bar, find the most recent swing high above current price (active resistance)
# and the most recent swing low below (active support)
# "Touch" = bar high enters tolerance zone (for resistance) or bar low enters (for support)
# "Reject" = touch but close stays on same side → fade
# "Break"  = close pierces through → momentum

horiz_reject_sh=[]   # swing-high resistance reject → short
horiz_break_sh =[]   # swing-high resistance break  → long
horiz_reject_sl=[]   # swing-low  support   reject → long
horiz_break_sl =[]   # swing-low  support   break  → short

# track touch count per level (for "tested-level" filter)
touch_count_sh={}; touch_count_sl={}

for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=TOL*at[i]

    # --- Resistance (swing highs above current price) ---
    candidates_sh=shi[shi<i-1]   # confirmed swing highs before current bar
    if len(candidates_sh):
        # nearest resistance above close
        res_levels=h[candidates_sh[candidates_sh>0]]  # prices
        res_levels=res_levels[res_levels>c[i-1]]      # above prior close
        if len(res_levels):
            level=res_levels.min()   # closest resistance
            key=round(level,1)
            # touch: high enters zone
            if abs(h[i]-level)<=tol or (h[i]>level-tol and h[i]<level+tol):
                cnt=touch_count_sh.get(key,0)
                touch_count_sh[key]=cnt+1
                if c[i]<level:   # rejected → short
                    horiz_reject_sh.append((i, level, cnt))
                else:            # broke through → long
                    horiz_break_sh.append((i, level, cnt))

    # --- Support (swing lows below current price) ---
    candidates_sl=sli[sli<i-1]
    if len(candidates_sl):
        sup_levels=l[candidates_sl]
        sup_levels=sup_levels[sup_levels<c[i-1]]
        if len(sup_levels):
            level=sup_levels.max()
            key=round(level,1)
            if abs(l[i]-level)<=tol or (l[i]<level+tol and l[i]>level-tol):
                cnt=touch_count_sl.get(key,0)
                touch_count_sl[key]=cnt+1
                if c[i]>level:
                    horiz_reject_sl.append((i, level, cnt))
                else:
                    horiz_break_sl.append((i, level, cnt))

# basic
r_sh=report("SwingHigh resistance REJECT → short (all)",
    [(trade_short(i), i) for i,_,_ in horiz_reject_sh])
r_bsh=report("SwingHigh resistance BREAK  → long (all)",
    [(trade_long(i), i)  for i,_,_ in horiz_break_sh])
r_sl=report("SwingLow  support    REJECT → long (all)",
    [(trade_long(i), i)  for i,_,_ in horiz_reject_sl])
r_bsl=report("SwingLow  support    BREAK  → short (all)",
    [(trade_short(i), i) for i,_,_ in horiz_break_sl])

# + regime
def filt(lst, reg_ok):
    return [(i,lv,c2) for i,lv,c2 in lst if reg[i] in reg_ok]

report("SwingHigh resist REJECT + SLOPING → short",
    [(trade_short(i),i) for i,_,_ in filt(horiz_reject_sh,('UP','DOWN'))])
report("SwingHigh resist REJECT + DOWN    → short",
    [(trade_short(i),i) for i,_,_ in filt(horiz_reject_sh,('DOWN',))])
report("SwingHigh resist BREAK  + UP      → long",
    [(trade_long(i),i)  for i,_,_ in filt(horiz_break_sh,('UP',))])
report("SwingLow  supprt REJECT + SLOPING → long",
    [(trade_long(i),i)  for i,_,_ in filt(horiz_reject_sl,('UP','DOWN'))])
report("SwingLow  supprt BREAK  + DOWN    → short",
    [(trade_short(i),i) for i,_,_ in filt(horiz_break_sl,('DOWN',))])

# + volume at touch
def filtv(lst, z_min):
    return [(i,lv,c2) for i,lv,c2 in lst
            if not np.isnan(vol_z[i]) and vol_z[i]>z_min]

report("SwingHigh resist REJECT + vol z>1 + SLOPING → short",
    [(trade_short(i),i) for i,lv,c2 in
     filt(filtv(horiz_reject_sh,1.0),('UP','DOWN'))])
report("SwingHigh resist BREAK  + vol z>1 + UP → long",
    [(trade_long(i),i)  for i,lv,c2 in
     filt(filtv(horiz_break_sh,1.0),('UP',))])

# + tested (touched ≥2 times = stronger level)
def filt_tested(lst, min_cnt=1):
    return [(i,lv,c2) for i,lv,c2 in lst if c2>=min_cnt]

report("SwingHigh resist REJECT + tested≥2x + SLOPING → short",
    [(trade_short(i),i) for i,lv,c2 in
     filt(filt_tested(horiz_reject_sh,1),('UP','DOWN'))])
report("SwingHigh resist BREAK  + tested≥2x + UP → long",
    [(trade_long(i),i)  for i,lv,c2 in
     filt(filt_tested(horiz_break_sh,1),('UP',))])

# + MA50 premium (price above MA50 when rejecting from resistance)
def filt_ma50_above(lst):
    return [(i,lv,c2) for i,lv,c2 in lst
            if not np.isnan(ma50[i]) and c[i]>ma50[i]]

report("SwingHigh resist REJECT + MA50上 + SLOPING → short",
    [(trade_short(i),i) for i,lv,c2 in
     filt(filt_ma50_above(horiz_reject_sh),('UP','DOWN'))])

# killzone hours
def filt_kz(lst):
    return [(i,lv,c2) for i,lv,c2 in lst if hr[i] in (7,8,9,10,13,14,15,16)]

report("SwingHigh resist REJECT + killzone + SLOPING → short",
    [(trade_short(i),i) for i,lv,c2 in
     filt(filt_kz(horiz_reject_sh),('UP','DOWN'))])

# ══════════════════════════════════════════════════════════════════════
# B) TRENDLINES  (dynamic, causal)
# ══════════════════════════════════════════════════════════════════════
print("\n### B. TRENDLINES — dynamic slope lines ###")
print("    Downtrend line: last 2 swing highs. Uptrend line: last 2 swing lows.")
print("    Touch+Reject=fade, Touch+Break=momentum. Min 5-bar gap between swing points.")

tl_dn_reject=[]; tl_dn_break=[]   # downtrend line events
tl_up_reject=[]; tl_up_break=[]   # uptrend  line events

for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=TOL*at[i]

    # ── Downtrend line: connect last 2 swing highs ──
    past_sh=shi[shi<i-2]
    if len(past_sh)>=2:
        j1=past_sh[-2]; j2=past_sh[-1]
        if j2-j1>=5 and h[j2]<h[j1]:   # must be lower high (downtrend)
            slope=(h[j2]-h[j1])/(j2-j1)
            projected=h[j1]+slope*(i-j1)  # trendline value at bar i
            # touch from below (price approaches from below)
            if abs(h[i]-projected)<=tol:
                if c[i]<projected:   # rejected off downtrend line → short
                    tl_dn_reject.append((i,projected,slope))
                elif c[i]>projected: # broke above → bullish break → long
                    tl_dn_break.append((i,projected,slope))

    # ── Uptrend line: connect last 2 swing lows ──
    past_sl=sli[sli<i-2]
    if len(past_sl)>=2:
        j1=past_sl[-2]; j2=past_sl[-1]
        if j2-j1>=5 and l[j2]>l[j1]:   # higher lows (uptrend)
            slope=(l[j2]-l[j1])/(j2-j1)
            projected=l[j1]+slope*(i-j1)
            if abs(l[i]-projected)<=tol:
                if c[i]>projected:   # rejected off uptrend line → long
                    tl_up_reject.append((i,projected,slope))
                elif c[i]<projected: # broke below → bearish break → short
                    tl_up_break.append((i,projected,slope))

report("Downtrend line REJECT → short (all)",
    [(trade_short(i),i) for i,_,_ in tl_dn_reject])
report("Downtrend line BREAK  → long  (all)",
    [(trade_long(i),i)  for i,_,_ in tl_dn_break])
report("Uptrend line   REJECT → long  (all)",
    [(trade_long(i),i)  for i,_,_ in tl_up_reject])
report("Uptrend line   BREAK  → short (all)",
    [(trade_short(i),i) for i,_,_ in tl_up_break])

# + regime
def filtR(lst,ok):
    return [(i,p,s) for i,p,s in lst if reg[i] in ok]

report("Downtrend line REJECT + DOWN → short",
    [(trade_short(i),i) for i,_,_ in filtR(tl_dn_reject,('DOWN',))])
report("Downtrend line REJECT + SLOPING → short",
    [(trade_short(i),i) for i,_,_ in filtR(tl_dn_reject,('UP','DOWN'))])
report("Downtrend line BREAK  + UP   → long",
    [(trade_long(i),i)  for i,_,_ in filtR(tl_dn_break,('UP',))])
report("Uptrend line   BREAK  + DOWN → short",
    [(trade_short(i),i) for i,_,_ in filtR(tl_up_break,('DOWN',))])
report("Uptrend line   REJECT + UP   → long",
    [(trade_long(i),i)  for i,_,_ in filtR(tl_up_reject,('UP',))])

# + volume at trendline touch
def filtVR(lst,z,ok):
    return [(i,p,s) for i,p,s in lst if reg[i] in ok
            and not np.isnan(vol_z[i]) and vol_z[i]>z]

report("Downtrend line REJECT + vol z>1 + SLOPING → short",
    [(trade_short(i),i) for i,_,_ in filtVR(tl_dn_reject,1.0,('UP','DOWN'))])
report("Downtrend line BREAK  + vol z>1 + UP → long",
    [(trade_long(i),i)  for i,_,_ in filtVR(tl_dn_break,1.0,('UP',))])
report("Uptrend line   BREAK  + vol z>1 + DOWN → short",
    [(trade_short(i),i) for i,_,_ in filtVR(tl_up_break,1.0,('DOWN',))])

# + MA200 alignment
def filtMA(lst, side):   # side='above' or 'below'
    return [(i,p,s) for i,p,s in lst
            if not np.isnan(ma200[i])
            and (c[i]>ma200[i] if side=='above' else c[i]<ma200[i])]

report("Downtrend line REJECT + MA200下 → short",
    [(trade_short(i),i) for i,_,_ in filtMA(tl_dn_reject,'below')])
report("Downtrend line BREAK  + MA200上 → long",
    [(trade_long(i),i)  for i,_,_ in filtMA(tl_dn_break,'above')])
report("Uptrend line   BREAK  + MA200下 → short",
    [(trade_short(i),i) for i,_,_ in filtMA(tl_up_break,'below')])

# ── stacked best combos ──
print("\n### C. STACKED COMBOS (trendline × horiz × MA × vol) ###")

# (1) Horiz resist reject + MA50上 + vol + killzone → short
combo1=[(i,lv,c2) for i,lv,c2 in horiz_reject_sh
        if reg[i] in ('UP','DOWN')
        and not np.isnan(ma50[i]) and c[i]>ma50[i]
        and hr[i] in (7,8,9,10,13,14,15,16)]
report("Horiz-resist-reject + MA50上 + killzone → short",
    [(trade_short(i),i) for i,_,_ in combo1])

# (2) Trendline dn reject + MA200下 + vol z>0.5 → short
combo2=[(i,p,s) for i,p,s in tl_dn_reject
        if not np.isnan(ma200[i]) and c[i]<ma200[i]
        and not np.isnan(vol_z[i]) and vol_z[i]>0.5]
report("TL-dn-reject + MA200下 + vol z>0.5 → short",
    [(trade_short(i),i) for i,_,_ in combo2])

# (3) Trendline dn BREAK + MA200上 + vol z>0.5 → long (breakout momentum)
combo3=[(i,p,s) for i,p,s in tl_dn_break
        if not np.isnan(ma200[i]) and c[i]>ma200[i]
        and not np.isnan(vol_z[i]) and vol_z[i]>0.5]
report("TL-dn-break + MA200上 + vol z>0.5 → long (breakout)",
    [(trade_long(i),i) for i,_,_ in combo3])

# (4) Horiz break + vol z>1 + regime momentum
combo4=[(i,lv,c2) for i,lv,c2 in horiz_break_sh
        if not np.isnan(vol_z[i]) and vol_z[i]>1.0
        and reg[i]=='UP']
report("Horiz-resist-BREAK + vol z>1 + UP → long (breakout momentum)",
    [(trade_long(i),i) for i,_,_ in combo4])

combo5=[(i,lv,c2) for i,lv,c2 in horiz_break_sl
        if not np.isnan(vol_z[i]) and vol_z[i]>1.0
        and reg[i]=='DOWN']
report("Horiz-support-BREAK + vol z>1 + DOWN → short (breakdown)",
    [(trade_short(i),i) for i,_,_ in combo5])

# ══════════════════════════════════════════════════════════════════════
# CHART: visualise trendlines + horiz S/R + trade events on H1
# ══════════════════════════════════════════════════════════════════════
print("\n[Generating chart...]")

# pick a 600-bar window (roughly 3 months visible)
WIN=600; start=max(0,n-WIN)
idx=np.arange(start,n)
dates=h1['Date'].values[start:n]

fig,ax=plt.subplots(figsize=(26,13))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
ax.tick_params(colors='#aaa',labelsize=9)
for sp in ax.spines.values(): sp.set_color('#333')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x)}"))

# candlesticks
for i,x in enumerate(idx):
    col='#26a69a' if c[x]>=o[x] else '#ef5350'
    ax.plot([i,i],[l[x],h[x]],color=col,lw=0.6,alpha=0.8)
    ax.add_patch(plt.Rectangle((i-0.3,min(o[x],c[x])),0.6,abs(c[x]-o[x]),
                 color=col,alpha=0.9,zorder=2))

# MA200 and MA50
ax.plot(range(len(idx)),ma200[start:n],color='#ff9800',lw=1.2,alpha=0.8,label='MA200')
ax.plot(range(len(idx)),ma50[start:n], color='#90caf9',lw=0.9,alpha=0.7,label='MA50')

# Horizontal swing-high levels (resistance, last 30 swing highs in window)
sh_in_win=shi[(shi>=start)&(shi<n)]
for j in sh_in_win[-30:]:
    lv=h[j]; xj=j-start
    # draw dashed horizontal line from that swing to end of chart
    ax.axhline(lv,xmin=xj/len(idx),color='#ef5350',lw=0.6,ls='--',alpha=0.4)

# Horizontal swing-low levels (support)
sl_in_win=sli[(sli>=start)&(sli<n)]
for j in sl_in_win[-30:]:
    lv=l[j]; xj=j-start
    ax.axhline(lv,xmin=xj/len(idx),color='#26a69a',lw=0.6,ls='--',alpha=0.4)

# Downtrend trendlines (last 8 events in window)
tl_dn_drawn=set()
for i,proj,slope in tl_dn_reject[-20:]+tl_dn_break[-20:]:
    if i<start: continue
    # reconstruct line: need j1,j2
    past_sh2=shi[shi<i-2];
    if len(past_sh2)<2: continue
    j1=past_sh2[-2]; j2=past_sh2[-1]
    key=(j1,j2)
    if key in tl_dn_drawn: continue
    tl_dn_drawn.add(key)
    if j2-j1<5 or h[j2]>=h[j1]: continue
    sp=(h[j2]-h[j1])/(j2-j1)
    x_st=max(start,j1); x_en=min(n-1,i+30)
    xs=np.arange(x_st,x_en+1)-start
    ys=h[j1]+sp*(np.arange(x_st,x_en+1)-j1)
    ax.plot(xs,ys,color='#ff6b6b',lw=1.2,ls='-',alpha=0.7)

# Uptrend trendlines
tl_up_drawn=set()
for i,proj,slope in tl_up_reject[-20:]+tl_up_break[-20:]:
    if i<start: continue
    past_sl2=sli[sli<i-2]
    if len(past_sl2)<2: continue
    j1=past_sl2[-2]; j2=past_sl2[-1]
    key=(j1,j2)
    if key in tl_up_drawn: continue
    tl_up_drawn.add(key)
    if j2-j1<5 or l[j2]<=l[j1]: continue
    sp=(l[j2]-l[j1])/(j2-j1)
    x_st=max(start,j1); x_en=min(n-1,i+30)
    xs=np.arange(x_st,x_en+1)-start
    ys=l[j1]+sp*(np.arange(x_st,x_en+1)-j1)
    ax.plot(xs,ys,color='#69f0ae',lw=1.2,ls='-',alpha=0.7)

# Mark REJECT events (arrows)
for i,lv,_ in horiz_reject_sh:
    if i<start: continue
    ax.annotate('',xy=(i-start,h[i]+at[i]*0.2),
                xytext=(i-start,h[i]+at[i]*0.7),
                arrowprops=dict(arrowstyle='->',color='#ff6b6b',lw=1.2))

for i,lv,_ in horiz_reject_sl:
    if i<start: continue
    ax.annotate('',xy=(i-start,l[i]-at[i]*0.2),
                xytext=(i-start,l[i]-at[i]*0.7),
                arrowprops=dict(arrowstyle='->',color='#69f0ae',lw=1.2))

# Mark BREAK events (triangles)
for i,lv,_ in horiz_break_sh:
    if i<start: continue
    ax.plot(i-start, h[i]+at[i]*0.3, '^', color='#90caf9', ms=5, alpha=0.8)

for i,lv,_ in horiz_break_sl:
    if i<start: continue
    ax.plot(i-start, l[i]-at[i]*0.3, 'v', color='#ffb74d', ms=5, alpha=0.8)

# x-axis labels (monthly)
xt=[]; xl=[]
for j,x in enumerate(idx):
    if h1['Date'].iloc[x].day==1 and h1['Date'].iloc[x].hour==0:
        xt.append(j); xl.append(h1['Date'].iloc[x].strftime('%b %Y'))
ax.set_xticks(xt[::max(1,len(xt)//10)]); ax.set_xticklabels(xl[::max(1,len(xl)//10)],fontsize=8,color='#aaa')

patches=[mpatches.Patch(color='#ff6b6b',label='Downtrend TL / Resist (↓reject)'),
         mpatches.Patch(color='#69f0ae',label='Uptrend TL / Support  (↑reject)'),
         mpatches.Patch(color='#90caf9',label='Resist BREAK ▲ (momentum long)'),
         mpatches.Patch(color='#ffb74d',label='Support BREAK ▼ (breakdown short)'),
         mpatches.Patch(color='#ff9800',label='MA200'),
         mpatches.Patch(color='#90caf9',alpha=0.5,label='MA50')]
ax.legend(handles=patches,loc='upper left',fontsize=8,facecolor='#1a1a2e',labelcolor='white')
ax.set_title('H1 XAUUSD — Trendlines + Horizontal S/R  (dashed=horiz, solid=TL, ↓↑=reject, ▲▼=break)',
             color='white',fontsize=11,pad=10)
plt.tight_layout()
plt.savefig('trendline_chart.png',dpi=130,bbox_inches='tight',facecolor='#0d1117')
print("Chart saved: trendline_chart.png")
plt.close()

print(f"\n{'='*100}")
print(f"# BETA benchmark: EV+{BETA_EV:.2f}  |  🔥=beats beta  ✅=both-half stable")
print(f"{'='*100}")
