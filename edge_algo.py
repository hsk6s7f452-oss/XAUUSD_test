"""
機関・アルゴ検知エッジ
1. Judas Swing — ロンドンオープン最初の動きに逆らう
2. Power of 3 — セッション内操作検知
3. ICT Killzone × 既存エッジ
4. 週初/月初水準からの乖離
5. ストップカスケード後反転
6. ロンドンフィックス(16:00UTC)前後
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

df=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
df=df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
n=len(df)
o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
hour=df['Date'].dt.hour.values
dow =df['Date'].dt.dayofweek.values
day =df['Date'].dt.date.values
week=df['Date'].dt.isocalendar().week.values
month=df['Date'].dt.month.values

tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
at=np.full(n,np.nan); at[1:]=pd.Series(tr).rolling(14).mean().values
ma200=pd.Series(c).rolling(200).mean().values
slope_k=50
reg=np.array(['NA']*n,dtype=object)
for i in range(n):
    if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
    s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
    if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
    elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
    else: reg[i]='RANGE'

def trade(i, direction='short', rr=1.0, hold=24):
    if i+1>=n: return None
    e=o[i+1]; r=at[i]
    if np.isnan(r) or r<=0: return None
    tp=e-r*rr if direction=='short' else e+r*rr
    sl=e+r    if direction=='short' else e-r
    for k in range(i+1, min(n,i+1+hold)):
        if direction=='short':
            if h[k]>=sl: return -r-SPREAD
            if l[k]<=tp: return  r*rr-SPREAD
        else:
            if l[k]<=sl: return -r-SPREAD
            if h[k]>=tp: return  r*rr-SPREAD
    return None

def rep(name, pnls):
    if not pnls: print(f"  {name:65s} n=0"); return False
    n_=len(pnls); wr=100*sum(1 for p in pnls if p>0)/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    ok=e1>0 and e2>0
    flag='✅STABLE' if ok else '❌'
    if ok and ev>5 and n_>=10: flag+='🔥'
    if ok and ev>10 and n_>=15: flag+='🔥'
    print(f"  {name:65s} WR={wr:4.0f}% EV={ev:+6.2f} n={n_:4d}(~{n_/5:.0f}/月) [{e1:+.1f}/{e2:+.1f}] {flag}")
    return ok

# ── 日次・週次データ前処理 ──────────────────────────────
# 各日の07:00UTCの始値 (ロンドンオープン)
london_open={}   # date -> (bar_idx, open_price)
for i in range(n):
    if hour[i]==7:
        london_open[day[i]]=(i, o[i])

# 各日の09:00UTC (2時間後の状態)
london_2h={}
for i in range(n):
    if hour[i]==9:
        london_2h[day[i]]=(i, c[i])

# 週初(月曜07:00)の価格
week_open={}
for i in range(n):
    if dow[i]==0 and hour[i]==0:
        wk=f"{df['Date'].iloc[i].year}-{week[i]}"
        if wk not in week_open:
            week_open[wk]=(i, o[i])

# 月初(1日または最初の営業日)
month_open={}
for i in range(n):
    ym=f"{df['Date'].iloc[i].year}-{month[i]}"
    if ym not in month_open and hour[i]==0:
        month_open[ym]=(i, o[i])

print("="*95)
print("# 機関・アルゴ検知エッジ検証")
print("="*95)

# ═══════════════════════════════════════════════════════════
# 1. Judas Swing
# ロンドンオープン(07:00)から09:00までの方向が「罠」
# 上げた → 本当は下 → 10:00でショート
# 下げた → 本当は上 → 10:00でロング
# ═══════════════════════════════════════════════════════════
print("\n### 1. Judas Swing (ロンドンオープン最初の動き = 罠) ###")

js_short=[]; js_short_sl=[]; js_short_sl_big=[]
js_long=[]; js_long_sl=[]

for i in range(n):
    if hour[i]!=10: continue  # 10:00でエントリー
    if np.isnan(at[i]) or at[i]<=0: continue
    d=day[i]
    if d not in london_open or d not in london_2h: continue

    lo_bar, lo_price = london_open[d]
    l2_bar, l2_price = london_2h[d]
    move = l2_price - lo_price  # 正=上昇, 負=下落

    # Judas上昇(罠上げ)→ショート
    if move > at[lo_bar]*0.3:
        p=trade(i,'short')
        if p is not None:
            js_short.append(p)
            if reg[i] in ('UP','DOWN'): js_short_sl.append(p)
            # 大きな罠(ATR以上)
            if move > at[lo_bar]*0.8:
                js_short_sl_big.append(p) if reg[i] in ('UP','DOWN') else None

    # Judas下落(罠下げ)→ロング
    if move < -at[lo_bar]*0.3:
        p=trade(i,'long')
        if p is not None:
            js_long.append(p)
            if reg[i] in ('UP','DOWN'): js_long_sl.append(p)

rep("Judas上昇罠→10:00ショート", js_short)
rep("Judas上昇罠→10:00ショート + SLOPING", js_short_sl)
rep("Judas大幅上昇罠(>ATR)→10:00ショート + SLOPING", js_short_sl_big)
rep("Judas下落罠→10:00ロング", js_long)
rep("Judas下落罠→10:00ロング + SLOPING", js_long_sl)

# ═══════════════════════════════════════════════════════════
# 2. Power of 3 — セッションの「操作」後に本方向
# 07:00-12:00 の高値・安値 → 片方だけ突き抜けた後に逆へ
# (当日の07:00-10:00でH/Lを形成 → 片方スイープ → 逆方向本番)
# ═══════════════════════════════════════════════════════════
print("\n### 2. Power of 3 (セッション内操作検知) ###")

po3_short=[]; po3_short_sl=[]
po3_long =[]; po3_long_sl =[]

# 各日の12:00-17:00のエントリー検討
for i in range(n):
    if not (12<=hour[i]<=16): continue
    if np.isnan(at[i]) or at[i]<=0: continue
    d=day[i]

    # 同日の07-11時のH/L
    morning_bars=[j for j in range(max(0,i-12),i) if day[j]==d and 7<=hour[j]<=11]
    if len(morning_bars)<3: continue
    morn_h=max(h[j] for j in morning_bars)
    morn_l=min(l[j] for j in morning_bars)
    morn_range=morn_h-morn_l
    if morn_range<at[i]*0.5: continue

    # 現在価格が午前レンジを上抜け→ショート (罠上げから本売り)
    if c[i]>morn_h and c[i]<morn_h+at[i]*0.5:
        p=trade(i,'short')
        if p is not None:
            po3_short.append(p)
            if reg[i] in ('UP','DOWN'): po3_short_sl.append(p)

    # 現在価格が午前レンジを下抜け→ロング
    if c[i]<morn_l and c[i]>morn_l-at[i]*0.5:
        p=trade(i,'long')
        if p is not None:
            po3_long.append(p)
            if reg[i] in ('UP','DOWN'): po3_long_sl.append(p)

rep("Power of 3: 午前H上抜け→午後ショート", po3_short)
rep("Power of 3: 午前H上抜け→午後ショート + SLOPING", po3_short_sl)
rep("Power of 3: 午前L下抜け→午後ロング", po3_long)
rep("Power of 3: 午前L下抜け→午後ロング + SLOPING", po3_long_sl)

# ═══════════════════════════════════════════════════════════
# 3. 週初水準・月初水準からの乖離
# 週初(月曜始値)より大幅上 → 売り / 大幅下 → 買い
# ═══════════════════════════════════════════════════════════
print("\n### 3. 週初/月初水準 乖離エッジ ###")

wk_above=[]; wk_below=[]; wk_above_sl=[]; wk_below_sl=[]
mo_above=[]; mo_below=[]; mo_above_sl=[]; mo_below_sl=[]

for i in range(n):
    if np.isnan(at[i]) or at[i]<=0: continue
    wk=f"{df['Date'].iloc[i].year}-{week[i]}"
    ym=f"{df['Date'].iloc[i].year}-{month[i]}"

    # 週初水準
    if wk in week_open:
        wk_bar, wk_price = week_open[wk]
        if i<=wk_bar: continue
        diff=(c[i]-wk_price)/at[i]
        if diff>2.0:  # 週初比+2ATR以上
            p=trade(i,'short')
            if p is not None:
                wk_above.append(p)
                if reg[i] in ('UP','DOWN'): wk_above_sl.append(p)
        if diff<-2.0:
            p=trade(i,'long')
            if p is not None:
                wk_below.append(p)
                if reg[i] in ('UP','DOWN'): wk_below_sl.append(p)

    # 月初水準
    if ym in month_open:
        mo_bar, mo_price = month_open[ym]
        if i<=mo_bar: continue
        diff=(c[i]-mo_price)/at[i]
        if diff>3.0:
            p=trade(i,'short')
            if p is not None:
                mo_above.append(p)
                if reg[i] in ('UP','DOWN'): mo_above_sl.append(p)
        if diff<-3.0:
            p=trade(i,'long')
            if p is not None:
                mo_below.append(p)
                if reg[i] in ('UP','DOWN'): mo_below_sl.append(p)

rep("週初比+2ATR以上 → ショート", wk_above)
rep("週初比+2ATR以上 → ショート + SLOPING", wk_above_sl)
rep("週初比-2ATR以下 → ロング", wk_below)
rep("週初比-2ATR以下 → ロング + SLOPING", wk_below_sl)
rep("月初比+3ATR以上 → ショート", mo_above)
rep("月初比+3ATR以上 → ショート + SLOPING", mo_above_sl)
rep("月初比-3ATR以下 → ロング", mo_below)
rep("月初比-3ATR以下 → ロング + SLOPING", mo_below_sl)

# ═══════════════════════════════════════════════════════════
# 4. ストップカスケード後の反転
# 単一バーで3ATR以上の急騰・急落 → 次足から反転ショート/ロング
# (機関がストップを刈った後の急反転)
# ═══════════════════════════════════════════════════════════
print("\n### 4. ストップカスケード後の急反転 ###")

sc_short=[]; sc_short_sl=[]; sc_short_sl_day=[]
sc_long=[]; sc_long_sl=[]

for i in range(5, n-2):
    if np.isnan(at[i]) or at[i]<=0: continue
    candle_range=h[i]-l[i]
    candle_body=abs(c[i]-o[i])

    # 急騰カスケード(大陽線 body≥2ATR) → 翌足〜3足で反転ショート待ち
    if c[i]>o[i] and candle_body>=2*at[i]:
        for j in range(i+1, min(n-1,i+4)):
            if c[j]<o[j] and h[j]<h[i]:  # 陰線で前の急騰高値を超えない
                p=trade(j,'short')
                if p is not None:
                    sc_short.append(p)
                    if reg[j] in ('UP','DOWN'): sc_short_sl.append(p)
                    if reg[j] in ('UP','DOWN') and 7<=hour[j]<=19: sc_short_sl_day.append(p)
                break

    # 急落カスケード(大陰線 body≥2ATR) → 翌足〜3足で反転ロング待ち
    if c[i]<o[i] and candle_body>=2*at[i]:
        for j in range(i+1, min(n-1,i+4)):
            if c[j]>o[j] and l[j]>l[i]:
                p=trade(j,'long')
                if p is not None:
                    sc_long.append(p)
                    if reg[j] in ('UP','DOWN'): sc_long_sl.append(p)
                break

rep("急騰(≥2ATR陽線)後の反転ショート", sc_short)
rep("急騰後の反転ショート + SLOPING", sc_short_sl)
rep("急騰後の反転ショート + SLOPING + UTC07-19", sc_short_sl_day)
rep("急落(≥2ATR陰線)後の反転ロング", sc_long)
rep("急落後の反転ロング + SLOPING", sc_long_sl)

# ═══════════════════════════════════════════════════════════
# 5. ロンドンフィックス前後 (15:00-16:00UTC)
# フィックス前: ポジション調整で動く
# フィックス直後: 方向が決まる
# ═══════════════════════════════════════════════════════════
print("\n### 5. ロンドンフィックス (15-16UTC) ###")

fix_short=[]; fix_short_sl=[]
fix_long=[]; fix_long_sl=[]

for i in range(n-1):
    if hour[i]!=16: continue  # フィックス直後の17:00足
    if np.isnan(at[i]) or at[i]<=0: continue
    d=day[i]

    # 15:00(フィックス前1時間)の始値
    fix_pre=[j for j in range(max(0,i-3),i) if hour[j]==15 and day[j]==d]
    if not fix_pre: continue
    fix_open=o[fix_pre[0]]
    fix_move=c[i]-fix_open  # フィックス前後の動き

    # フィックス前後で上昇 → 17時以降ショート (ポジション解消)
    if fix_move>at[i]*0.5:
        p=trade(i,'short')
        if p is not None:
            fix_short.append(p)
            if reg[i] in ('UP','DOWN'): fix_short_sl.append(p)
    if fix_move<-at[i]*0.5:
        p=trade(i,'long')
        if p is not None:
            fix_long.append(p)
            if reg[i] in ('UP','DOWN'): fix_long_sl.append(p)

rep("フィックス上昇→17UTC以降ショート", fix_short)
rep("フィックス上昇→17UTC以降ショート + SLOPING", fix_short_sl)
rep("フィックス下落→17UTC以降ロング", fix_long)
rep("フィックス下落→17UTC以降ロング + SLOPING", fix_long_sl)

# ═══════════════════════════════════════════════════════════
# 6. 月曜ギャップ (週末ギャップ)
# 月曜始値が前週金曜終値から乖離 → ギャップ埋めを狙う
# ═══════════════════════════════════════════════════════════
print("\n### 6. 週明けギャップ戦略 ###")

gap_up=[]; gap_up_sl=[]
gap_down=[]; gap_down_sl=[]

fri_close={}
for i in range(n):
    if dow[i]==4:  # 金曜
        ym=f"{df['Date'].iloc[i].year}-{week[i]}"
        fri_close[ym]=(i,c[i])

for i in range(n):
    if dow[i]!=0 or hour[i]!=0: continue  # 月曜00:00
    if np.isnan(at[i]) or at[i]<=0: continue
    prev_wk=f"{df['Date'].iloc[i].year}-{week[i]-1}"
    if prev_wk not in fri_close: continue
    _,fri_c=fri_close[prev_wk]
    gap=(o[i]-fri_c)/at[i]

    if gap>0.5:  # ギャップアップ → 埋め期待でショート
        p=trade(i,'short')
        if p is not None:
            gap_up.append(p)
            if reg[i] in ('UP','DOWN'): gap_up_sl.append(p)
    if gap<-0.5:  # ギャップダウン → 埋め期待でロング
        p=trade(i,'long')
        if p is not None:
            gap_down.append(p)
            if reg[i] in ('UP','DOWN'): gap_down_sl.append(p)

rep("週明けギャップアップ→ショート(埋め)", gap_up)
rep("週明けギャップアップ→ショート + SLOPING", gap_up_sl)
rep("週明けギャップダウン→ロング(埋め)", gap_down)
rep("週明けギャップダウン→ロング + SLOPING", gap_down_sl)

print("\n" + "="*95)
