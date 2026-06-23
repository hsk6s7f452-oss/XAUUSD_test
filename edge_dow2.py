"""
ダウ理論 Part2 — H1 XAUUSD
1. 高安切り上げ/切り下げ確認後の押し目エントリー
2. 3波エントリー (1波→2波→3波の起点)
3. フィボリトレース (38.2/50/61.8%) からのエントリー
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

def calc_regime(c, slope_k=50):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n, dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

def swings(h,l,n,w=2):
    """w=2: 5本確定スイング (より明確な構造)"""
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if all(h[i]>=h[i-j] for j in range(1,w+1)) and all(h[i]>=h[i+j] for j in range(1,w+1)):
            sh[i]=True
        if all(l[i]<=l[i-j] for j in range(1,w+1)) and all(l[i]<=l[i+j] for j in range(1,w+1)):
            sl[i]=True
    return sh,sl

def eval_trade(o,h,l,c,at,entry_bar,direction,n,rr=1.0,hold=24):
    """direction: 'short' or 'long'"""
    if entry_bar+1>=n: return None,None
    entry=o[entry_bar+1]; rng=at[entry_bar]
    if np.isnan(rng) or rng<=0: return None,None
    if direction=='short':
        tp=entry-rng*rr; slv=entry+rng
    else:
        tp=entry+rng*rr; slv=entry-rng
    for k in range(entry_bar+1, min(n, entry_bar+1+hold)):
        if direction=='short':
            if h[k]>=slv: return False, -rng-SPREAD
            if l[k]<=tp:  return True,   rng*rr-SPREAD
        else:
            if l[k]<=slv: return False, -rng-SPREAD
            if h[k]>=tp:  return True,   rng*rr-SPREAD
    return None, None

def report(name, results, n_months=5):
    if not results: print(f"  {name}: n=0"); return
    wins=[r for r in results if r>0]; losses=[r for r in results if r<=0]
    wr=100*len(wins)/len(results); ev=np.mean(results)
    n=len(results); nmo=n/n_months
    # half-split
    h1r=results[:n//2]; h2r=results[n//2:]
    ev1=np.mean(h1r) if h1r else 0; ev2=np.mean(h2r) if h2r else 0
    stable = ev1>0 and ev2>0
    flag='✅STABLE' if stable else '❌unstable'
    print(f"  {name:60s} WR={wr:4.0f}% EV={ev:+6.2f} n={n:4d} ~{nmo:.0f}/mo  "
          f"H1={ev1:+.2f}/H2={ev2:+.2f}  {flag}")
    return stable, ev, n

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values
c=h1['c'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
reg=calc_regime(c)

# w=2 スイング (5本確定, より明確)
sh_arr,sl_arr=swings(h,l,n,w=2)
shi=np.where(sh_arr)[0]
sli=np.where(sl_arr)[0]

print("="*100)
print("# ダウ理論 Part2 — 高安切り上げ/3波/フィボ")
print("="*100)

# ──────────────────────────────────────────────────────────────
# 1. 高安切り上げ / 切り下げ 確認後の押し目エントリー
# ──────────────────────────────────────────────────────────────
print("\n### 1. 高安切り上げ/切り下げ後の押し目 ###")
print("    切り下げ確認(LH+LL)→次の戻りでショート / 切り上げ確認(HH+HL)→次の押し目でロング")

res_sh_pullback=[]  # 切り下げ後プルバックショート
res_lg_pullback=[]  # 切り上げ後プルバックロング
res_sh_pb_dow=[]    # 切り下げ + MA200下
res_lg_pb_dow=[]    # 切り上げ + MA200上

for i in range(10, n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if len(psh)<2 or len(psl)<2: continue

    sh1=h[psh[-2]]; sh2=h[psh[-1]]
    sl1=l[psl[-2]]; sl2=l[psl[-1]]

    # 切り下げ確認 (LH+LL)
    if sh2<sh1 and sl2<sl1:
        # 現在価格が直前スイングハイ(LH=sh2)付近に戻っている → ショート
        tol=0.25*at[i]
        if abs(c[i]-sh2)<=tol or (c[i]>sh2-tol and c[i]<sh2+tol):
            if c[i]<sh2:  # 戻ったが上抜けしていない
                win,pnl=eval_trade(o,h,l,c,at,i,'short',n)
                if win is not None:
                    res_sh_pullback.append(pnl)
                    if not np.isnan(ma200[i]) and c[i]<ma200[i]:
                        res_sh_pb_dow.append(pnl)

    # 切り上げ確認 (HH+HL)
    if sh2>sh1 and sl2>sl1:
        # 現在価格が直前スイングロー(HL=sl2)付近に押している → ロング
        tol=0.25*at[i]
        if abs(c[i]-sl2)<=tol or (c[i]>sl2-tol and c[i]<sl2+tol):
            if c[i]>sl2:
                win,pnl=eval_trade(o,h,l,c,at,i,'long',n)
                if win is not None:
                    res_lg_pullback.append(pnl)
                    if not np.isnan(ma200[i]) and c[i]>ma200[i]:
                        res_lg_pb_dow.append(pnl)

report("切り下げ(LH+LL)→LH戻りショート", res_sh_pullback)
report("切り下げ(LH+LL)→LH戻りショート + MA200下", res_sh_pb_dow)
report("切り上げ(HH+HL)→HL押しロング", res_lg_pullback)
report("切り上げ(HH+HL)→HL押しロング + MA200上", res_lg_pb_dow)

# ──────────────────────────────────────────────────────────────
# 2. 3波エントリー
# 1波 = スイングの最初の大きな動き
# 2波 = 38-62%フィボ戻り (浅すぎず深すぎず)
# 3波起点 = 2波終了確認後にエントリー
# ──────────────────────────────────────────────────────────────
print("\n### 2. 3波エントリー ###")
print("    1波(インパルス)→2波(フィボ戻り38-62%)→3波起点でエントリー")

res_wave3_short=[]
res_wave3_long=[]
res_wave3_s_slop=[]
res_wave3_l_slop=[]

for i in range(15, n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if len(psh)<2 or len(psl)<2: continue

    # === ショート: 下降3波 ===
    # 1波下降: 直近スイングハイ(wave_top) → その後のスイングロー(wave_bot) で定義
    # 2波戻り: wave_bot からwave_topへの38-62%戻り
    # 3波起点: 2波の高値(LH)から価格が再び下落し始めた確認後

    # 直近の明確な下降1波を探す
    # 方法: 最新SH → その後の最初のSL
    last_sh_idx=psh[-1]; last_sh_val=h[last_sh_idx]
    # そのSHの後のSL
    post_sl=psl[psl>last_sh_idx]
    if not len(post_sl): continue
    wave1_bot_idx=post_sl[0]; wave1_bot=l[wave1_bot_idx]

    if wave1_bot>=last_sh_val: continue  # 下降していない
    wave1_range=last_sh_val-wave1_bot
    if wave1_range<at[i]*0.5: continue  # 1波が小さすぎる

    # 2波の戻り: wave1_bot後の戻り高値(SH)が38-78%以内
    post_w1_sh=psh[psh>wave1_bot_idx]
    if not len(post_w1_sh): continue
    wave2_top_idx=post_w1_sh[-1]; wave2_top=h[wave2_top_idx]
    retrace=(wave2_top-wave1_bot)/wave1_range

    if not (0.30<=retrace<=0.80): continue  # フィボ範囲外

    # 3波起点: wave2_topが確定し、現在価格がwave2_topを下回って下落中
    if wave2_top_idx>=i: continue
    if c[i]>=wave2_top: continue  # まだwave2の高値を超えている
    if c[i]>=wave2_top-at[i]*0.3: continue  # wave2topから十分離れていない

    # 追加: 現在がwave2_topから直近のスイングロー(3波進行中の確認)
    # wave2_topが確定した後に初めてエントリー(i > wave2_top_idx + 2)
    if i <= wave2_top_idx + 2: continue

    win,pnl=eval_trade(o,h,l,c,at,i,'short',n)
    if win is not None:
        res_wave3_short.append(pnl)
        if reg[i] in ('UP','DOWN'):
            res_wave3_s_slop.append(pnl)

    # === ロング: 上昇3波 ===
    last_sl_idx=psl[-1]; last_sl_val=l[last_sl_idx]
    post_sh2=psh[psh>last_sl_idx]
    if not len(post_sh2): continue
    wave1_top_idx=post_sh2[0]; wave1_top=h[wave1_top_idx]
    if wave1_top<=last_sl_val: continue
    wave1_range_l=wave1_top-last_sl_val
    if wave1_range_l<at[i]*0.5: continue

    post_w1_sl=psl[psl>wave1_top_idx]
    if not len(post_w1_sl): continue
    wave2_bot_idx=post_w1_sl[-1]; wave2_bot=l[wave2_bot_idx]
    retrace_l=(wave1_top-wave2_bot)/wave1_range_l

    if not (0.30<=retrace_l<=0.80): continue
    if wave2_bot_idx>=i: continue
    if c[i]<=wave2_bot: continue
    if c[i]<=wave2_bot+at[i]*0.3: continue
    if i <= wave2_bot_idx + 2: continue

    win,pnl=eval_trade(o,h,l,c,at,i,'long',n)
    if win is not None:
        res_wave3_long.append(pnl)
        if reg[i] in ('UP','DOWN'):
            res_wave3_l_slop.append(pnl)

report("3波ショート (下降1波→2波フィボ戻り→3波起点)", res_wave3_short)
report("3波ショート + SLOPING", res_wave3_s_slop)
report("3波ロング (上昇1波→2波フィボ押し→3波起点)", res_wave3_long)
report("3波ロング + SLOPING", res_wave3_l_slop)

# ──────────────────────────────────────────────────────────────
# 3. フィボリトレース 38.2 / 50.0 / 61.8%
# スイングHigh→Lowのフィボ水準へのリテスト → エントリー
# ──────────────────────────────────────────────────────────────
print("\n### 3. フィボリトレース エントリー ###")
print("    スイング幅のフィボ水準(38.2/50/61.8%)到達→エントリー")
print("    下降後の戻りでフィボ=ショート / 上昇後の押しでフィボ=ロング")

FIBO_LEVELS=[0.382, 0.500, 0.618, 0.786]

for fib in FIBO_LEVELS:
    res_fib_s=[]; res_fib_l=[]; res_fib_s_s=[]; res_fib_l_s=[]

    for i in range(10, n):
        if np.isnan(at[i]) or at[i]<=0: continue
        psh=shi[shi<i]; psl=sli[sli<i]
        if len(psh)<1 or len(psl)<1: continue

        last_sh_idx=psh[-1]; last_sl_idx=psl[-1]

        # 下降後の戻りショート: SH→SL(下降) → フィボ戻りへのリテスト
        if last_sh_idx>last_sl_idx:
            # SH→SLの下降が直近
            top=h[last_sh_idx]; bot=l[last_sl_idx]
            if top<=bot: continue
            fib_level=top-(top-bot)*fib  # 下から数えたフィボ (戻り水準)
            tol=0.20*at[i]
            if abs(c[i]-fib_level)<=tol and c[i]<top:
                win,pnl=eval_trade(o,h,l,c,at,i,'short',n)
                if win is not None:
                    res_fib_s.append(pnl)
                    if reg[i] in ('UP','DOWN'): res_fib_s_s.append(pnl)

        # 上昇後の押しロング: SL→SH(上昇) → フィボ押しへのリテスト
        if last_sl_idx>last_sh_idx:
            bot=l[last_sl_idx]; top=h[last_sh_idx]
            # 修正: SL→SH上昇の場合
            sl_before_sh=psl[psl<last_sh_idx]
            sh_before_sl=psh[psh<last_sl_idx]
            if not len(sh_before_sl): continue
            # 上昇: sh_before_sl[-1]→psl[-1]→psh[-1]みたいな...
            # 簡易: 直近SL後のSH(top) → 現在価格が押している
            post_sh=psh[psh>last_sl_idx]
            if not len(post_sh): continue
            swing_top=h[post_sh[-1]]; swing_bot=l[last_sl_idx]
            if swing_top<=swing_bot: continue
            fib_level=swing_top-(swing_top-swing_bot)*fib  # 上から数えたフィボ (押し水準)
            tol=0.20*at[i]
            if abs(c[i]-fib_level)<=tol and c[i]>swing_bot:
                win,pnl=eval_trade(o,h,l,c,at,i,'long',n)
                if win is not None:
                    res_fib_l.append(pnl)
                    if reg[i] in ('UP','DOWN'): res_fib_l_s.append(pnl)

    report(f"Fib{fib:.1%} 戻りショート", res_fib_s)
    report(f"Fib{fib:.1%} 戻りショート + SLOPING", res_fib_s_s)
    report(f"Fib{fib:.1%} 押しロング", res_fib_l)
    report(f"Fib{fib:.1%} 押しロング + SLOPING", res_fib_l_s)

# ──────────────────────────────────────────────────────────────
# 4. ダウ理論 × フィボ合わせ技
# 切り下げ確認後 + フィボ61.8%戻りでショート (最強フィルター)
# ──────────────────────────────────────────────────────────────
print("\n### 4. 合わせ技: 切り下げ確認 × フィボ戻り ###")

res_combo_s=[]; res_combo_s_ma=[]

for i in range(15, n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if len(psh)<2 or len(psl)<2: continue

    sh1=h[psh[-2]]; sh2=h[psh[-1]]
    sl1=l[psl[-2]]; sl2=l[psl[-1]]

    # 切り下げ確認
    if not (sh2<sh1 and sl2<sl1): continue

    # 直近の下降スイング: sh2→sl2
    if sh2<=sl2: continue
    swing_range=sh2-sl2

    # フィボ50-61.8%戻り水準
    fib50=sh2-swing_range*0.50
    fib618=sh2-swing_range*0.618
    # 価格がfib50〜sh2の間にある (戻り過程)
    if not (fib618-at[i]*0.2 <= c[i] <= sh2-at[i]*0.1): continue

    win,pnl=eval_trade(o,h,l,c,at,i,'short',n)
    if win is not None:
        res_combo_s.append(pnl)
        if not np.isnan(ma200[i]) and c[i]<ma200[i]:
            res_combo_s_ma.append(pnl)

report("切り下げ確認 × Fib50-61.8%戻り → ショート", res_combo_s)
report("切り下げ確認 × Fib50-61.8%戻り × MA200下 → ショート", res_combo_s_ma)

print("\n" + "="*100)
print("# まとめ")
print("="*100)
