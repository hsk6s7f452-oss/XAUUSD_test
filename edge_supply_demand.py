"""
需給エッジ検証
1. Liquidity Pool (等値高値/安値スイープ)
2. Order Block 精密版 (MSB直前の最終反対色ローソク)
3. Premium / Discount Zone
4. Breaker Block (元サポ→レジ / 元レジ→サポ)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

df=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
df=df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
n=len(df); hour=df['Date'].dt.hour.values
o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values

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

sh1=np.zeros(n,bool); sl1=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh1[i]=True
    if l[i]==l[i-1:i+2].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl1[i]=True
shi=np.where(sh1)[0]; sli=np.where(sl1)[0]

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
    if not pnls: print(f"  {name:60s} n=0"); return False
    n_=len(pnls); wr=100*sum(1 for p in pnls if p>0)/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    ok=e1>0 and e2>0
    flag='✅STABLE' if ok else '❌'
    if ok and ev>5 and n_>=15: flag+='🔥'
    print(f"  {name:60s} WR={wr:4.0f}% EV={ev:+6.2f} n={n_:4d}(~{n_/5:.0f}/月) [{e1:+.1f}/{e2:+.1f}] {flag}")
    return ok

# ═══════════════════════════════════════════════════════════
# 1. Liquidity Pool — 等値高値/安値
# 等値高値: 過去N本以内に同水準(±ATR*0.1)の高値が2本以上 → 流動性蓄積
# スイープ: 現在足が等値高値を上回る(ヒゲで超えて実体は下) → 反転ショート
# ═══════════════════════════════════════════════════════════
print("="*90)
print("# 1. Liquidity Pool スイープ")
print("="*90)

res_lp_s=[]; res_lp_s_sl=[]; res_lp_s_day=[]
res_lp_l=[]; res_lp_l_sl=[]

LOOKBACK=50

for i in range(LOOKBACK+5, n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=at[i]*0.15

    # 等値高値: 過去LOOKBACK本の高値クラスター
    past_h=h[max(0,i-LOOKBACK):i]
    for ref_h in past_h:
        # ref_hと似た高値が他に何本あるか
        cluster=np.sum(np.abs(past_h-ref_h)<=tol)
        if cluster<2: continue
        # スイープ条件: 現在足の高値がref_hを超えたが終値は下
        if h[i]>ref_h+tol*0.3 and c[i]<ref_h:
            p=trade(i,'short')
            if p is not None:
                res_lp_s.append(p)
                if reg[i] in ('UP','DOWN'): res_lp_s_sl.append(p)
                if reg[i] in ('UP','DOWN') and 7<=hour[i]<=19: res_lp_s_day.append(p)
            break

    # 等値安値スイープ → ロング
    past_l=l[max(0,i-LOOKBACK):i]
    for ref_l in past_l:
        cluster=np.sum(np.abs(past_l-ref_l)<=tol)
        if cluster<2: continue
        if l[i]<ref_l-tol*0.3 and c[i]>ref_l:
            p=trade(i,'long')
            if p is not None:
                res_lp_l.append(p)
                if reg[i] in ('UP','DOWN'): res_lp_l_sl.append(p)
            break

rep("等値高値スイープ→ショート (素)", res_lp_s)
rep("等値高値スイープ→ショート + SLOPING", res_lp_s_sl)
rep("等値高値スイープ→ショート + SLOPING + UTC07-19", res_lp_s_day)
rep("等値安値スイープ→ロング (素)", res_lp_l)
rep("等値安値スイープ→ロング + SLOPING", res_lp_l_sl)

# ═══════════════════════════════════════════════════════════
# 2. Order Block 精密版
# 下降OB: 直近の下降MSBを引き起こした直前の最後の陽線 → 戻りでショート
# MSB = 直前スイングローを実体でブレイク
# ═══════════════════════════════════════════════════════════
print("\n" + "="*90)
print("# 2. Order Block (精密版)")
print("="*90)

res_ob_s=[]; res_ob_s_sl=[]; res_ob_s_ma=[]
res_ob_l=[]; res_ob_l_sl=[]

ob_zones_bear=[]  # (ob_top, ob_bot, created_at_bar)
ob_zones_bull=[]

for i in range(10, n):
    if np.isnan(at[i]) or at[i]<=0: continue

    # 下降MSB: 直前スイングロー(sli)の実体を下ブレイク
    psl=sli[sli<i-2]
    if len(psl)>=1:
        sl_lv=l[psl[-1]]
        if c[i]<sl_lv and c[i-1]>=sl_lv:  # このバーでブレイク確定
            # OBを探す: ブレイクの2〜10本前の最後の陽線
            for k in range(i-1, max(0,i-15), -1):
                if c[k]>o[k]:  # 陽線 = ベアOB
                    ob_top=max(o[k],c[k]); ob_bot=min(o[k],c[k])
                    ob_zones_bear.append((ob_top, ob_bot, k, i))
                    break

    # 上昇MSB: 直前スイングハイを実体で上ブレイク
    psh=shi[shi<i-2]
    if len(psh)>=1:
        sh_lv=h[psh[-1]]
        if c[i]>sh_lv and c[i-1]<=sh_lv:
            for k in range(i-1, max(0,i-15), -1):
                if c[k]<o[k]:  # 陰線 = ブルOB
                    ob_top=max(o[k],c[k]); ob_bot=min(o[k],c[k])
                    ob_zones_bull.append((ob_top, ob_bot, k, i))
                    break

# OBへの戻りでエントリー
used_ob=set()
for i in range(20, n-1):
    if np.isnan(at[i]) or at[i]<=0: continue

    # ベアOBへの戻り→ショート
    for ob_top, ob_bot, ob_bar, msb_bar in ob_zones_bear:
        if i<=msb_bar or (ob_bar,msb_bar) in used_ob: continue
        # 価格がOBゾーン内に入った
        if ob_bot<=c[i]<=ob_top or (h[i]>=ob_bot and l[i]<=ob_top):
            if c[i]<=ob_top:  # まだ上抜けていない
                p=trade(i,'short')
                if p is not None:
                    res_ob_s.append(p)
                    used_ob.add((ob_bar,msb_bar))
                    if reg[i] in ('UP','DOWN'): res_ob_s_sl.append(p)
                    if reg[i] in ('UP','DOWN') and not np.isnan(ma200[i]) and c[i]<ma200[i]:
                        res_ob_s_ma.append(p)
                break

    # ブルOBへの戻り→ロング
    for ob_top, ob_bot, ob_bar, msb_bar in ob_zones_bull:
        if i<=msb_bar or (ob_bar,msb_bar,'l') in used_ob: continue
        if ob_bot<=c[i]<=ob_top or (h[i]>=ob_bot and l[i]<=ob_top):
            if c[i]>=ob_bot:
                p=trade(i,'long')
                if p is not None:
                    res_ob_l.append(p)
                    used_ob.add((ob_bar,msb_bar,'l'))
                    if reg[i] in ('UP','DOWN'): res_ob_l_sl.append(p)
                break

rep("ベアOB戻り→ショート (素)", res_ob_s)
rep("ベアOB戻り→ショート + SLOPING", res_ob_s_sl)
rep("ベアOB戻り→ショート + SLOPING + MA200下", res_ob_s_ma)
rep("ブルOB戻り→ロング (素)", res_ob_l)
rep("ブルOB戻り→ロング + SLOPING", res_ob_l_sl)

# ═══════════════════════════════════════════════════════════
# 3. Premium / Discount Zone
# 直近スイングHigh〜Low の50%より上 = プレミアム → 売り場
# 直近スイングHigh〜Low の50%より下 = ディスカウント → 買い場
# ═══════════════════════════════════════════════════════════
print("\n" + "="*90)
print("# 3. Premium / Discount Zone")
print("="*90)

res_prem=[]; res_prem_sl=[]; res_prem_ob=[]; res_prem_fib=[]
res_disc=[]; res_disc_sl=[]

for i in range(30, n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if not len(psh) or not len(psl): continue

    # 直近スイング高値と安値
    recent_sh=h[psh[-1]]; recent_sl=l[psl[-1]]
    if recent_sh<=recent_sl: continue
    midpoint=(recent_sh+recent_sl)/2
    eq38=recent_sl+(recent_sh-recent_sl)*0.38
    eq62=recent_sl+(recent_sh-recent_sl)*0.62

    # プレミアムゾーン(上62%以上) × SLOPING → ショート
    if c[i]>eq62:
        p=trade(i,'short')
        if p is not None:
            res_prem.append(p)
            if reg[i] in ('UP','DOWN'): res_prem_sl.append(p)

    # ディスカウントゾーン(下38%以下) × SLOPING → ロング
    if c[i]<eq38:
        p=trade(i,'long')
        if p is not None:
            res_disc.append(p)
            if reg[i] in ('UP','DOWN'): res_disc_sl.append(p)

rep("プレミアムゾーン(62%上) → ショート", res_prem)
rep("プレミアムゾーン(62%上) + SLOPING → ショート", res_prem_sl)
rep("ディスカウントゾーン(38%下) → ロング", res_disc)
rep("ディスカウントゾーン(38%下) + SLOPING → ロング", res_disc_sl)

# ═══════════════════════════════════════════════════════════
# 4. Breaker Block
# 元サポート(スイングロー)がブレイクされた → 戻りでレジスタンスに転換
# 元レジスタンス(スイングハイ)がブレイクされた → 戻りでサポートに転換
# ═══════════════════════════════════════════════════════════
print("\n" + "="*90)
print("# 4. Breaker Block")
print("="*90)

res_bb_s=[]; res_bb_s_sl=[]; res_bb_s_day=[]
res_bb_l=[]; res_bb_l_sl=[]

breakers_bear=[]  # 元サポート→ブレイク→ブレーカー(レジ)
breakers_bull=[]  # 元レジスタンス→ブレイク→ブレーカー(サポ)

for i in range(10, n):
    if np.isnan(at[i]) or at[i]<=0: continue

    # ブレーカー生成: スイングローがブレイクされた
    psl=sli[sli<i-2]
    if len(psl)>=1:
        sl_idx=psl[-1]; sl_lv=l[sl_idx]
        if c[i]<sl_lv-at[i]*0.1:  # 実体でブレイク
            breakers_bear.append((sl_lv, l[sl_idx], sl_idx, i))

    # ブレーカー生成: スイングハイがブレイクされた
    psh=shi[shi<i-2]
    if len(psh)>=1:
        sh_idx=psh[-1]; sh_lv=h[sh_idx]
        if c[i]>sh_lv+at[i]*0.1:
            breakers_bull.append((sh_lv, h[sh_idx], sh_idx, i))

used_bb=set()
for i in range(20, n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=at[i]*0.20

    # ブレーカーレジ(元サポ)への戻り→ショート
    for bb_lv,_,bb_bar,brk_bar in breakers_bear[-30:]:
        key=(bb_bar,'s')
        if i<=brk_bar or key in used_bb: continue
        if abs(c[i]-bb_lv)<=tol and c[i]<bb_lv+tol:
            p=trade(i,'short')
            if p is not None:
                res_bb_s.append(p); used_bb.add(key)
                if reg[i] in ('UP','DOWN'): res_bb_s_sl.append(p)
                if reg[i] in ('UP','DOWN') and 7<=hour[i]<=19: res_bb_s_day.append(p)
            break

    # ブレーカーサポ(元レジ)への押し→ロング
    for bb_lv,_,bb_bar,brk_bar in breakers_bull[-30:]:
        key=(bb_bar,'l')
        if i<=brk_bar or key in used_bb: continue
        if abs(c[i]-bb_lv)<=tol and c[i]>bb_lv-tol:
            p=trade(i,'long')
            if p is not None:
                res_bb_l.append(p); used_bb.add(key)
                if reg[i] in ('UP','DOWN'): res_bb_l_sl.append(p)
            break

rep("ブレーカー(元サポ→レジ)戻り→ショート", res_bb_s)
rep("ブレーカー(元サポ→レジ) + SLOPING", res_bb_s_sl)
rep("ブレーカー(元サポ→レジ) + SLOPING + UTC07-19", res_bb_s_day)
rep("ブレーカー(元レジ→サポ)押し→ロング", res_bb_l)
rep("ブレーカー(元レジ→サポ) + SLOPING", res_bb_l_sl)

print("\n" + "="*90)
