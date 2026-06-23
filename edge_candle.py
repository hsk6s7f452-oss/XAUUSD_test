"""
ローソク足パターン検証
1. ピンバー (シューティングスター/ハンマー)
2. 包み足 (ベアリッシュ/ブリッシュエンガルフィング)
3. 孕み足 (インサイドバー)
4. 十字線 (ドージ) at キーレベル
5. 三羽烏 / 三兵 (3本連続)
6. ツイーザートップ/ボトム
7. モーニングスター/イブニングスター
+ 各パターン × SLOPING × 時間フィルター
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

tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
at=np.full(n,np.nan); at[1:]=pd.Series(tr).rolling(14).mean().values
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values
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
    if not pnls: print(f"  {name:62s} n=0"); return False
    n_=len(pnls); wr=100*sum(1 for p in pnls if p>0)/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    ok=e1>0 and e2>0
    flag='✅STABLE' if ok else '❌'
    if ok and ev>5 and n_>=15: flag+='🔥'
    if ok and ev>10 and n_>=15: flag+='🔥'
    print(f"  {name:62s} WR={wr:4.0f}% EV={ev:+6.2f} n={n_:4d}(~{n_/5:.0f}/月) [{e1:+.1f}/{e2:+.1f}] {flag}")
    return ok

# ── パターン判定ヘルパー ──
def body(i):    return abs(c[i]-o[i])
def upper_w(i): return h[i]-max(c[i],o[i])
def lower_w(i): return min(c[i],o[i])-l[i]
def total_r(i): return h[i]-l[i]
def is_bear(i): return c[i]<o[i]
def is_bull(i): return c[i]>o[i]

print("="*95)
print("# ローソク足パターン × SLOPING / 時間フィルター")
print("="*95)

# ═══════════════════════════════════════════════════
# 1. ピンバー
# シューティングスター: 上ヒゲが実体の2倍以上、下ヒゲ小さい、実体が下半分
# ハンマー: 下ヒゲが実体の2倍以上、上ヒゲ小さい、実体が上半分
# ═══════════════════════════════════════════════════
print("\n### 1. ピンバー (シューティングスター / ハンマー) ###")

ps_s=[]; ps_s_sl=[]; ps_s_day=[]; ps_s_sl_ma50=[]
ham_l=[]; ham_l_sl=[]

for i in range(5,n-1):
    if np.isnan(at[i]) or at[i]<=0 or total_r(i)<at[i]*0.3: continue
    uw=upper_w(i); lw=lower_w(i); bd=body(i); tr_=total_r(i)

    # シューティングスター (ベアリッシュピンバー)
    # 上ヒゲ≥実体2倍 & 上ヒゲ≥全体50% & 下ヒゲ小さい
    if uw>=bd*2.0 and uw>=tr_*0.5 and lw<=bd*0.5:
        p=trade(i,'short')
        if p is not None:
            ps_s.append(p)
            if reg[i] in ('UP','DOWN'): ps_s_sl.append(p)
            if reg[i] in ('UP','DOWN') and 7<=hour[i]<=19: ps_s_day.append(p)
            if reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
                ps_s_sl_ma50.append(p)

    # ハンマー (ブリッシュピンバー)
    if lw>=bd*2.0 and lw>=tr_*0.5 and uw<=bd*0.5:
        p=trade(i,'long')
        if p is not None:
            ham_l.append(p)
            if reg[i] in ('UP','DOWN'): ham_l_sl.append(p)

rep("シューティングスター→ショート", ps_s)
rep("シューティングスター + SLOPING", ps_s_sl)
rep("シューティングスター + SLOPING + UTC07-19", ps_s_day)
rep("シューティングスター + SLOPING + MA50上", ps_s_sl_ma50)
rep("ハンマー→ロング", ham_l)
rep("ハンマー + SLOPING", ham_l_sl)

# ═══════════════════════════════════════════════════
# 2. 包み足 (エンガルフィング)
# ベアリッシュ: 今足の実体が前足実体を完全包含 & 今足=陰線
# ブリッシュ: 今足の実体が前足実体を完全包含 & 今足=陽線
# ═══════════════════════════════════════════════════
print("\n### 2. 包み足 (エンガルフィング) ###")

eng_s=[]; eng_s_sl=[]; eng_s_day=[]; eng_s_ma50=[]
eng_l=[]; eng_l_sl=[]

for i in range(1,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    # ベアリッシュ包み足
    if (is_bear(i) and is_bull(i-1) and
        o[i]>=c[i-1] and c[i]<=o[i-1] and
        body(i)>body(i-1)):
        p=trade(i,'short')
        if p is not None:
            eng_s.append(p)
            if reg[i] in ('UP','DOWN'): eng_s_sl.append(p)
            if reg[i] in ('UP','DOWN') and 7<=hour[i]<=19: eng_s_day.append(p)
            if reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
                eng_s_ma50.append(p)

    # ブリッシュ包み足
    if (is_bull(i) and is_bear(i-1) and
        o[i]<=c[i-1] and c[i]>=o[i-1] and
        body(i)>body(i-1)):
        p=trade(i,'long')
        if p is not None:
            eng_l.append(p)
            if reg[i] in ('UP','DOWN'): eng_l_sl.append(p)

rep("ベアリッシュ包み足→ショート", eng_s)
rep("ベアリッシュ包み足 + SLOPING", eng_s_sl)
rep("ベアリッシュ包み足 + SLOPING + UTC07-19", eng_s_day)
rep("ベアリッシュ包み足 + SLOPING + MA50上", eng_s_ma50)
rep("ブリッシュ包み足→ロング", eng_l)
rep("ブリッシュ包み足 + SLOPING", eng_l_sl)

# ═══════════════════════════════════════════════════
# 3. 孕み足 (インサイドバー)
# 今足の高値・安値が前足の範囲内 → ブレイクアウト方向へ
# ═══════════════════════════════════════════════════
print("\n### 3. 孕み足 (インサイドバー) ###")

ib_s=[]; ib_s_sl=[]; ib_l=[]; ib_l_sl=[]

for i in range(1,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    # インサイドバー
    if h[i]<h[i-1] and l[i]>l[i-1]:
        # 次足がどちらにブレイクしたか (i+1が確定)
        if i+2>=n: continue
        # 翌足でブレイクダウン → ショート
        if c[i+1]<l[i] and is_bear(i-1):  # 親足が陰線
            p=trade(i+1,'short')
            if p is not None:
                ib_s.append(p)
                if reg[i] in ('UP','DOWN'): ib_s_sl.append(p)
        if c[i+1]>h[i] and is_bull(i-1):
            p=trade(i+1,'long')
            if p is not None:
                ib_l.append(p)
                if reg[i] in ('UP','DOWN'): ib_l_sl.append(p)

rep("孕み足ブレイクダウン→ショート", ib_s)
rep("孕み足ブレイクダウン + SLOPING", ib_s_sl)
rep("孕み足ブレイクアップ→ロング", ib_l)
rep("孕み足ブレイクアップ + SLOPING", ib_l_sl)

# ═══════════════════════════════════════════════════
# 4. 十字線 (ドージ) — 実体が全体の10%以下
# ═══════════════════════════════════════════════════
print("\n### 4. ドージ (十字線) at キーレベル ###")

doji_s=[]; doji_s_sl=[]; doji_l=[]; doji_l_sl=[]

sh1=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh1[i]=True
shi=np.where(sh1)[0]
ma50_arr=pd.Series(c).rolling(50).mean().values

for i in range(5,n-1):
    if np.isnan(at[i]) or at[i]<=0 or total_r(i)<at[i]*0.2: continue
    if body(i)>total_r(i)*0.15: continue  # ドージ条件

    # スイングハイ付近のドージ → ショート
    psh=shi[shi<i]
    if len(psh):
        lv=h[psh[-1]]
        if abs(c[i]-lv)<=at[i]*0.3 and c[i]<lv:
            p=trade(i,'short')
            if p is not None:
                doji_s.append(p)
                if reg[i] in ('UP','DOWN'): doji_s_sl.append(p)

    # MA200付近のドージ → トレンド方向
    if not np.isnan(ma200[i]):
        if abs(c[i]-ma200[i])<=at[i]*0.5:
            if reg[i]=='DOWN':
                p=trade(i,'short')
                if p is not None: doji_l.append(p)  # DOWNなのでショート
            elif reg[i]=='UP':
                p=trade(i,'long')
                if p is not None: doji_l.append(p)

rep("ドージ at SH抵抗→ショート", doji_s)
rep("ドージ at SH抵抗 + SLOPING", doji_s_sl)
rep("ドージ at MA200 → トレンド方向", doji_l)

# ═══════════════════════════════════════════════════
# 5. 三羽烏 / 赤三兵
# ═══════════════════════════════════════════════════
print("\n### 5. 三羽烏 / 赤三兵 ###")

crow_s=[]; crow_s_sl=[]
sol_l=[]; sol_l_sl=[]

for i in range(3,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    # 三羽烏: 3本連続陰線、各実体が大きい、各終値が前足終値より低い
    if (all(is_bear(i-k) for k in range(3)) and
        all(body(i-k)>=at[i]*0.4 for k in range(3)) and
        c[i]<c[i-1]<c[i-2]):
        p=trade(i,'short')
        if p is not None:
            crow_s.append(p)
            if reg[i] in ('UP','DOWN'): crow_s_sl.append(p)

    # 赤三兵: 3本連続陽線
    if (all(is_bull(i-k) for k in range(3)) and
        all(body(i-k)>=at[i]*0.4 for k in range(3)) and
        c[i]>c[i-1]>c[i-2]):
        p=trade(i,'long')
        if p is not None:
            sol_l.append(p)
            if reg[i] in ('UP','DOWN'): sol_l_sl.append(p)

rep("三羽烏(3陰線)→ショート継続", crow_s)
rep("三羽烏 + SLOPING", crow_s_sl)
rep("赤三兵(3陽線)→ロング継続", sol_l)
rep("赤三兵 + SLOPING", sol_l_sl)

# ═══════════════════════════════════════════════════
# 6. ツイーザートップ/ボトム
# ═══════════════════════════════════════════════════
print("\n### 6. ツイーザートップ / ボトム ###")

tw_s=[]; tw_s_sl=[]; tw_l=[]; tw_l_sl=[]

for i in range(1,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=at[i]*0.1

    # ツイーザートップ: 2本連続でほぼ同じ高値、2本目が陰線
    if abs(h[i]-h[i-1])<=tol and is_bear(i) and is_bull(i-1):
        p=trade(i,'short')
        if p is not None:
            tw_s.append(p)
            if reg[i] in ('UP','DOWN'): tw_s_sl.append(p)

    # ツイーザーボトム: 2本連続でほぼ同じ安値、2本目が陽線
    if abs(l[i]-l[i-1])<=tol and is_bull(i) and is_bear(i-1):
        p=trade(i,'long')
        if p is not None:
            tw_l.append(p)
            if reg[i] in ('UP','DOWN'): tw_l_sl.append(p)

rep("ツイーザートップ→ショート", tw_s)
rep("ツイーザートップ + SLOPING", tw_s_sl)
rep("ツイーザーボトム→ロング", tw_l)
rep("ツイーザーボトム + SLOPING", tw_l_sl)

# ═══════════════════════════════════════════════════
# 7. イブニングスター (3本パターン)
# 陽線 → 小実体(星) → 陰線(半値以上食い込み)
# ═══════════════════════════════════════════════════
print("\n### 7. イブニングスター / モーニングスター ###")

eve_s=[]; eve_s_sl=[]
mor_l=[]; mor_l_sl=[]

for i in range(2,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    # イブニングスター
    if (is_bull(i-2) and body(i-2)>=at[i]*0.5 and  # 1本目: 陽線
        body(i-1)<=body(i-2)*0.3 and               # 2本目: 小実体(星)
        h[i-1]>c[i-2] and                          # 星はギャップ上
        is_bear(i) and body(i)>=at[i]*0.4 and      # 3本目: 陰線
        c[i]<=(o[i-2]+c[i-2])/2):                  # 半値以上食い込み
        p=trade(i,'short')
        if p is not None:
            eve_s.append(p)
            if reg[i] in ('UP','DOWN'): eve_s_sl.append(p)

    # モーニングスター
    if (is_bear(i-2) and body(i-2)>=at[i]*0.5 and
        body(i-1)<=body(i-2)*0.3 and
        l[i-1]<c[i-2] and
        is_bull(i) and body(i)>=at[i]*0.4 and
        c[i]>=(o[i-2]+c[i-2])/2):
        p=trade(i,'long')
        if p is not None:
            mor_l.append(p)
            if reg[i] in ('UP','DOWN'): mor_l_sl.append(p)

rep("イブニングスター→ショート", eve_s)
rep("イブニングスター + SLOPING", eve_s_sl)
rep("モーニングスター→ロング", mor_l)
rep("モーニングスター + SLOPING", mor_l_sl)

# ═══════════════════════════════════════════════════
# 8. マルボウズ (全体がほぼ実体 = 強いモメンタム)
# ═══════════════════════════════════════════════════
print("\n### 8. マルボウズ (強モメンタム継続) ###")

mar_s=[]; mar_s_sl=[]; mar_l=[]; mar_l_sl=[]

for i in range(1,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    if body(i)<total_r(i)*0.85 or body(i)<at[i]*0.5: continue  # マルボウズ条件

    if is_bear(i):  # 大陰線マルボウズ → 継続ショート
        p=trade(i,'short')
        if p is not None:
            mar_s.append(p)
            if reg[i] in ('UP','DOWN'): mar_s_sl.append(p)

    if is_bull(i):  # 大陽線マルボウズ → 継続ロング
        p=trade(i,'long')
        if p is not None:
            mar_l.append(p)
            if reg[i] in ('UP','DOWN'): mar_l_sl.append(p)

rep("ベアマルボウズ→継続ショート", mar_s)
rep("ベアマルボウズ + SLOPING", mar_s_sl)
rep("ブルマルボウズ→継続ロング", mar_l)
rep("ブルマルボウズ + SLOPING", mar_l_sl)

print("\n" + "="*95)
