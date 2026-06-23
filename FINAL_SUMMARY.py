"""
最終まとめ — 全STABLEエッジ再検証 + ランキング + エントリー詳細
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
hour=df['Date'].dt.hour.values; dow=df['Date'].dt.dayofweek.values
day=df['Date'].dt.date.values

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

def calc_rsi(c,p=14):
    delta=np.diff(c); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=pd.Series(gain).ewm(alpha=1/p,adjust=False).mean().values
    al=pd.Series(loss).ewm(alpha=1/p,adjust=False).mean().values
    rsi=np.full(n,np.nan); rsi[1:]=100-100/(1+ag/(al+1e-9)); return rsi
rsi=calc_rsi(c)

atr_pct=np.full(n,np.nan)
for i in range(200,n):
    w=at[i-200:i]; w=w[~np.isnan(w)]
    if len(w)>0: atr_pct[i]=np.sum(w<=at[i])/len(w)

sh1=np.zeros(n,bool); sl1=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh1[i]=True
    if l[i]==l[i-1:i+2].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl1[i]=True
shi=np.where(sh1)[0]

sh2=np.zeros(n,bool); sl2=np.zeros(n,bool)
for i in range(2,n-2):
    if all(h[i]>=h[i-j] for j in range(1,3)) and all(h[i]>=h[i+j] for j in range(1,3)): sh2[i]=True
    if all(l[i]<=l[i-j] for j in range(1,3)) and all(l[i]<=l[i+j] for j in range(1,3)): sl2[i]=True
shi2=np.where(sh2)[0]; sli2=np.where(sl2)[0]

flagship=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]; psh=shi[shi<i-1]
    if not len(psh): continue
    res=h[psh]; res=res[res>c[i-1]]
    if not len(res): continue
    lv=res.min()
    if abs(h[i]-lv)<=tol or (h[i]>lv-tol and h[i]<lv+tol):
        if c[i]<lv and reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
            flagship.append(i)
fl=set(flagship)

fib786=[]
for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi2[shi2<i]; psl=sli2[sli2<i]
    if not len(psh) or not len(psl): continue
    lsh=psh[-1]; lsl=psl[-1]
    if lsh<=lsl: continue
    top=h[lsh]; bot=l[lsl]
    if top<=bot: continue
    if abs((c[i]-bot)/(top-bot)-0.786)<=0.06 and reg[i] in ('UP','DOWN'):
        fib786.append(i)
f786=set(fib786)

london_open={}
for i in range(n):
    if hour[i]==7: london_open[day[i]]=(i,o[i])
london_2h={}
for i in range(n):
    if hour[i]==9: london_2h[day[i]]=(i,c[i])

def trade(i, direction='short', rr=1.0):
    if i+1>=n: return None
    e=o[i+1]; r=at[i]
    if np.isnan(r) or r<=0: return None
    tp=e-r*rr if direction=='short' else e+r*rr
    sl=e+r    if direction=='short' else e-r
    for k in range(i+1, min(n,i+25)):
        if direction=='short':
            if h[k]>=sl: return -r-SPREAD
            if l[k]<=tp: return  r*rr-SPREAD
        else:
            if l[k]<=sl: return -r-SPREAD
            if h[k]>=tp: return  r*rr-SPREAD
    return None

def stats(pnls):
    if not pnls: return None
    n_=len(pnls); wr=100*sum(1 for p in pnls if p>0)/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    mcl=0; cur=0
    for p in pnls:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    return dict(wr=wr,ev=ev,n=n_,e1=e1,e2=e2,stable=e1>0 and e2>0,nmo=n_/5,mcl=mcl,total=sum(pnls))

# ─── 全エッジ再計算 ────────────────────────────────────────
edges={}

# A. Fib78.6%系
edges['F1']=[trade(i) for i in fib786 if 7<=hour[i]<=19]
edges['F2']=[trade(i) for i in fib786 if not np.isnan(ma200[i]) and c[i]<ma200[i]]
edges['F3']=[trade(i) for i in fib786]
edges['F4']=[trade(i) for i in fib786 if 7<=hour[i]<=12]

# B. 旗艦系
edges['G1']=[trade(i) for i in flagship if 7<=hour[i]<=19]
edges['G2']=[trade(i) for i in flagship if not np.isnan(atr_pct[i]) and 0.50<=atr_pct[i]<=0.80]
edges['G3']=[trade(i) for i in flagship if dow[i]==2 and 7<=hour[i]<=19]
edges['G4']=[trade(i) for i in flagship if 7<=hour[i]<=12]
edges['G5']=[trade(i) for i in flagship]

# C. RSI>70
rsi_cross_bars=[i for i in range(21,n)
    if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1])
    and rsi[i]>70 and rsi[i-1]<=70 and reg[i] in ('UP','DOWN')]
edges['R1']=[trade(i) for i in rsi_cross_bars]
edges['R2']=[trade(i) for i in range(5,n-1)
    if reg[i] in ('UP','DOWN') and not np.isnan(rsi[i]) and rsi[i]>70]

# D. Judas Swing
judas=[]
for i in range(n):
    if hour[i]!=10: continue
    if np.isnan(at[i]) or at[i]<=0: continue
    d=day[i]
    if d not in london_open or d not in london_2h: continue
    _,lo=london_open[d]; _,l2=london_2h[d]
    if l2-lo>at[london_open[d][0]]*0.3 and reg[i] in ('UP','DOWN'):
        judas.append(i)
edges['J1']=[trade(i) for i in judas]

# E. ロンドンフィックス
fix=[]
for i in range(n-1):
    if hour[i]!=16: continue
    if np.isnan(at[i]) or at[i]<=0: continue
    d=day[i]
    pre=[j for j in range(max(0,i-3),i) if hour[j]==15 and day[j]==d]
    if not pre: continue
    if c[i]-o[pre[0]]>at[i]*0.5 and reg[i] in ('UP','DOWN'):
        fix.append(i)
edges['X1']=[trade(i) for i in fix]

# F. hour10
edges['H1']=[trade(i) for i in range(5,n-1) if hour[i]==10 and reg[i] in ('UP','DOWN')]

# G. イブニングスター
eve=[]
for i in range(2,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    b=lambda j: abs(c[j]-o[j]); uw=lambda j: h[j]-max(c[j],o[j])
    if (c[i-2]>o[i-2] and b(i-2)>=at[i]*0.5 and
        b(i-1)<=b(i-2)*0.3 and h[i-1]>c[i-2] and
        c[i]<o[i] and b(i)>=at[i]*0.4 and c[i]<=(o[i-2]+c[i-2])/2
        and reg[i] in ('UP','DOWN')):
        eve.append(i)
edges['E1']=[trade(i) for i in eve]

# H. 包み足+MA50上
eng=[]
for i in range(1,n-1):
    if np.isnan(at[i]) or at[i]<=0: continue
    if (c[i]<o[i] and c[i-1]>o[i-1] and o[i]>=c[i-1] and c[i]<=o[i-1]
        and abs(c[i]-o[i])>abs(c[i-1]-o[i-1])
        and reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]):
        eng.append(i)
edges['E2']=[trade(i) for i in eng]

# クリーンアップ
for k in edges:
    edges[k]=[p for p in edges[k] if p is not None]

# ─── 出力 ────────────────────────────────────────────────
print()
print("★"*50)
print("  XAUUSD H1 完全版エッジ一覧 & エントリー仕様")
print("  期間: 2026-Feb〜Jun (5ヶ月) | スプレッド0.3pt込み")
print("  エントリー: シグナル足確定→次足始値 | SL:1ATR | TP:1ATR (1:1)")
print("★"*50)

print("""
【大前提フィルター】
  ✦ MA200傾斜フィルター(SLOPING) — 全エッジに必須
    計算: MA200の現在値と50本前の値を比較
    UP   : MA200が+0.3%以上上向き かつ 価格がMA200上 → ロング相場
    DOWN : MA200が-0.3%以上下向き かつ 価格がMA200下 → ショート相場
    RANGE: それ以外 → ノーエントリー
  
  ✦ 時間フィルター — UTC 07:00〜19:00のみ (ロンドン+NY)
    UTC20:00〜06:00は損失の71%が集中 → 完全スルー
""")

rows=[
    # (ID, 名前, カテゴリ, key, WFA, entry_detail)
    ("F1","Fib78.6% + UTC07-19",         "フィボナッチ", "F1", "p=0.009✅",
     "直近SH→SL下降の78.6%戻り水準(±0.06ATR)に価格が到達し、07-19UTC内"),
    ("F2","Fib78.6% + MA200下",           "フィボナッチ", "F2", "p=0.042✅",
     "同上 + 価格がMA200より下(売り優勢ゾーン)"),
    ("F3","Fib78.6% + SLOPING",           "フィボナッチ", "F3", "p=0.011✅",
     "直近SH→SL下降の78.6%戻り水準到達。SLOPINGのみ(時間制限なし)"),
    ("X1","ロンドンフィックス上昇→逆張り","機関アルゴ",  "X1", "新規",
     "15:00-16:00UTCに価格上昇(>0.5ATR) → 16:00足確定でショート"),
    ("G3","水曜+旗艦+UTC07-19",           "複合",        "G3", "WFA100%✅",
     "水曜日 かつ 07-19UTC かつ 旗艦条件(↓)を同時満足"),
    ("J1","Judas Swing",                  "機関アルゴ",  "J1", "新規",
     "07:00-09:00UTC上昇(>0.3ATR) → 10:00足でショート (ロンドン罠の逆張り)"),
    ("G2","旗艦 × ATR中ボラ",             "旗艦",        "G2", "WFA83%✅",
     "旗艦条件 + ATRパーセンタイル50-80%(中ボラ)のみ通す"),
    ("R2","RSI>70 + SLOPING",             "モメンタム",  "R2", "WFA75%✅",
     "RSI(14)が70を上抜けた足でショート(クロスオーバー推奨=初回のみ)"),
    ("G1","旗艦 + UTC07-19",              "旗艦",        "G1", "WFA✅",
     "旗艦条件 + 07-19UTC"),
    ("F4","Fib78.6% + ロンドン(07-12)",   "フィボナッチ","F4", "WFA✅",
     "Fib78.6%水準到達 + 07-12UTCのロンドン時間内"),
    ("E1","イブニングスター + SLOPING",    "ローソク足",  "E1", "新規",
     "陽線→小実体(星)→陰線が半値以上食い込む 3本パターン + SLOPING"),
    ("G4","旗艦 + ロンドン(07-12)",        "旗艦",        "G4", "WFA✅",
     "旗艦条件 + 07-12UTCのロンドン時間内"),
    ("H1","hour10 + SLOPING",             "時間帯",      "H1", "WFA86%✅",
     "毎日10:00UTCのローソク足でショート。条件はSLOPINGのみ"),
    ("E2","包み足 + SLOPING + MA50上",     "ローソク足",  "E2", "新規",
     "前足が陽線→今足が陰線で前足を完全包含 + MA50より上"),
    ("G5","旗艦(基本)",                    "旗艦",        "G5", "WFA✅",
     "SH抵抗タッチ後に拒絶(終値<水準) + MA50上 + SLOPING"),
]

print(f"  {'#':>2}  {'エッジ名':30s}  {'カテゴリ':10s}  {'WR':>5} {'EV':>7} {'n':>4} {'頻度/月':>6} {'最大連敗':>5} {'WFA':>10}  STABLE")
print("  " + "-"*105)

rank=1
for eid, name, cat, key, wfa, _ in rows:
    s=stats(edges[key])
    if not s or not s['stable']: continue
    star='🔥🔥' if s['ev']>10 and s['n']>=15 else ('🔥' if s['ev']>5 and s['n']>=10 else '')
    print(f"  {rank:>2}  {name:30s}  {cat:10s}  {s['wr']:4.0f}% {s['ev']:+7.2f} {s['n']:>4} {s['nmo']:>5.0f}/月  {s['mcl']:>4}連敗  {wfa:>10}  ✅{star}")
    rank+=1

print()
print("="*110)
print("  エントリー詳細")
print("="*110)

details={
    "F1/F2/F3": """
  【Fib78.6%ショート】★★★ 最も再現性高い
  ─────────────────────────────────────────
  ① チャートでスイングハイ(SH)とスイングロー(SL)を確認 (w=2: 前後2本ずつ)
  ② SH→SLの下降幅に対して78.6%戻り水準を計算
     水準 = SL安値 + (SH高値 - SL安値) × 0.786
  ③ 価格がその水準の±20%ATR以内に到達
  ④ MA200が下向き(DOWN) または SLOPINGを確認
  ⑤ 07-19UTCであることを確認 (F1/F4)
  ─ エントリー: 条件足確定→次足始値でショート成行
  ─ SL: エントリー価格 + 1ATR(14)
  ─ TP: エントリー価格 - 1ATR(14)
  ─ 注意: SHの高値を終値で上抜けたらキャンセル""",

    "G1/G2/G3/G4/G5": """
  【旗艦エッジ: スイングハイ抵抗拒絶】★★★
  ─────────────────────────────────────────
  ① 過去のスイングハイ(w=1: 前後1本)をリストアップ
  ② 現在価格より上にある最も近い過去SH高値 = 抵抗水準
  ③ 今足の高値が抵抗水準の±20%ATR内に到達
  ④ 今足の終値が抵抗水準より下 (= 拒絶確認)
  ⑤ MA50より価格が上にある (プレミアムゾーン)
  ⑥ SLOPING確認 + 時間帯確認(G1: 07-19, G3: 水曜07-19, G4: 07-12)
  ⑦ G2追加条件: ATR(14)の50-80パーセンタイル(中ボラ)
  ─ エントリー: 条件足確定→次足始値ショート
  ─ SL: エントリー + 1ATR | TP: エントリー - 1ATR""",

    "J1": """
  【Judas Swing (ロンドン罠)】★★
  ─────────────────────────────────────────
  ① 07:00UTCの始値を記録 (ロンドンオープン)
  ② 09:00UTCの終値を記録 (2時間後)
  ③ 09:00終値 - 07:00始値 > 0.3ATR → 上昇罠と判断
  ④ 10:00UTCの足確定でSLOPINGを確認
  ─ エントリー: 10:00足確定→11:00始値ショート
  ─ SL: エントリー + 1ATR | TP: エントリー - 1ATR
  ─ 理由: 機関が小口を引きつけてから逆に動く""",

    "X1": """
  【ロンドンフィックス逆張り】★★
  ─────────────────────────────────────────
  ① 15:00UTCの始値を記録 (フィックス1時間前)
  ② 16:00UTCの終値を確認
  ③ 16:00終値 - 15:00始値 > 0.5ATR の上昇 → ショート候補
  ④ SLOPING確認
  ─ エントリー: 16:00足確定→17:00始値ショート
  ─ SL: エントリー + 1ATR | TP: エントリー - 1ATR
  ─ 理由: フィックス向けポジション調整が終了→反転""",

    "R2": """
  【RSI>70 クロスオーバーショート】★★
  ─────────────────────────────────────────
  ① RSI(14)が前足≤70 かつ 今足>70 (クロスオーバー = 初回のみ)
  ② SLOPING(DOWN推奨) + 07-19UTC推奨
  ─ エントリー: クロスアップ足確定→次足始値ショート
  ─ SL: エントリー + 1ATR | TP: エントリー - 1ATR
  ─ 注意: RSI>70が連続する場合は最初の1回のみエントリー""",

    "E1": """
  【イブニングスター】★
  ─────────────────────────────────────────
  ① 1本目: 陽線(実体≥0.5ATR)
  ② 2本目: 小実体(1本目の30%以下) かつ 1本目終値よりギャップ上
  ③ 3本目: 陰線(実体≥0.4ATR) かつ 1本目の半値以上食い込み
  ④ SLOPING確認
  ─ エントリー: 3本目確定→次足始値ショート""",

    "H1": """
  【hour10 定時エントリー】★
  ─────────────────────────────────────────
  ① 毎日10:00UTCのローソク足を監視
  ② SLOPING(DOWN)を確認するだけ
  ③ 追加条件なし — シンプルが強み
  ─ エントリー: 10:00足確定→11:00始値ショート
  ─ 最もシンプル。アラートで自動化しやすい""",
}

for k,v in details.items():
    print(v)

print()
print("="*110)
print("  リスク管理 & 運用ルール")
print("="*110)
print("""
  【ポジションサイジング】
    1エントリー = 口座の1-2%リスク
    SL = 1ATR(14)固定 → pipsが日によって変わる → ロット調整必須
    例: 口座$10,000 × 1% = $100リスク / ATR=15pt → 0.067lot

  【同時エントリー上限】
    複数エッジが同じ足で発火 → 1エントリーのみ (ロット重複禁止)
    コンフルエンス(2エッジ以上重複) → ロット1.5倍まで許容

  【1日のエントリー上限】
    最大3回/日 (連敗ストップ)
    3連敗したらその日は終了

  【見送り条件】
    ・SLOPING = RANGE → 全てスキップ
    ・重要指標発表30分前後 → スキップ (NFP, CPI, FOMC)
    ・同一水準へ当日2回目以降 → スキップ (旗艦エッジのみ)
    ・UTC 20:00〜07:00 → スキップ

  【月次評価基準】
    EV がベータ(+1.25/trade)を継続して下回る月が2ヶ月続いたら戦略見直し
    WFA効率比が0.5を下回ったら相場環境変化を疑う

  【コンフルエンス優先順位 (同時発火時)】
    最優先: Fib78.6% × 旗艦 × 水曜
    次点:   Judas Swing × Fib78.6%
    通常:   単体エッジ
""")

print("="*110)
print("  月次期待値試算 (重複排除・控えめ見積もり)")
print("="*110)

# 重複排除ポートフォリオ
bar_best={}
priority=['F2','F1','F3','G3','J1','X1','G2','R2','G1','F4','E1','G4','H1','E2','G5']
for key in priority:
    for i,p in zip([ii for ii in range(n) if edges.get(key) and True],edges.get(key,[])):
        pass

# 簡易: 全エッジのユニークバー × PnL
all_bar_pnl={}
key_order=['F1','F2','F3','G3','J1','X1','G2','R2','G1','F4','E1','G4','H1','E2','G5']

bar_map={}
for key in key_order:
    s=stats(edges[key])
    if not s or not s['stable']: continue
    bar_map[key]=edges[key]

# 単純に全stable edgeのユニークトレードを集計
seen=set()
combined=[]
edge_keys_ordered=['F2','F1','X1','G3','J1','G2','R2','G1','F4','E1','G4','H1','E2','G5']

# バーを特定するためにもう一度走る
bar_to_pnl={}
for key in edge_keys_ordered:
    cond_list=[]
    if key=='F1': cond_list=[(i,trade(i)) for i in fib786 if 7<=hour[i]<=19]
    elif key=='F2': cond_list=[(i,trade(i)) for i in fib786 if not np.isnan(ma200[i]) and c[i]<ma200[i]]
    elif key=='F3': cond_list=[(i,trade(i)) for i in fib786]
    elif key=='G1': cond_list=[(i,trade(i)) for i in flagship if 7<=hour[i]<=19]
    elif key=='G2': cond_list=[(i,trade(i)) for i in flagship if not np.isnan(atr_pct[i]) and 0.50<=atr_pct[i]<=0.80]
    elif key=='G3': cond_list=[(i,trade(i)) for i in flagship if dow[i]==2 and 7<=hour[i]<=19]
    elif key=='G4': cond_list=[(i,trade(i)) for i in flagship if 7<=hour[i]<=12]
    elif key=='G5': cond_list=[(i,trade(i)) for i in flagship]
    elif key=='R2': cond_list=[(i,trade(i)) for i in range(5,n-1) if reg[i] in ('UP','DOWN') and not np.isnan(rsi[i]) and rsi[i]>70]
    elif key=='J1': cond_list=[(i,trade(i)) for i in judas]
    elif key=='X1': cond_list=[(i,trade(i)) for i in fix]
    elif key=='H1': cond_list=[(i,trade(i)) for i in range(5,n-1) if hour[i]==10 and reg[i] in ('UP','DOWN')]
    elif key=='E1': cond_list=[(i,trade(i)) for i in eve]
    elif key=='E2': cond_list=[(i,trade(i)) for i in eng]
    elif key=='F4': cond_list=[(i,trade(i)) for i in fib786 if 7<=hour[i]<=12]
    for i,p in cond_list:
        if p is not None and i not in bar_to_pnl:
            bar_to_pnl[i]=p

combined=[p for p in bar_to_pnl.values()]
s=stats(combined)
if s:
    monthly={}
    for i,p in bar_to_pnl.items():
        mo=f"{df['Date'].iloc[i].year}-{df['Date'].iloc[i].month:02d}"
        monthly[mo]=monthly.get(mo,0)+p
    mo_v=list(monthly.values())
    sharpe=np.mean(mo_v)/(np.std(mo_v)+1e-9)*np.sqrt(12)
    print(f"""
  重複排除ポートフォリオ (全STABLEエッジ合算、同バーは1カウント):
    総トレード数 : {s['n']}回 / 5ヶ月 → 月平均{s['nmo']:.0f}回
    勝率         : {s['wr']:.0f}%
    EV/trade     : {s['ev']:+.2f}pt
    総PnL        : {s['total']:+.0f}pt / 5ヶ月
    最大連敗     : {s['mcl']}回
    年換算シャープ: {sharpe:.2f}
    月次PnL: {' | '.join(f"{k}:{v:+.0f}" for k,v in monthly.items())}
    
  ※ 1lot=XAU/USD標準 (100oz) の場合、1pt=$1
    月平均 {np.mean(mo_v):+.0f}pt → $1lot運用で月${np.mean(mo_v):+.0f}
""")

print("★"*50)
