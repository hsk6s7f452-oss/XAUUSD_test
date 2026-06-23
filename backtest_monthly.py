"""
リアルタイム想定バックテスト: 2026年3月〜 | 0.05ロット固定
XAUUSD H1 | スプレッド0.3pt | SL=1ATR | TP=1ATR (1:1)
重複排除 (同バー複数エッジ→最優先1本のみ)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3
LOT=0.05        # ロット数
OZ_PER_LOT=100  # 1lot = 100oz (XAU標準)
USD_PER_PT=LOT*OZ_PER_LOT  # 1pt あたりの損益 ($)

df=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
df=df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)

# 3月以降のみ
df=df[df['Date']>='2026-03-01'].reset_index(drop=True)
n=len(df)
o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
dates=df['Date'].values
hour=df['Date'].dt.hour.values
dow=df['Date'].dt.dayofweek.values
day=df['Date'].dt.date.values
month=df['Date'].dt.to_period('M').values

# ─── インジケーター ────────────────────────────────────────────
# 注: 3月以降でスライスしたのでMA等は最初はNaN多め
# → H1全期間を使ってMA計算してからスライスする方がリアル

df_all=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df_all['Date']=pd.to_datetime(df_all['Date'],format='%Y.%m.%d %H:%M')
df_all=df_all.sort_values('Date').reset_index(drop=True)
df_all.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
na=len(df_all)
oa=df_all['o'].values; ha=df_all['h'].values; la=df_all['l'].values; ca=df_all['c'].values

tr_all=np.maximum(ha[1:]-la[1:],np.maximum(np.abs(ha[1:]-ca[:-1]),np.abs(la[1:]-ca[:-1])))
at_all=np.full(na,np.nan); at_all[1:]=pd.Series(tr_all).rolling(14).mean().values
ma200_all=pd.Series(ca).rolling(200).mean().values
ma50_all =pd.Series(ca).rolling(50).mean().values

slope_k=50
reg_all=np.array(['NA']*na,dtype=object)
for i in range(na):
    if i<200+slope_k or np.isnan(ma200_all[i]) or np.isnan(ma200_all[i-slope_k]): continue
    s=(ma200_all[i]-ma200_all[i-slope_k])/ma200_all[i-slope_k]
    if   s> 0.003 and ca[i]>ma200_all[i]: reg_all[i]='UP'
    elif s<-0.003 and ca[i]<ma200_all[i]: reg_all[i]='DOWN'
    else: reg_all[i]='RANGE'

def calc_rsi(c,p=14):
    na_=len(c)
    delta=np.diff(c); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=pd.Series(gain).ewm(alpha=1/p,adjust=False).mean().values
    al=pd.Series(loss).ewm(alpha=1/p,adjust=False).mean().values
    rsi=np.full(na_,np.nan); rsi[1:]=100-100/(1+ag/(al+1e-9)); return rsi
rsi_all=calc_rsi(ca)

atr_pct_all=np.full(na,np.nan)
for i in range(200,na):
    w=at_all[i-200:i]; w=w[~np.isnan(w)]
    if len(w)>0: atr_pct_all[i]=np.sum(w<=at_all[i])/len(w)

hour_all=df_all['Date'].dt.hour.values
dow_all=df_all['Date'].dt.dayofweek.values
day_all=df_all['Date'].dt.date.values

sh1_all=np.zeros(na,bool); sl1_all=np.zeros(na,bool)
for i in range(1,na-1):
    if ha[i]==ha[i-1:i+2].max() and ha[i]>ha[i-1] and ha[i]>ha[i+1]: sh1_all[i]=True
    if la[i]==la[i-1:i+2].min() and la[i]<la[i-1] and la[i]<la[i+1]: sl1_all[i]=True
shi_all=np.where(sh1_all)[0]

sh2_all=np.zeros(na,bool); sl2_all=np.zeros(na,bool)
for i in range(2,na-2):
    if all(ha[i]>=ha[i-j] for j in range(1,3)) and all(ha[i]>=ha[i+j] for j in range(1,3)): sh2_all[i]=True
    if all(la[i]<=la[i-j] for j in range(1,3)) and all(la[i]<=la[i+j] for j in range(1,3)): sl2_all[i]=True
shi2_all=np.where(sh2_all)[0]; sli2_all=np.where(sl2_all)[0]

# 3月以降のインデックスを全期間配列で特定
start_idx=df_all[df_all['Date']>='2026-03-01'].index[0]

# ─── エッジシグナル生成 (全期間で走らせて3月以降の結果だけ使う) ─────

def trade_from_alldf(i_all, direction='short', rr=1.0):
    """全期間配列でのインデックスからトレード評価"""
    if i_all+1>=na: return None, None
    e=oa[i_all+1]; r=at_all[i_all]
    if np.isnan(r) or r<=0: return None, None
    tp=e-r*rr if direction=='short' else e+r*rr
    sl=e+r    if direction=='short' else e-r
    for k in range(i_all+1, min(na,i_all+25)):
        if direction=='short':
            if ha[k]>=sl: return -r-SPREAD, i_all
            if la[k]<=tp: return  r*rr-SPREAD, i_all
        else:
            if la[k]<=sl: return -r-SPREAD, i_all
            if ha[k]>=tp: return  r*rr-SPREAD, i_all
    return None, None

signals={}  # {i_all: (edge_name, pnl_pt)}

# 優先度順 (高優先が先)
priority_names=[
    'Fib78.6+MA200下','Fib78.6+UTC07-19','Fib78.6+SLOPING',
    '水曜+旗艦+UTC07-19','JudasSwing','ロンドンFix',
    '旗艦×ATR中ボラ','RSI>70','旗艦+UTC07-19',
    'Fib78.6+London','イブニングスター','旗艦+London',
    'hour10','包み足','旗艦(基本)'
]

def add_signal(i_all, name):
    if i_all < start_idx: return  # 3月以前は無視
    if i_all in signals: return   # 既に高優先エッジが登録済み
    p,_=trade_from_alldf(i_all)
    if p is not None:
        signals[i_all]=(name, p, at_all[i_all])

# ── Fib78.6% ────────────────────────────────────────────────
for i in range(10,na):
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    psh=shi2_all[shi2_all<i]; psl=sli2_all[sli2_all<i]
    if not len(psh) or not len(psl): continue
    lsh=psh[-1]; lsl=psl[-1]
    if lsh<=lsl: continue
    top=ha[lsh]; bot=la[lsl]
    if top<=bot: continue
    if abs((ca[i]-bot)/(top-bot)-0.786)<=0.06 and reg_all[i] in ('UP','DOWN'):
        # F2: MA200下
        if not np.isnan(ma200_all[i]) and ca[i]<ma200_all[i]:
            add_signal(i,'Fib78.6+MA200下')
        # F1: UTC07-19
        elif 7<=hour_all[i]<=19:
            add_signal(i,'Fib78.6+UTC07-19')
        # F3: SLOPING
        else:
            add_signal(i,'Fib78.6+SLOPING')

# ── 旗艦 ────────────────────────────────────────────────────
flagship_all=[]
for i in range(5,na):
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    tol=0.20*at_all[i]; psh=shi_all[shi_all<i-1]
    if not len(psh): continue
    res=ha[psh]; res=res[res>ca[i-1]]
    if not len(res): continue
    lv=res.min()
    if abs(ha[i]-lv)<=tol or (ha[i]>lv-tol and ha[i]<lv+tol):
        if ca[i]<lv and reg_all[i] in ('UP','DOWN') and not np.isnan(ma50_all[i]) and ca[i]>ma50_all[i]:
            flagship_all.append(i)

for i in flagship_all:
    # G3: 水曜+UTC07-19
    if dow_all[i]==2 and 7<=hour_all[i]<=19:
        add_signal(i,'水曜+旗艦+UTC07-19')
    # G2: ATR中ボラ
    elif not np.isnan(atr_pct_all[i]) and 0.50<=atr_pct_all[i]<=0.80:
        add_signal(i,'旗艦×ATR中ボラ')
    # G1: UTC07-19
    elif 7<=hour_all[i]<=19:
        add_signal(i,'旗艦+UTC07-19')
    # G4: London
    elif 7<=hour_all[i]<=12:
        add_signal(i,'旗艦+London')
    # G5: 基本
    else:
        add_signal(i,'旗艦(基本)')

# ── Judas Swing ──────────────────────────────────────────────
london_open_all={}; london_2h_all={}
for i in range(na):
    if hour_all[i]==7: london_open_all[day_all[i]]=(i,oa[i])
    if hour_all[i]==9: london_2h_all[day_all[i]]=(i,ca[i])

for i in range(na):
    if hour_all[i]!=10: continue
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    d=day_all[i]
    if d not in london_open_all or d not in london_2h_all: continue
    _,lo=london_open_all[d]; _,l2=london_2h_all[d]
    if l2-lo>at_all[london_open_all[d][0]]*0.3 and reg_all[i] in ('UP','DOWN'):
        add_signal(i,'JudasSwing')

# ── ロンドンフィックス ────────────────────────────────────────
for i in range(na-1):
    if hour_all[i]!=16: continue
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    d=day_all[i]
    pre=[j for j in range(max(0,i-3),i) if hour_all[j]==15 and day_all[j]==d]
    if not pre: continue
    if ca[i]-oa[pre[0]]>at_all[i]*0.5 and reg_all[i] in ('UP','DOWN'):
        add_signal(i,'ロンドンFix')

# ── RSI>70 ───────────────────────────────────────────────────
for i in range(21,na):
    if np.isnan(rsi_all[i]) or np.isnan(rsi_all[i-1]): continue
    if rsi_all[i]>70 and rsi_all[i-1]<=70 and reg_all[i] in ('UP','DOWN'):
        add_signal(i,'RSI>70')

# ── hour10 ───────────────────────────────────────────────────
for i in range(5,na-1):
    if hour_all[i]==10 and reg_all[i] in ('UP','DOWN'):
        add_signal(i,'hour10')

# ── イブニングスター ──────────────────────────────────────────
for i in range(2,na-1):
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    b=lambda j: abs(ca[j]-oa[j])
    if (ca[i-2]>oa[i-2] and b(i-2)>=at_all[i]*0.5 and
        b(i-1)<=b(i-2)*0.3 and ha[i-1]>ca[i-2] and
        ca[i]<oa[i] and b(i)>=at_all[i]*0.4 and ca[i]<=(oa[i-2]+ca[i-2])/2
        and reg_all[i] in ('UP','DOWN')):
        add_signal(i,'イブニングスター')

# ── 包み足 ───────────────────────────────────────────────────
for i in range(1,na-1):
    if np.isnan(at_all[i]) or at_all[i]<=0: continue
    if (ca[i]<oa[i] and ca[i-1]>oa[i-1] and oa[i]>=ca[i-1] and ca[i]<=oa[i-1]
        and abs(ca[i]-oa[i])>abs(ca[i-1]-oa[i-1])
        and reg_all[i] in ('UP','DOWN') and not np.isnan(ma50_all[i]) and ca[i]>ma50_all[i]):
        add_signal(i,'包み足')

# ─── トレードリスト (時系列順) ───────────────────────────────────
trades=sorted(signals.items())  # [(i_all, (name, pnl_pt, atr)), ...]

# 1日3回上限
daily_count={}
filtered_trades=[]
for i_all,(name,pnl_pt,atr) in trades:
    d=day_all[i_all]
    if daily_count.get(d,0)>=3: continue
    daily_count[d]=daily_count.get(d,0)+1
    filtered_trades.append((i_all,name,pnl_pt,atr,df_all['Date'].iloc[i_all]))

# ─── 月別集計 ────────────────────────────────────────────────────

print()
print("="*90)
print("  XAUUSD H1 リアルタイムバックテスト | 0.05ロット | 2026年3月〜")
print("  SL=1ATR(14) | TP=1ATR(14) 1:1 | スプレッド0.3pt | 最大3エントリー/日")
print("  1pt = $5 (0.05lot × 100oz/lot)")
print("="*90)

monthly_data={}
for i_all,name,pnl_pt,atr,dt in filtered_trades:
    mo=dt.strftime('%Y-%m')
    if mo not in monthly_data:
        monthly_data[mo]={'trades':[],'wins':0,'losses':0,'pnl_pt':0}
    monthly_data[mo]['trades'].append((dt,name,pnl_pt,atr))
    monthly_data[mo]['pnl_pt']+=pnl_pt
    if pnl_pt>0: monthly_data[mo]['wins']+=1
    else: monthly_data[mo]['losses']+=1

# ─── 月別詳細 ────────────────────────────────────────────────────
all_pnl_pt=[]
all_pnl_usd=[]
equity=0.0  # pt cumulative
peak_equity=0.0
max_dd_pt=0.0
total_wins=0; total_losses=0

print()
for mo in sorted(monthly_data.keys()):
    d=monthly_data[mo]
    n_tr=len(d['trades'])
    wr=100*d['wins']/n_tr if n_tr>0 else 0
    pnl_pt=d['pnl_pt']
    pnl_usd=pnl_pt*USD_PER_PT
    total_wins+=d['wins']; total_losses+=d['losses']
    all_pnl_pt.append(pnl_pt)
    all_pnl_usd.append(pnl_usd)
    equity+=pnl_pt
    if equity>peak_equity: peak_equity=equity
    dd=peak_equity-equity
    if dd>max_dd_pt: max_dd_pt=dd

    print(f"  ┌── {mo} ─────────────────────────────────────────────")
    print(f"  │  トレード数: {n_tr:3d}  勝:{d['wins']:3d}  負:{d['losses']:3d}  勝率:{wr:4.0f}%")
    print(f"  │  損益(pt) : {pnl_pt:+8.2f}pt    損益($) : ${pnl_usd:+8.2f}")
    print(f"  │  累積(pt) : {equity:+8.2f}pt    累積($) : ${equity*USD_PER_PT:+8.2f}")
    print(f"  │")
    print(f"  │  内訳 (上位エッジ):")

    # エッジ別集計
    edge_pnl={}
    for dt_,name_,pnl_,atr_ in d['trades']:
        if name_ not in edge_pnl: edge_pnl[name_]={'n':0,'pnl':0,'w':0}
        edge_pnl[name_]['n']+=1; edge_pnl[name_]['pnl']+=pnl_
        if pnl_>0: edge_pnl[name_]['w']+=1
    for ename in sorted(edge_pnl.keys(), key=lambda x: -edge_pnl[x]['pnl']):
        ep=edge_pnl[ename]
        ep_usd=ep['pnl']*USD_PER_PT
        ewr=100*ep['w']/ep['n'] if ep['n']>0 else 0
        print(f"  │    {ename:22s}: {ep['n']:3d}回  {ewr:3.0f}%  {ep['pnl']:+7.2f}pt  ${ep_usd:+7.2f}")

    print(f"  │")
    print(f"  │  全トレード:")
    print(f"  │  {'日時':22s} {'エッジ名':24s} {'PnL(pt)':>8} {'ATR':>7} {'PnL($)':>8}")
    print(f"  │  " + "-"*78)
    for dt_,name_,pnl_,atr_ in d['trades']:
        sign='✅' if pnl_>0 else '❌'
        print(f"  │  {sign} {str(dt_):22s} {name_:24s} {pnl_:+8.2f}  {atr_:6.2f}  ${pnl_*USD_PER_PT:+8.2f}")
    print(f"  └{'─'*85}")
    print()

# ─── 全期間サマリー ───────────────────────────────────────────────
all_results=[p for mo_d in monthly_data.values() for _,_,p,_ in mo_d['trades']]
total_n=len(all_results)
total_wr=100*total_wins/total_n if total_n>0 else 0
total_pnl_pt=sum(all_pnl_pt)
total_pnl_usd=total_pnl_pt*USD_PER_PT
avg_pnl_pt=np.mean(all_results) if all_results else 0
mo_list=all_pnl_pt

# 月次シャープ
sharpe=np.mean(mo_list)/(np.std(mo_list)+1e-9)*np.sqrt(12) if len(mo_list)>1 else 0

# 最大連敗
mcl=0; cur_l=0
for p in all_results:
    if p<0: cur_l+=1; mcl=max(mcl,cur_l)
    else: cur_l=0

# 最大DD in $
max_dd_usd=max_dd_pt*USD_PER_PT

# 最良/最悪月
best_mo=sorted(monthly_data.keys(), key=lambda x: monthly_data[x]['pnl_pt'])[-1]
worst_mo=sorted(monthly_data.keys(), key=lambda x: monthly_data[x]['pnl_pt'])[0]

n_mo=len(monthly_data)

print("="*90)
print("  ▶ 全期間サマリー (2026年3月〜)")
print("="*90)
print(f"""
  期間            : {n_mo}ヶ月 ({min(monthly_data.keys())} ～ {max(monthly_data.keys())})
  総トレード数    : {total_n}回 (月平均{total_n/n_mo:.0f}回)
  総勝ち/負け     : {total_wins}勝 / {total_losses}敗
  勝率            : {total_wr:.1f}%
  EV/trade(pt)    : {avg_pnl_pt:+.2f}pt
  EV/trade($)     : ${avg_pnl_pt*USD_PER_PT:+.2f}

  ── 損益 ──
  総損益(pt)      : {total_pnl_pt:+.2f}pt
  総損益($)       : ${total_pnl_usd:+.2f}
  月平均損益(pt)  : {np.mean(mo_list):+.2f}pt/月
  月平均損益($)   : ${np.mean(mo_list)*USD_PER_PT:+.2f}/月

  ── リスク ──
  最大DD(pt)      : {max_dd_pt:.2f}pt
  最大DD($)       : ${max_dd_usd:.2f}
  最大連敗        : {mcl}回
  年換算シャープ  : {sharpe:.2f}

  ── 月次実績 ──
  最良月          : {best_mo}  {monthly_data[best_mo]['pnl_pt']:+.2f}pt  ${monthly_data[best_mo]['pnl_pt']*USD_PER_PT:+.2f}
  最悪月          : {worst_mo}  {monthly_data[worst_mo]['pnl_pt']:+.2f}pt  ${monthly_data[worst_mo]['pnl_pt']*USD_PER_PT:+.2f}
  プラス月/全月   : {sum(1 for p in mo_list if p>0)}/{n_mo}ヶ月

  ── ロット換算 ──
  使用ロット      : {LOT}lot (1pt = ${USD_PER_PT:.0f})
  証拠金目安      : $2,000〜$5,000 (0.05lot, XAU/USD)
  月次リターン率  : 約 {np.mean(mo_list)*USD_PER_PT/3000*100:.1f}% (証拠金$3,000換算)
""")

print("="*90)
print("  ▶ 月次損益グラフ (pt)")
print("="*90)
max_bar=40
max_val=max(abs(p) for p in mo_list) if mo_list else 1
for mo,ppt in zip(sorted(monthly_data.keys()),mo_list):
    bar_len=int(abs(ppt)/max_val*max_bar)
    if ppt>=0:
        bar='█'*bar_len
        print(f"  {mo} | {bar:<40s} +{ppt:.1f}pt  ${ppt*USD_PER_PT:+.0f}")
    else:
        bar='░'*bar_len
        print(f"  {mo} | {'':40s}{'░'*bar_len} {ppt:.1f}pt  ${ppt*USD_PER_PT:+.0f}")

print()
print("  累積エクイティカーブ (pt):")
eq=0
for mo,ppt in zip(sorted(monthly_data.keys()),mo_list):
    eq+=ppt
    bar_len=int(eq/max(abs(sum(mo_list)),1)*max_bar)
    bar='▓'*max(0,bar_len)
    print(f"  {mo} | {bar:<40s} 累積:{eq:+.1f}pt  累積:${eq*USD_PER_PT:+.0f}")

print()
print("="*90)
print(f"  ※ SPREAD={SPREAD}pt込み | SL=TP=ATR(14) | 1日最大3エントリー")
print(f"  ※ 同一バー複数エッジは優先度順1本のみ | UTC20-07はノーエントリー")
print("="*90)
