"""
④ ポートフォリオシミュレーション
全STABLEエッジを同時運用したとき:
- 月次PnL / 累積曲線
- 最大DD / シャープ比
- エッジ間の重複(同日同時エントリー)処理
"""
import pandas as pd, numpy as np, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

df=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
df=df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
n=len(df); dates=df['Date'].values
o=df['o'].values; h=df['h'].values; l=df['l'].values
c=df['c'].values; v=df['v'].values
hour=df['Date'].dt.hour.values; dow=df['Date'].dt.dayofweek.values

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

sh1=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh1[i]=True
shi1=np.where(sh1)[0]
sh2=np.zeros(n,bool); sl2=np.zeros(n,bool)
for i in range(2,n-2):
    if all(h[i]>=h[i-j] for j in range(1,3)) and all(h[i]>=h[i+j] for j in range(1,3)): sh2[i]=True
    if all(l[i]<=l[i-j] for j in range(1,3)) and all(l[i]<=l[i+j] for j in range(1,3)): sl2[i]=True
shi2=np.where(sh2)[0]; sli2=np.where(sl2)[0]
atr_pct=np.full(n,np.nan)
for i in range(200,n):
    w=at[i-200:i]; w=w[~np.isnan(w)]
    if len(w)>0: atr_pct[i]=np.sum(w<=at[i])/len(w)

flagship=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]; psh=shi1[shi1<i-1]
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

def trade_result(i, direction='short'):
    if i+1>=n: return None
    e=o[i+1]; r=at[i]
    if np.isnan(r) or r<=0: return None
    tp=e-r if direction=='short' else e+r
    sl=e+r if direction=='short' else e-r
    for k in range(i+1,min(n,i+25)):
        if direction=='short':
            if h[k]>=sl: return -r-SPREAD
            if l[k]<=tp: return  r-SPREAD
        else:
            if l[k]<=sl: return -r-SPREAD
            if h[k]>=tp: return  r-SPREAD
    return None

# ── 全エッジ定義 (WFA通過済み or STABLE) ──────────────────
EDGES = {
    "Fib78.6%+SLOPING":      lambda i: i in f786,
    "Fib78.6%+UTC07-19":     lambda i: i in f786 and 7<=hour[i]<=19,
    "Fib78.6%+MA200下":      lambda i: i in f786 and not np.isnan(ma200[i]) and c[i]<ma200[i],
    "RSI>70+SLOPING":        lambda i: reg[i] in ('UP','DOWN') and not np.isnan(rsi[i]) and rsi[i]>70,
    "旗艦×ATR中ボラ":         lambda i: i in fl and not np.isnan(atr_pct[i]) and 0.50<=atr_pct[i]<=0.80,
    "旗艦+UTC07-19":         lambda i: i in fl and 7<=hour[i]<=19,
    "水曜+旗艦+UTC07-19":    lambda i: i in fl and dow[i]==2 and 7<=hour[i]<=19,
    "水曜+SLOPING":           lambda i: dow[i]==2 and reg[i] in ('UP','DOWN'),
    "hour10+SLOPING":        lambda i: hour[i]==10 and reg[i] in ('UP','DOWN'),
}

# ── シミュレーション ──────────────────────────────────────────
# 各バーでどのエッジが発火したか記録 → トレードリスト生成
# 同じバーに複数エッジ → 1トレードとしてカウント (重複排除)
# 別バー → 別トレード

# 戦略A: 各エッジ独立 (全部別ポジション、EV単純合計)
# 戦略B: 重複排除 (同バーは1エントリー、EV1回のみ)
# 戦略C: コンフルエンス重み付け (同バーに複数エッジ=ロット増し)

print("="*90)
print("# ポートフォリオシミュレーション")
print("="*90)

# 全エッジの発火バーとPnLを収集
all_trades = []  # (bar, edge_name, pnl)
for name, cond in EDGES.items():
    for i in range(5,n-1):
        if cond(i):
            p=trade_result(i)
            if p is not None:
                all_trades.append((i, name, p))

all_trades.sort(key=lambda x: x[0])

# 戦略A: 独立運用
print("\n### 戦略A: 各エッジ独立運用 (全ポジション合計) ###")
trades_a = [(bar, pnl) for bar,_,pnl in all_trades]
pnl_by_bar_a = {}
for bar,pnl in trades_a:
    pnl_by_bar_a[bar] = pnl_by_bar_a.get(bar,0)+pnl

# 戦略B: 重複排除 (同バーは1エントリーのみ)
print("### 戦略B: 重複排除 (同バー=1エントリー) ###")
seen_bars = {}
for bar,name,pnl in all_trades:
    if bar not in seen_bars:
        seen_bars[bar] = pnl  # 最初のエッジのみ

# 戦略C: コンフルエンス (同バーのエッジ数に比例)
print("### 戦略C: コンフルエンス重み付け ###")
bar_edges = {}
for bar,name,pnl in all_trades:
    if bar not in bar_edges:
        bar_edges[bar] = {'pnl':pnl, 'count':1, 'names':[name]}
    else:
        bar_edges[bar]['count']+=1
        bar_edges[bar]['names'].append(name)

# 月次集計関数
def monthly_stats(pnl_dict, label):
    if not pnl_dict: return
    bars=sorted(pnl_dict.keys())
    pnls=[pnl_dict[b] for b in bars]
    bar_dates=[pd.Timestamp(dates[b]) for b in bars]

    total=sum(pnls); n_=len(pnls)
    wr=100*sum(1 for p in pnls if p>0)/n_
    ev=np.mean(pnls)

    # 累積曲線
    cum=np.cumsum(pnls)
    max_dd=0; peak=0
    for x in cum:
        if x>peak: peak=x
        dd=peak-x
        if dd>max_dd: max_dd=dd

    # 月次
    monthly={}
    for d,p in zip(bar_dates,pnls):
        key=f"{d.year}-{d.month:02d}"
        monthly[key]=monthly.get(key,0)+p
    mo_vals=list(monthly.values())
    sharpe=np.mean(mo_vals)/(np.std(mo_vals)+1e-9)*np.sqrt(12) if len(mo_vals)>1 else 0

    print(f"\n  [{label}]")
    print(f"  総トレード={n_} 総PnL={total:+.0f}pt WR={wr:.0f}% EV/trade={ev:+.2f}")
    print(f"  最大DD={max_dd:.0f}pt  年換算シャープ={sharpe:.2f}")
    print(f"  月次: {' | '.join(f'{k}:{v:+.0f}' for k,v in monthly.items())}")
    return bars, cum, monthly

fig,axes=plt.subplots(3,1,figsize=(18,16))
fig.patch.set_facecolor('#0d1117')
colors=['#00e676','#29b6f6','#ffab40']

for ax,(strat_dict, label, col) in zip(axes,[
    (pnl_by_bar_a, "A: 独立運用 (全エッジ合計)", colors[0]),
    (seen_bars,    "B: 重複排除 (同バー1エントリー)", colors[1]),
    ({b:d['pnl']*d['count'] for b,d in bar_edges.items()},
                   "C: コンフルエンス重み付け", colors[2]),
]):
    res=monthly_stats(strat_dict, label)
    if res is None: continue
    bars,cum,monthly=res

    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#aaa',labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#333')

    ax.plot(range(len(cum)), cum, color=col, lw=1.5, label=label)
    ax.fill_between(range(len(cum)), 0, cum,
                    where=[x>0 for x in cum], alpha=0.15, color=col)
    ax.fill_between(range(len(cum)), 0, cum,
                    where=[x<=0 for x in cum], alpha=0.15, color='#ef5350')
    ax.axhline(0, color='#555', lw=0.8, ls='--')

    # 月境界線
    prev_mo=None
    for j,(b) in enumerate(bars):
        mo=pd.Timestamp(dates[b]).strftime('%Y-%m')
        if mo!=prev_mo:
            ax.axvline(j,color='#444',lw=0.5,ls=':')
            ax.text(j+1,ax.get_ylim()[0] if j>0 else 0,mo,color='#666',fontsize=7,va='bottom')
            prev_mo=mo

    peak=max(cum); trough=min(cum)
    ax.set_title(label, color='white', fontsize=10, pad=6)
    ax.set_ylabel('累積PnL (pt)', color='#aaa')
    ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white')

# コンフルエンス分析
print("\n### コンフルエンス分析 (複数エッジが重なったバー) ###")
for cnt in range(1,6):
    bars_c=[b for b,d in bar_edges.items() if d['count']==cnt]
    pnls_c=[bar_edges[b]['pnl'] for b in bars_c]
    if not pnls_c: continue
    wr=100*sum(1 for p in pnls_c if p>0)/len(pnls_c)
    ev=np.mean(pnls_c)
    h1p=pnls_c[:len(pnls_c)//2]; h2p=pnls_c[len(pnls_c)//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    ok='✅' if e1>0 and e2>0 else '❌'
    print(f"  エッジ数={cnt}: WR={wr:.0f}% EV={ev:+.2f} n={len(pnls_c)} [{e1:+.1f}/{e2:+.1f}] {ok}")

plt.suptitle('ポートフォリオ シミュレーション — 全STABLEエッジ同時運用\n'
             'A=独立 / B=重複排除 / C=コンフルエンス重み付け',
             color='white', fontsize=12)
plt.tight_layout(h_pad=3)
plt.savefig('portfolio_chart.png', dpi=130, bbox_inches='tight', facecolor='#0d1117')
print("\nChart saved: portfolio_chart.png")
plt.close()
