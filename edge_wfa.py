"""
WALK-FORWARD ANALYSIS (WFA)
============================
目的: 半分割よりも厳密に「過去で確認 → 未来で検定」を繰り返し、
     エッジが時間的に安定しているか検証。

方式: Anchored WFA (起点固定) + Rolling WFA (窓をずらす)
  IS (In-Sample)  = 学習期間: エッジ確認
  OOS (Out-of-Sample) = 検定期間: 未来の未知データで評価

上位5エッジ:
  1. H1 スイングハイスイープ(3本) + RANGE除外
  2. 水平線(SH)拒否 + MA50上 + SLOPING
  3. H1 hour10(UTC) ショート + SLOPING
  4. H1 hour07(UTC) ショート + SLOPING
  5. 水平線(SH)拒否 + テスト≥2回 + SLOPING

指標:
  - 各OOS窓のWR/EV
  - WFA効率比 = mean(OOS_EV) / mean(IS_EV)  (1.0=完全移転、0=全劣化)
  - OOS勝率 = 何割の窓がEV>0か
  - t検定: OOS EVの平均 ≠ 0 か

1:1 ATR, SPREAD=0.3, no look-ahead.
"""
import pandas as pd, numpy as np, warnings
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

def swings_w(h,l,n,w=1):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

# ── load ──
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values
c=h1['c'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values
reg=calc_regime(c)
hr=h1['Date'].dt.hour.values
dates=h1['Date'].values

sh_arr,_=swings_w(h,l,n,w=1); shi=np.where(sh_arr)[0]

# touch count (needed for edge #5)
touch_count={}
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]
    past_sh=shi[shi<i-1]
    if not len(past_sh): continue
    res_levels=h[past_sh]; res_levels=res_levels[res_levels>c[i-1]]
    if not len(res_levels): continue
    level=res_levels.min(); key=round(level,1)
    if abs(h[i]-level)<=tol or (h[i]>level-tol and h[i]<level+tol):
        touch_count[key]=touch_count.get(key,0)+1

# ── build all signals upfront (bar index → pnl) ──
def short_trade(i, horizon=24):
    if i+1>=n or np.isnan(at[i]) or at[i]<=0: return None
    entry=o[i+1]; rng=at[i]; tp=entry-rng; slv=entry+rng
    for k in range(i+1,min(n,i+1+horizon)):
        if h[k]>=slv: return -rng-SPREAD
        if l[k]<=tp:  return  rng-SPREAD
    return None

# Edge 1: sweep + RANGE excluded
e1={}
for i in range(2,n):
    ps=shi[shi<i-1]
    if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]] and reg[i] in ('UP','DOWN'):
        p=short_trade(i)
        if p is not None: e1[i]=p

# Edge 2: horiz SH reject + MA50上 + SLOPING
e2={}
tc2={}
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]
    past_sh=shi[shi<i-1]
    if not len(past_sh): continue
    res_levels=h[past_sh]; res_levels=res_levels[res_levels>c[i-1]]
    if not len(res_levels): continue
    level=res_levels.min(); key=round(level,1)
    if abs(h[i]-level)<=tol or (h[i]>level-tol and h[i]<level+tol):
        tc2[key]=tc2.get(key,0)+1
        if c[i]<level and reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
            p=short_trade(i)
            if p is not None: e2[i]=p

# Edge 3: hour10 + SLOPING
e3={i:short_trade(i) for i in range(n)
    if hr[i]==10 and reg[i] in ('UP','DOWN') and short_trade(i) is not None}

# Edge 4: hour07 + SLOPING
e4={i:short_trade(i) for i in range(n)
    if hr[i]==7 and reg[i] in ('UP','DOWN') and short_trade(i) is not None}

# Edge 5: horiz SH reject + tested≥2x + SLOPING
e5={}
tc5={}
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]
    past_sh=shi[shi<i-1]
    if not len(past_sh): continue
    res_levels=h[past_sh]; res_levels=res_levels[res_levels>c[i-1]]
    if not len(res_levels): continue
    level=res_levels.min(); key=round(level,1)
    if abs(h[i]-level)<=tol or (h[i]>level-tol and h[i]<level+tol):
        tc5[key]=tc5.get(key,0)+1
        if c[i]<level and reg[i] in ('UP','DOWN') and tc5.get(key,0)>=2:
            p=short_trade(i)
            if p is not None: e5[i]=p

edges={
    'E1 Sweep(3bar)+RANGE除外'      : e1,
    'E2 水平線拒否+MA50上+SLOPING'  : e2,
    'E3 hour10+SLOPING'             : e3,
    'E4 hour07+SLOPING'             : e4,
    'E5 水平線拒否+tested≥2x+SLOPING': e5,
}

# ══════════════════════════════════════════════════════════
# WFA ENGINE
# ══════════════════════════════════════════════════════════
def stat(pnls):
    if len(pnls)<3: return None
    return dict(wr=100*np.mean([p>0 for p in pnls]),
                ev=np.mean(pnls), n=len(pnls), tot=sum(pnls))

def wfa(signals, label,
        is_bars=600, oos_bars=200, step=200, min_is=5, min_oos=3):
    """
    Anchored: IS always starts from 0, OOS window steps forward.
    Rolling:  both IS and OOS step forward together.
    """
    all_idx=sorted(signals.keys())
    results_anchored=[]; results_rolling=[]

    # convert bar indices to positions in date-space
    # OOS windows
    oos_starts=list(range(is_bars, n-oos_bars+1, step))

    for oos_start in oos_starts:
        oos_end=oos_start+oos_bars

        # Anchored: IS = [0, oos_start)
        is_pnls=[signals[i] for i in all_idx if i<oos_start]
        oos_pnls=[signals[i] for i in all_idx if oos_start<=i<oos_end]

        if len(is_pnls)>=min_is and len(oos_pnls)>=min_oos:
            results_anchored.append({
                'oos_start': oos_start,
                'is': stat(is_pnls),
                'oos': stat(oos_pnls),
                'date': pd.Timestamp(dates[oos_start]).strftime('%Y-%m-%d'),
            })

        # Rolling: IS = [oos_start-is_bars, oos_start)
        roll_is_start=max(0, oos_start-is_bars)
        is_pnls_r=[signals[i] for i in all_idx if roll_is_start<=i<oos_start]
        if len(is_pnls_r)>=min_is and len(oos_pnls)>=min_oos:
            results_rolling.append({
                'oos_start': oos_start,
                'is': stat(is_pnls_r),
                'oos': stat(oos_pnls),
                'date': pd.Timestamp(dates[oos_start]).strftime('%Y-%m-%d'),
            })

    return results_anchored, results_rolling

def print_wfa(label, results, kind):
    if not results:
        print(f"  {kind}: データ不足"); return
    oos_evs=[r['oos']['ev'] for r in results]
    oos_wrs=[r['oos']['wr'] for r in results]
    is_evs =[r['is']['ev']  for r in results]
    eff=np.mean(oos_evs)/np.mean(is_evs) if np.mean(is_evs)!=0 else 0
    win_rate=100*np.mean([e>0 for e in oos_evs])
    # t-test: OOS EV > 0
    if len(oos_evs)>=3:
        t,p=stats.ttest_1samp(oos_evs,0,alternative='greater')
    else:
        t,p=np.nan,np.nan

    print(f"\n  [{kind}] OOS窓数={len(results)}")
    print(f"  {'OOS窓':>12} {'IS_EV':>7} {'OOS_EV':>7} {'OOS_WR':>7} {'OOS_n':>6}")
    for r in results:
        oos=r['oos']; is_=r['is']
        flag="✅" if oos['ev']>0 else "❌"
        print(f"  {r['date']:>12} IS:{is_['ev']:+5.2f}  OOS:{oos['ev']:+5.2f}  "
              f"WR={oos['wr']:.0f}%  n={oos['n']:3d}  {flag}")

    print(f"  → OOS勝率(EV>0): {win_rate:.0f}%  効率比: {eff:.2f}  "
          f"t={t:.2f}  p={p:.3f}  ", end="")
    if p<0.05:  print("★統計的に有意(p<0.05)")
    elif p<0.10: print("△ほぼ有意(p<0.10)")
    else:        print("✗有意でない")

# ── Chart: OOS equity curves for each edge ──
fig,axes=plt.subplots(3,2,figsize=(22,16))
fig.patch.set_facecolor('#0d1117')
axes=axes.flatten()

print("="*80)
print("# WALK-FORWARD ANALYSIS — 上位5エッジ")
print("# IS=600bar(~25週), OOS=200bar(~8週), step=200bar")
print("="*80)

for ax_i,(label,signals) in enumerate(edges.items()):
    ax=axes[ax_i]
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#aaa',labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#333')

    print(f"\n{'─'*80}")
    print(f"## {label}  (全体n={len(signals)}, 全体EV={np.mean(list(signals.values())):+.2f})")
    print(f"{'─'*80}")

    anc,rol=wfa(signals, label)
    print_wfa(label, anc, "Anchored(起点固定)")
    print_wfa(label, rol, "Rolling(窓スライド)")

    # equity curve (cumulative PnL over time)
    all_sorted=sorted(signals.items())
    xs=[h1['Date'].iloc[i] for i,_ in all_sorted]
    ys=np.cumsum([p for _,p in all_sorted])
    ax.plot(xs,ys,color='#26a69a',lw=1.5,label='Cumulative PnL')
    ax.axhline(0,color='#555',lw=0.7,ls='--')

    # shade OOS windows
    is_bars=600; oos_bars=200; step=200
    colors=['#1a3a5c','#3a1a1a','#1a3a1a','#3a3a1a']
    for ci,oos_start in enumerate(range(is_bars,n-oos_bars+1,step)):
        oos_end=min(n-1,oos_start+oos_bars)
        d_s=h1['Date'].iloc[oos_start]; d_e=h1['Date'].iloc[oos_end]
        ax.axvspan(d_s,d_e,alpha=0.15,color=colors[ci%len(colors)],label='OOS窓' if ci==0 else '')

    ax.set_title(label,color='white',fontsize=9,pad=5)
    ax.set_ylabel('Cumulative PnL (pt)',color='#aaa',fontsize=8)
    ax.legend(fontsize=7,facecolor='#1a1a2e',labelcolor='white')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

# last panel: summary bar chart of OOS win rates
ax=axes[5]
ax.set_facecolor('#0d1117')
ax.tick_params(colors='#aaa',labelsize=8)
for sp in ax.spines.values(): sp.set_color('#333')

labels_short=['E1 Sweep','E2 水平+MA50','E3 hr10','E4 hr07','E5 水平tested']
oos_win_rates=[]
oos_eff_ratios=[]
for label,signals in edges.items():
    anc,_=wfa(signals,label)
    if anc:
        evs=[r['oos']['ev'] for r in anc]
        is_evs=[r['is']['ev'] for r in anc]
        oos_win_rates.append(100*np.mean([e>0 for e in evs]))
        eff=np.mean(evs)/np.mean(is_evs) if np.mean(is_evs)!=0 else 0
        oos_eff_ratios.append(eff)
    else:
        oos_win_rates.append(0); oos_eff_ratios.append(0)

x=np.arange(len(labels_short))
bars=ax.bar(x-0.2, oos_win_rates, 0.35,
            color=['#26a69a' if v>=60 else '#ef5350' if v<40 else '#ff9800' for v in oos_win_rates],
            label='OOS窓勝率(%)',alpha=0.85)
ax.bar(x+0.2, [e*50+50 for e in oos_eff_ratios], 0.35,
       color='#90caf9',alpha=0.6,label='効率比×50+50')
ax.axhline(50,color='#555',lw=0.8,ls='--')
ax.axhline(60,color='#26a69a',lw=0.8,ls=':',alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels_short,fontsize=7,color='#aaa')
ax.set_title('WFA Summary: OOS窓勝率 & 効率比',color='white',fontsize=9)
ax.legend(fontsize=7,facecolor='#1a1a2e',labelcolor='white')
for bar,v in zip(bars,oos_win_rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{v:.0f}%',ha='center',va='bottom',color='white',fontsize=8)

plt.suptitle('Walk-Forward Analysis — XAUUSD H1 上位5エッジ\n'
             '薄色シェード=OOS(未来)検定窓  折れ線=累積PnL',
             color='white',fontsize=11,y=1.01)
plt.tight_layout(h_pad=2.5,w_pad=2)
plt.savefig('wfa_chart.png',dpi=120,bbox_inches='tight',facecolor='#0d1117')
print("\n\nChart saved: wfa_chart.png")
plt.close()

print("\n" + "="*80)
print("# WFA 判定基準")
print("# OOS勝率≥60% & 効率比≥0.5 & p<0.10 → 実戦レベル")
print("# OOS勝率≥50% & 効率比≥0.3        → フォワード継続推奨")
print("# OOS勝率<50% or 効率比<0.0        → 過剰最適化疑い、棄却")
print("="*80)
