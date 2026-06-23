"""
HONEST CHART — show ONLY the filtered setup (SH resist reject + MA50上 + SLOPING)
Color-code: GREEN arrow = winner, RED arrow = loser, after-the-fact.
This strips away visual bias and shows exactly where it worked and didn't.
"""
import pandas as pd, numpy as np, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def swings(h,l,n,w=1):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values
c=h1['c'].values; v=h1['v'].values
at=calc_atr(h,l,c)
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values
reg=calc_regime(c)

sh_arr,sl_arr=swings(h,l,n,w=1)
shi=np.where(sh_arr)[0]

# ── rebuild the filtered signals + trade outcomes ──
TOL=0.20
trades=[]  # (bar_idx, level, win_bool, pnl)

touch_count={}
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=TOL*at[i]
    past_sh=shi[shi<i-1]
    if not len(past_sh): continue
    res_levels=h[past_sh]
    res_levels=res_levels[res_levels>c[i-1]]
    if not len(res_levels): continue
    level=res_levels.min()
    key=round(level,1)
    if abs(h[i]-level)<=tol or (h[i]>level-tol and h[i]<level+tol):
        cnt=touch_count.get(key,0)
        touch_count[key]=cnt+1
        if c[i]<level:  # rejected
            # apply filters: SLOPING + MA50上
            if reg[i] not in ('UP','DOWN'): continue
            if np.isnan(ma50[i]) or c[i]<=ma50[i]: continue
            # trade
            if i+1>=n: continue
            entry=o[i+1]; rng=at[i]; tp=entry-rng; slv=entry+rng; win=None; pnl=None
            for k in range(i+1,min(n,i+1+24)):
                if h[k]>=slv: win=False; pnl=-rng-SPREAD; break
                if l[k]<=tp:  win=True;  pnl= rng-SPREAD; break
            if win is None: continue
            trades.append((i,level,win,pnl))

wins  =[t for t in trades if t[2]]
losses=[t for t in trades if not t[2]]
print(f"Total trades: {len(trades)}  Wins: {len(wins)} ({100*len(wins)/len(trades):.0f}%)  Losses: {len(losses)}")
print(f"EV: {np.mean([t[3] for t in trades]):+.2f}  Total: {sum(t[3] for t in trades):+.0f}")

# ── print loser locations so user can see pattern ──
print("\nLOSS locations (month, hour, regime):")
for i,lv,w,p in losses:
    print(f"  bar={i:4d} {h1['Date'].iloc[i].strftime('%Y-%m-%d %H:%M')} "
          f"reg={reg[i]} MA200={'above' if c[i]>ma200[i] else 'below'} "
          f"hour={h1['Date'].iloc[i].hour:02d} price={c[i]:.0f} vs MA50={ma50[i]:.0f}")

# ── draw one chart per ~2-month window so arrows are readable ──
h1['Date_dt']=h1['Date']
windows=[
    ('Feb–Mar 2026', 0,       n//2),
    ('Apr–Jun 2026', n//2,    n),
]

fig, axes = plt.subplots(2,1,figsize=(26,20))
fig.patch.set_facecolor('#0d1117')

for ax,(title,wstart,wend) in zip(axes,windows):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#aaa',labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x)}"))

    idx=np.arange(wstart,wend)
    for j,x in enumerate(idx):
        col='#26a69a' if c[x]>=o[x] else '#ef5350'
        ax.plot([j,j],[l[x],h[x]],color=col,lw=0.5,alpha=0.7)
        ax.add_patch(plt.Rectangle((j-0.3,min(o[x],c[x])),0.6,abs(c[x]-o[x]),
                     color=col,alpha=0.85,zorder=2))

    # MAs
    ax.plot(range(len(idx)),ma200[wstart:wend],color='#ff9800',lw=1.3,label='MA200',alpha=0.9)
    ax.plot(range(len(idx)),ma50[wstart:wend], color='#90caf9',lw=1.0,label='MA50', alpha=0.8)

    # swing-high resistance lines (dashed, faint) — only from SH within the window or just before
    sh_visible=shi[(shi>=max(0,wstart-100))&(shi<wend)]
    for j in sh_visible:
        lv=h[j]; xj=max(0,j-wstart)
        ax.plot([xj,len(idx)-1],[lv,lv],color='#ef5350',lw=0.5,ls='--',alpha=0.25)

    # MA50 fill (premium zone)
    x_range=range(len(idx))
    ax.fill_between(x_range, ma50[wstart:wend],
                    np.maximum(ma50[wstart:wend], h[wstart:wend].max()),
                    alpha=0.04,color='#ef5350',label='Premium zone (above MA50)')

    # trade signals — WIN=green arrow, LOSS=red X
    for i,lv,win,pnl in trades:
        if i<wstart or i>=wend: continue
        xj=i-wstart
        entry_price=o[i+1] if i+1<n else c[i]
        rng=at[i]
        tp_price=entry_price-rng
        sl_price=entry_price+rng
        if win:
            # green down-arrow at rejection high, green TP line
            ax.annotate('',xy=(xj,h[i]+rng*0.15),xytext=(xj,h[i]+rng*0.7),
                        arrowprops=dict(arrowstyle='-|>',color='#00e676',lw=2.0))
            ax.plot([xj,xj+6],[tp_price,tp_price],color='#00e676',lw=1.2,ls=':',alpha=0.7)
        else:
            # red X at rejection high, red SL line
            ax.plot(xj, h[i]+rng*0.3, 'x', color='#ff1744', ms=10, mew=2.5, zorder=5)
            ax.plot([xj,xj+6],[sl_price,sl_price],color='#ff1744',lw=1.2,ls=':',alpha=0.7)

        # shade the TP/SL zone lightly
        color='#00e676' if win else '#ff1744'
        ax.add_patch(plt.Rectangle((xj,tp_price),8,sl_price-tp_price,
                     color=color,alpha=0.06,zorder=1))

    # x labels
    xt=[]; xl=[]
    for j,x in enumerate(idx):
        if h1['Date'].iloc[x].day in (1,8,15,22) and h1['Date'].iloc[x].hour==0:
            xt.append(j); xl.append(h1['Date'].iloc[x].strftime('%b %d'))
    ax.set_xticks(xt[::max(1,len(xt)//15)])
    ax.set_xticklabels(xl[::max(1,len(xl)//15)],fontsize=8,color='#aaa')

    w_cnt=sum(1 for i,_,win,_ in trades if win  and wstart<=i<wend)
    l_cnt=sum(1 for i,_,win,_ in trades if not win and wstart<=i<wend)
    tot_pnl=sum(p for i,_,_,p in trades if wstart<=i<wend)
    wr_h=100*w_cnt/(w_cnt+l_cnt) if (w_cnt+l_cnt)>0 else 0
    ax.set_title(f"{title}  |  WR={wr_h:.0f}%  Wins={w_cnt} (↓green)  Losses={l_cnt} (✕red)  PnL={tot_pnl:+.0f}pt  "
                 f"Filter: SwingHigh reject + MA50上 + SLOPING",
                 color='white',fontsize=10,pad=8)

    patches=[
        mpatches.Patch(color='#00e676',label='WIN (↓green arrow + dotted TP)'),
        mpatches.Patch(color='#ff1744',label='LOSS (✕red + dotted SL hit)'),
        mpatches.Patch(color='#ff9800',label='MA200'),
        mpatches.Patch(color='#90caf9',label='MA50'),
    ]
    ax.legend(handles=patches,loc='upper left',fontsize=9,facecolor='#1a1a2e',labelcolor='white')

plt.suptitle('HONEST VIEW — SwingHigh Resistance Reject + MA50上 + SLOPING → Short\n'
             'Green ↓ = winner  |  Red ✕ = loser  |  62% WR, EV+5.07, n=146',
             color='white',fontsize=12,y=1.01)
plt.tight_layout(h_pad=3)
plt.savefig('honest_chart.png',dpi=130,bbox_inches='tight',facecolor='#0d1117')
print("\nChart saved: honest_chart.png")
plt.close()
