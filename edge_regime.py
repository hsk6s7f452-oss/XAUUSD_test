"""
REGIME ROBUSTNESS — are the short edges just downtrend bets?
1) classify each bar UP/RANGE/DOWN causally (MA200 slope + price side).
2) bucket each edge's trades by regime, report WR/EV/n.
3) SYMMETRY test: short-sweep in DOWN  vs  long-sweep (mirror) in UP.
   If mirror works in UP -> edge generalizes ("fade sweep WITH the trend").
   If only short wins regardless -> it's a downtrend artifact.
No look-ahead. 1:1 ATR lens (the profile where sweep works).
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3
def load(p):
    df=pd.read_csv(p); df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
    return df
def atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a
def swings_w(h,l,n,w):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl
def evbar(df,i,side,horizon):
    """single trade, 1:1 ATR, return (win,pnl) or None."""
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values;a=df['atr'].values
    n=len(df)
    if i+1>=n or np.isnan(a[i]) or a[i]<=0: return None
    entry=o[i+1]; rng=a[i]; tp=entry+rng*side; sl=entry-rng*side; end=min(n,i+1+horizon)
    for k in range(i+1,end):
        if side>0:
            if l[k]<=sl: return (False,-rng-SPREAD)
            if h[k]>=tp: return (True,rng-SPREAD)
        else:
            if h[k]>=sl: return (False,-rng-SPREAD)
            if l[k]<=tp: return (True,rng-SPREAD)
    return None
def agg(trades):
    t=[x for x in trades if x]
    if len(t)<10: return None
    w=[x[0] for x in t]; p=[x[1] for x in t]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(t))
def fmt(s): return f"WR={s['wr']:.0f}% EV={s['ev']:+.2f} n={s['n']}" if s else "n<10"

def run(tf,path,horizon,w=2,slope_k=50):
    df=load(path); n=len(df)
    c=df['c'].values;h=df['h'].values;l=df['l'].values
    df['atr']=atr(h,l,c)
    ma200=pd.Series(c).rolling(200).mean().values
    # causal regime: MA200 slope over slope_k bars + price side
    reg=np.array(['RANGE']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        sl_pct=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if sl_pct> 0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif sl_pct< -0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    sh,sl=swings_w(h,l,n,w); shi=np.where(sh)[0];sli=np.where(sl)[0]

    # sweep signals
    short_sweep=[]; long_sweep=[]
    for i in range(w+1,n):
        ps=shi[shi<i-1]
        if len(ps):
            lv=h[ps[-1]]
            if h[i]>lv and c[i]<lv: short_sweep.append(i)
        pl=sli[sli<i-1]
        if len(pl):
            lv=l[pl[-1]]
            if l[i]<lv and c[i]>lv: long_sweep.append(i)

    print(f"\n{'='*78}\n{tf}  regime mix: " +
          ", ".join(f"{r}={np.sum(reg==r)}" for r in ['UP','RANGE','DOWN']) +
          f"  (slope_k={slope_k})\n{'='*78}")

    print("SHORT after swing-HIGH sweep, by regime:")
    for r in ['DOWN','RANGE','UP']:
        s=agg([evbar(df,i,-1,horizon) for i in short_sweep if reg[i]==r])
        print(f"   {r:<6}: {fmt(s)}")
    print("LONG  after swing-LOW sweep (mirror), by regime:")
    for r in ['UP','RANGE','DOWN']:
        s=agg([evbar(df,i,1,horizon) for i in long_sweep if reg[i]==r])
        print(f"   {r:<6}: {fmt(s)}")

    # SYMMETRY verdict
    sD=agg([evbar(df,i,-1,horizon) for i in short_sweep if reg[i]=='DOWN'])
    lU=agg([evbar(df,i,1,horizon) for i in long_sweep if reg[i]=='UP'])
    print(f"\n  >>> SYMMETRY: short@DOWN [{fmt(sD)}]   vs   long@UP [{fmt(lU)}]")
    if sD and lU and sD['wr']>=55 and lU['wr']>=55:
        print("  >>> VERDICT: GENERALIZES — mechanism = fade sweep WITH the trend (both directions).")
    elif sD and sD['wr']>=55 and (not lU or lU['wr']<52):
        print("  >>> VERDICT: DOWNTREND ARTIFACT — only short wins; long-mirror fails in uptrend.")
    else:
        print("  >>> VERDICT: inconclusive (sample/strength).")

    # monthly breakdown of short-sweep
    mon=df['Date'].dt.strftime('%Y-%m').values
    print("\n  short-sweep by month:")
    for m in sorted(set(mon)):
        s=agg([evbar(df,i,-1,horizon) for i in short_sweep if mon[i]==m])
        print(f"     {m}: {fmt(s)}")

run('H1', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24,w=2,slope_k=50)
run('H1-3bar', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24,w=1,slope_k=50)
run('M5', f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72,w=2,slope_k=200)
