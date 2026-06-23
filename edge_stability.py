"""
STABILITY TEST — does an edge hold in BOTH halves of the period?
Also a market-neutral baseline: what's the WR of 'short any random bar'?
If an edge's WR isn't clearly above the directional baseline, it's just trend drift.
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
def swings(h,l,n):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(2,n-2):
        if h[i]==h[i-2:i+3].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-2:i+3].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl
def evaluate(df,signals,horizon,amult=1.0,lo=0,hi=None):
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values;a=df['atr'].values
    n=len(df); hi=hi or n; res=[]
    for i,side in signals.items():
        if i<lo or i>=hi: continue
        if i+1>=n or np.isnan(a[i]): continue
        entry=o[i+1]; rng=a[i]*amult
        if rng<=0: continue
        tp=entry+rng*side; sl=entry-rng*side
        win=None; end=min(n,i+1+horizon)
        for k in range(i+1,end):
            if side>0:
                if l[k]<=sl: win=False;break
                if h[k]>=tp: win=True;break
            else:
                if h[k]>=sl: win=False;break
                if l[k]<=tp: win=True;break
        if win is None:
            ex=c[end-1]; pnl=(ex-entry)*side; win=pnl>0; pnl-=SPREAD
        else:
            pnl=(rng-SPREAD) if win else (-rng-SPREAD)
        res.append((win,pnl))
    return res
def st(res):
    if len(res)<10: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res))

def build(df,horizon):
    n=len(df);o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values
    df['atr']=atr(h,l,c)
    sh,sl=swings(h,l,n); shi=np.where(sh)[0];sli=np.where(sl)[0]
    ma200=pd.Series(c).rolling(200).mean().values
    ma50=pd.Series(c).rolling(50).mean().values
    hr=df['Date'].dt.hour.values
    sig={}
    # baseline: every bar short / long
    sig['_ALL short']={i:-1 for i in range(n)}
    sig['_ALL long']={i:1 for i in range(n)}
    # MSB bear
    msb_bear={};lsh=None;lsl=None
    for i in range(n):
        if i>=2 and sh[i-2]: lsh=h[i-2]
        if i>=2 and sl[i-2]: lsl=l[i-2]
        if lsh is not None and c[i]>lsh and c[i-1]<=lsh: lsh=None
        if lsl is not None and c[i]<lsl and c[i-1]>=lsl: msb_bear[i]=-1; lsl=None
    sig['MSB-bear short']=msb_bear
    # SwingHigh sweep + below MA200
    sweep={}
    for i in range(3,n):
        ps=[s for s in shi if s<i-1]
        if ps:
            lv=h[ps[-1]]
            if h[i]>lv and c[i]<lv and not np.isnan(ma200[i]) and c[i]<ma200[i]:
                sweep[i]=-1
    sig['SwingHigh sweep+belowMA200 short']=sweep
    # SwingHigh sweep (no filter)
    sweep2={}
    for i in range(3,n):
        ps=[s for s in shi if s<i-1]
        if ps:
            lv=h[ps[-1]]
            if h[i]>lv and c[i]<lv: sweep2[i]=-1
    sig['SwingHigh sweep short']=sweep2
    # hour-based notable
    for H in (10,7,23,3):
        side=1 if H==23 else -1
        sig[f'hour{H:02d} {"long" if side>0 else "short"}']={i:side for i in range(n) if hr[i]==H}
    return sig

def report(tf,df,horizon):
    n=len(df); mid=n//2
    sig=build(df,horizon)
    print(f"\n{'='*84}\n{tf}  (n={n}, split at bar {mid})  horizon={horizon}\n{'='*84}")
    print(f"{'signal':<34}{'FULL wr/n':>16}{'1stHALF wr/n':>16}{'2ndHALF wr/n':>16}")
    for name,s in sig.items():
        full=st(evaluate(df,s,horizon,1.0,0,n))
        h1=st(evaluate(df,s,horizon,1.0,0,mid))
        h2=st(evaluate(df,s,horizon,1.0,mid,n))
        def f(x): return f"{x['wr']:.0f}%/{x['n']}" if x else "-"
        flag=""
        if full and h1 and h2 and h1['wr']>=53 and h2['wr']>=53: flag="  <<STABLE"
        print(f"{name:<34}{f(full):>16}{f(h1):>16}{f(h2):>16}{flag}")

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv")
m5=load(f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv")
report('H1',h1,24)
report('M15',m15,32)
report('M5',m5,48)
