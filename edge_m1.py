"""
M1 edge screen + stability (was charted only, never statistically tested).
Same triple-barrier 1:1 ATR, no look-ahead. M1 = 100k bars, keep it lean.
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
def evaluate(df,signals,horizon,lo=0,hi=None):
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values;a=df['atr'].values
    n=len(df); hi=hi or n; res=[]
    for i,side in signals.items():
        if i<lo or i>=hi or i+1>=n or np.isnan(a[i]): continue
        entry=o[i+1]; rng=a[i]
        if rng<=0: continue
        tp=entry+rng*side; sl=entry-rng*side; win=None; end=min(n,i+1+horizon)
        for k in range(i+1,end):
            if side>0:
                if l[k]<=sl: win=False;break
                if h[k]>=tp: win=True;break
            else:
                if h[k]>=sl: win=False;break
                if l[k]<=tp: win=True;break
        if win is None:
            ex=c[end-1]; pnl=(ex-entry)*side; win=pnl>0; pnl-=SPREAD
        else: pnl=(rng-SPREAD) if win else (-rng-SPREAD)
        res.append((win,pnl))
    return res
def st(res):
    if len(res)<10: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res))

m1=load(f"{UP}/9c76cb47-XAUUSD_M1_Mar_Jun.csv")
n=len(m1); o=m1['o'].values;h=m1['h'].values;l=m1['l'].values;c=m1['c'].values
m1['atr']=atr(h,l,c)
print(f"M1 loaded: {n} bars  {m1['Date'].iloc[0]} -> {m1['Date'].iloc[-1]}")
sh,sl=swings(h,l,n); shi=np.where(sh)[0];sli=np.where(sl)[0]
ma200=pd.Series(c).rolling(200).mean().values
hr=m1['Date'].dt.hour.values
H=60  # horizon ~1h

sig={}
sig['_ALL short']={i:-1 for i in range(n)}
sig['_ALL long']={i:1 for i in range(n)}
# SwingHigh sweep +/- MA200 filter
sw={}; swf={}
for i in range(3,n):
    ps=shi[shi<i-1]
    if len(ps):
        lv=h[ps[-1]]
        if h[i]>lv and c[i]<lv:
            sw[i]=-1
            if not np.isnan(ma200[i]) and c[i]<ma200[i]: swf[i]=-1
sig['SwingHigh sweep short']=sw
sig['SwingHigh sweep+belowMA200 short']=swf
# SwingLow sweep long
sl2={}
for i in range(3,n):
    ps=sli[sli<i-1]
    if len(ps):
        lv=l[ps[-1]]
        if l[i]<lv and c[i]>lv: sl2[i]=1
sig['SwingLow sweep long']=sl2
# hours (UTC) short, sample to keep light
for Hh in (7,8,9,10,13,14):
    sig[f'hour{Hh:02d} short']={i:-1 for i in range(n) if hr[i]==Hh}

mid=n//2
print(f"\n{'signal':<34}{'FULL wr/n':>16}{'1stHALF':>14}{'2ndHALF':>14}")
for name,s in sig.items():
    full=st(evaluate(m1,s,H,0,n)); h1=st(evaluate(m1,s,H,0,mid)); h2=st(evaluate(m1,s,H,mid,n))
    def f(x): return f"{x['wr']:.0f}%/{x['n']}" if x else "-"
    flag="  <<STABLE" if (full and h1 and h2 and h1['wr']>=53 and h2['wr']>=53) else ""
    print(f"{name:<34}{f(full):>16}{f(h1):>14}{f(h2):>14}{flag}")
