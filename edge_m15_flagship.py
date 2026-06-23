"""
Port flagship to M15 to grow sample.
A) M15 self-contained: sweep(3bar) + M15 RANGE-excluded + M15 above-MA50.
B) Multi-TF: M15 sweep entry, but H1 regime as environment filter (HTF env -> LTF trigger).
1:1 ATR, no look-ahead. Half-split + monthly + tradeability metrics.
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
def regime(c,slope_k):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if s>0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

# H1 regime, mapped by timestamp
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
h1reg=regime(h1['c'].values,50)
h1map=dict(zip(h1['Date'].values, h1reg))  # exact hour timestamp -> regime
h1ts=h1['Date'].values
def h1_regime_at(ts):
    # most recent H1 bar at or before ts (no look-ahead)
    idx=np.searchsorted(h1ts, ts, side='right')-1
    if idx<0: return 'NA'
    return h1reg[idx]

m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv"); n=len(m15)
o=m15['o'].values;h=m15['h'].values;l=m15['l'].values;c=m15['c'].values
a=atr(h,l,c); m15['atr']=a
ma50=pd.Series(c).rolling(50).mean().values
m15reg=regime(c,200)   # ~50h on M15
mon=m15['Date'].dt.strftime('%Y-%m').values
ts=m15['Date'].values

sh,sl=swings_w(h,l,n,1); shi=np.where(sh)[0]
base=[]
for i in range(2,n):
    ps=shi[shi<i-1]
    if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]]: base.append(i)

def trade(i,horizon=48):
    if i+1>=n or np.isnan(a[i]) or a[i]<=0: return None
    entry=o[i+1]; rng=a[i]; tp=entry-rng; slv=entry+rng; end=min(n,i+1+horizon)
    for k in range(i+1,end):
        if h[k]>=slv: return (False,-rng-SPREAD)
        if l[k]<=tp: return (True,rng-SPREAD)
    return None
def report(label, idxs):
    tr=[(trade(i),i) for i in idxs]; tr=[(x,i) for x,i in tr if x]
    if len(tr)<10: print(f"{label:<46} n={len(tr)} too few"); return None
    res=[x[0] for x in tr]
    wr=100*np.mean([r[0] for r in res]); ev=np.mean([r[1] for r in res]); tot=sum(r[1] for r in res)
    mcl=cur=0
    for r in res:
        if not r[0]: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    months=len(set(mon[i] for x,i in tr)); tpm=len(res)/max(months,1)
    # half-split
    mid=n//2
    h1_=[r for r,i in zip(res,[i for x,i in tr]) if i<mid]
    h2_=[r for r,i in zip(res,[i for x,i in tr]) if i>=mid]
    def hw(x): return f"{100*np.mean([r[0] for r in x]):.0f}%/{len(x)}" if len(x)>=8 else "-"
    print(f"{label:<46} WR={wr:4.0f}% EV={ev:+5.2f} n={len(res):4d} tot={tot:+5.0f} MCL={mcl} ~{tpm:.0f}/mo  half={hw(h1_)},{hw(h2_)}")
    return idxs

print("#"*100)
print("# M15 flagship port")
print("#"*100)
report("raw M15 sweep", base)
A=[i for i in base if m15reg[i] in ('UP','DOWN')]
report("A1. + M15 RANGE excluded", A)
A2=[i for i in A if not np.isnan(ma50[i]) and c[i]>ma50[i]]
report("A2. + M15 above MA50 (premium)", A2)
# B: H1 regime env filter
B=[i for i in base if h1_regime_at(ts[i]) in ('UP','DOWN')]
report("B1. + H1 regime sloping (multi-TF)", B)
B2=[i for i in B if not np.isnan(ma50[i]) and c[i]>ma50[i]]
report("B2. + H1 sloping + M15 above MA50", B2)
# B down-only
Bd=[i for i in base if h1_regime_at(ts[i])=='DOWN']
report("B3. + H1 DOWN only", Bd)
Bd2=[i for i in Bd if not np.isnan(ma50[i]) and c[i]>ma50[i]]
report("B4. + H1 DOWN + M15 above MA50", Bd2)

print("\n# monthly (B2 = H1 sloping + M15 premium):")
for m in sorted(set(mon)):
    sub=[i for i in B2 if mon[i]==m]
    tr=[trade(i) for i in sub]; tr=[x for x in tr if x]
    if len(tr)>=5:
        wr=100*np.mean([r[0] for r in tr]); ev=np.mean([r[1] for r in tr])
        print(f"   {m}: WR={wr:.0f}% EV={ev:+.2f} n={len(tr)}")
    else:
        print(f"   {m}: n={len(tr)} (few)")
