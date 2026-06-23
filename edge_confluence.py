"""
CONFLUENCE — stack filters on the flagship (H1 swing-high sweep 3bar short + sloping).
Add: killzone hours, MA-overextension, premium (above MA50), big-sweep (deep wick).
Watch WR/EV vs sample depletion. Also realistic-tradeability metrics:
trades/month, MaxConsecLoss, avg hold (bars), expectancy in $.
1:1 ATR. No look-ahead.
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

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values;h=h1['h'].values;l=h1['l'].values;c=h1['c'].values
a=atr(h,l,c); h1['atr']=a
ma50=pd.Series(c).rolling(50).mean().values
ma200=pd.Series(c).rolling(200).mean().values
hr=h1['Date'].dt.hour.values
mon=h1['Date'].dt.strftime('%Y-%m').values
# regime
slope_k=50
reg=np.array(['NA']*n,dtype=object)
for i in range(n):
    if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
    s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
    if s>0.003 and c[i]>ma200[i]: reg[i]='UP'
    elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
    else: reg[i]='RANGE'

sh,sl=swings_w(h,l,n,1); shi=np.where(sh)[0]
# base sweep events with metadata
base=[]
for i in range(2,n):
    ps=shi[shi<i-1]
    if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]]:
        base.append(i)

def trade(i,horizon=24):
    if i+1>=n or np.isnan(a[i]) or a[i]<=0: return None
    entry=o[i+1]; rng=a[i]; tp=entry-rng; sl_=entry+rng; end=min(n,i+1+horizon)
    for k in range(i+1,end):
        if h[k]>=sl_: return (False,-rng-SPREAD,k-i)
        if l[k]<=tp: return (True,rng-SPREAD,k-i)
    return None
def report(label, idxs):
    tr=[trade(i) for i in idxs]; tr=[(x,i) for x,i in zip(tr,idxs) if x]
    if len(tr)<10:
        print(f"{label:<48} n={len(tr)} (too few)"); return
    res=[x[0] for x in tr]
    wr=100*np.mean([r[0] for r in res]); ev=np.mean([r[1] for r in res])
    tot=sum(r[1] for r in res); hold=np.mean([r[2] for r in res])
    # MCL
    mcl=cur=0
    for r in res:
        if not r[0]: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    months=len(set(mon[i] for x,i in tr))
    tpm=len(res)/max(months,1)
    print(f"{label:<48} WR={wr:4.0f}% EV={ev:+5.2f} n={len(res):3d} tot={tot:+5.0f} "
          f"MCL={mcl} hold={hold:3.1f}h ~{tpm:.0f}/mo")

print("#"*96)
print("# CONFLUENCE on flagship: H1 swing-high sweep (3bar) short")
print("#"*96)
report("1. raw sweep (no filter)", base)
slope=[i for i in base if reg[i] in ('UP','DOWN')]
report("2. + RANGE excluded (=flagship)", slope)
report("3. + DOWN regime only", [i for i in base if reg[i]=='DOWN'])
# killzone
kz=[i for i in slope if hr[i] in (7,8,9,10,13,14,15,16)]
report("4. flagship + killzone(UTC7-10,13-16)", kz)
report("5. flagship + hour 7-10 only", [i for i in slope if hr[i] in (7,8,9,10)])
# overextension: price above MA50 by >1 ATR at sweep
ovx=[i for i in slope if not np.isnan(ma50[i]) and (c[i]-ma50[i])/a[i]>0.5]
report("6. flagship + above MA50 (premium)", ovx)
ovx2=[i for i in slope if not np.isnan(ma50[i]) and (h[i]-ma50[i])/a[i]>1.5]
report("7. flagship + high stretched >1.5ATR vs MA50", ovx2)
# deep wick sweep (rejection strength): wick above prior high large
deep=[]
for i in slope:
    ps=shi[shi<i-1]
    if len(ps):
        lv=h[ps[-1]]; wick=h[i]-max(o[i],c[i])
        if (h[i]-lv)>0.3*a[i] and wick>0.4*a[i]: deep.append(i)
report("8. flagship + deep rejection wick", deep)
# stacked best: flagship + killzone + premium
stack=[i for i in slope if hr[i] in (7,8,9,10,13,14,15,16) and not np.isnan(ma50[i]) and c[i]>ma50[i]]
report("9. flagship + killzone + above MA50", stack)

print("\n# half-split of flagship(2) and best stack:")
mid=n//2
for label,idxs in [("flagship",slope),("stack#9",stack)]:
    for half,lo,hi in [("1st",0,mid),("2nd",mid,n)]:
        sub=[i for i in idxs if lo<=i<hi]
        tr=[trade(i) for i in sub]; tr=[x for x in tr if x]
        if len(tr)>=8:
            wr=100*np.mean([r[0] for r in tr]); ev=np.mean([r[1] for r in tr])
            print(f"   {label} {half}: WR={wr:.0f}% EV={ev:+.2f} n={len(tr)}")
