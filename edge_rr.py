"""
M5 / M15 RR-FOCUSED EDGE HUNT
New concepts (previously untested): Fibonacci OTE, MA deviation bounce, S/R bounce,
FVG+OTE confluence. Evaluate with asymmetric SL/TP brackets aiming for >=5pt TP.
No look-ahead. Entry = next bar open. Spread 0.3 cost.
Report per (setup, SL, TP): WR, EV(pt), n, breakeven WR, +half-split stability.
"""
import pandas as pd, numpy as np, warnings, itertools
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

def bracket(df,signals,sl_pt,tp_pt,horizon,lo=0,hi=None):
    """fixed-pt SL/TP first-touch. side in signals: +1 long / -1 short."""
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values
    n=len(df); hi=hi or n; res=[]
    for i,side in signals.items():
        if i<lo or i>=hi or i+1>=n: continue
        entry=o[i+1]
        tp=entry+tp_pt*side; sl=entry-sl_pt*side; win=None; end=min(n,i+1+horizon)
        for k in range(i+1,end):
            if side>0:
                if l[k]<=sl: win=False;break
                if h[k]>=tp: win=True;break
            else:
                if h[k]>=sl: win=False;break
                if l[k]<=tp: win=True;break
        if win is None: continue  # neither hit -> discard (timeout) for RR clarity
        pnl=(tp_pt-SPREAD) if win else (-sl_pt-SPREAD)
        res.append((win,pnl))
    return res
def stat(res):
    if len(res)<20: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res),tot=sum(p))

# ---- causal signal builders ----
def build_signals(df):
    n=len(df);o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values
    a=atr(h,l,c); df['atr']=a
    sh,sl=swings(h,l,n); shi=np.where(sh)[0];sli=np.where(sl)[0]
    ma50=pd.Series(c).rolling(50).mean().values
    ma200=pd.Series(c).rolling(200).mean().values
    S={}

    # ---- Fibonacci OTE ----
    # bullish leg: swing low A then swing high B; retrace into 0.618-0.786 -> long
    fib_long={}; fib_short={}
    # build ordered confirmed swings (avail at idx+2) as (idx, price, type)
    piv=sorted([(i,'H') for i in shi]+[(i,'L') for i in sli])
    for a_ in range(len(piv)-1):
        i1,t1=piv[a_]; i2,t2=piv[a_+1]
        if t1=='L' and t2=='H':
            A=l[i1]; B=h[i2]; size=B-A
            if size< max(5, 1.5*np.nanmean(a)): continue
            zlo=B-0.786*size; zhi=B-0.618*size  # OTE zone
            start=i2+2
            for k in range(start,min(n,start+horizon_lookup)):
                if l[k]<=zhi and l[k]>=zlo*0:  # price dips into zone
                    if l[k]<=zhi and l[k]>=zlo:
                        fib_long[k]=1; break
                if c[k] > B: break  # leg invalidated by new high before retrace
                if l[k] < zlo: break  # blew past zone
        if t1=='H' and t2=='L':
            A=h[i1]; B=l[i2]; size=A-B
            if size< max(5, 1.5*np.nanmean(a)): continue
            zlo=B+0.618*size; zhi=B+0.786*size
            start=i2+2
            for k in range(start,min(n,start+horizon_lookup)):
                if h[k]>=zlo and h[k]<=zhi:
                    fib_short[k]=-1; break
                if c[k] < B: break
                if h[k] > zhi: break
    S['Fib OTE long']=fib_long
    S['Fib OTE short']=fib_short

    # ---- MA deviation bounce (mean reversion) ----
    for mname,ma in (('MA50',ma50),('MA200',ma200)):
        for thr in (2.0,2.5,3.0):
            dl={}; ds={}
            prev_below=False; prev_above=False
            for i in range(n):
                if np.isnan(ma[i]) or np.isnan(a[i]) or a[i]<=0: continue
                dev=(c[i]-ma[i])/a[i]
                if dev<=-thr and not prev_below: dl[i]=1
                if dev>=thr and not prev_above: ds[i]=-1
                prev_below = dev<=-thr
                prev_above = dev>=thr
            S[f'{mname} dev<=-{thr} long']=dl
            S[f'{mname} dev>=+{thr} short']=ds

    # ---- S/R bounce off prior swing ----
    sr_long={}; sr_short={}
    tol=0.6
    for i in range(3,n):
        ps_h=shi[shi<i-1]; ps_l=sli[sli<i-1]
        if len(ps_h):
            lv=h[ps_h[-1]]
            if abs(h[i]-lv)<=tol and c[i]<o[i] and c[i]<lv: sr_short[i]=-1
        if len(ps_l):
            lv=l[ps_l[-1]]
            if abs(l[i]-lv)<=tol and c[i]>o[i] and c[i]>lv: sr_long[i]=1
    S['S/R swing-high reject short']=sr_short
    S['S/R swing-low bounce long']=sr_long

    # ---- Fib OTE + trend filter (discount/premium via MA200) ----
    S['Fib OTE long + below MA200']={i:1 for i in fib_long if not np.isnan(ma200[i]) and c[i]<ma200[i]}
    S['Fib OTE short + above MA200']={i:-1 for i in fib_short if not np.isnan(ma200[i]) and c[i]>ma200[i]}
    S['Fib OTE short + below MA200']={i:-1 for i in fib_short if not np.isnan(ma200[i]) and c[i]<ma200[i]}
    return S

horizon_lookup=60  # bars to wait for retrace into fib zone

def run(tf,path,horizon):
    df=load(path); n=len(df); mid=n//2
    print(f"\n{'#'*90}\n# {tf}  ({n} bars)  eval-horizon={horizon}\n{'#'*90}")
    S=build_signals(df)
    brackets=[(2,5),(2,8),(3,8),(3,10),(2.5,5),(5,10),(2,6),(2.5,8)]
    out=[]
    for name,sig in S.items():
        if len(sig)<20: continue
        for sl_pt,tp_pt in brackets:
            res=bracket(df,sig,sl_pt,tp_pt,horizon,0,n)
            s=stat(res)
            if not s: continue
            be=100*sl_pt/(sl_pt+tp_pt)  # breakeven WR (ignoring spread)
            if s['ev']>0 and s['n']>=25:
                # half-split
                r1=stat(bracket(df,sig,sl_pt,tp_pt,horizon,0,mid))
                r2=stat(bracket(df,sig,sl_pt,tp_pt,horizon,mid,n))
                stable = r1 and r2 and r1['ev']>0 and r2['ev']>0
                out.append((name,sl_pt,tp_pt,s,be,stable,r1,r2))
    out.sort(key=lambda x:-x[3]['tot'])
    print(f"{'setup':<30}{'SL':>4}{'TP':>4}{'WR%':>6}{'BE%':>6}{'EV':>7}{'n':>6}{'totPt':>8}  stable")
    for name,slp,tpp,s,be,stable,r1,r2 in out:
        tag=" <<STABLE+5pt" if (stable and tpp>=5) else (" stable" if stable else "")
        print(f"{name:<30}{slp:>4}{tpp:>4}{s['wr']:>6.1f}{be:>6.0f}{s['ev']:>+7.2f}{s['n']:>6}{s['tot']:>8.0f}{tag}")
    return out

run('M15',f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv",48)
run('M5',f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72)
