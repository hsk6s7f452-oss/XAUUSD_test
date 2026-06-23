"""
FIBONACCI LEVEL TEST — which ratio actually works?
Leg = consecutive confirmed 5-bar swings (A->B). Mechanical, reproducible.
Retracement levels: enter CONTINUATION on pullback to level.
Extension levels (>1): enter FADE (reversal) when price reaches projection.
No look-ahead (leg available at B+2). Bracket eval. Compare WR vs breakeven per level.
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
def bracket(df,signals,sl_pt,tp_pt,horizon,lo=0,hi=None):
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values
    n=len(df); hi=hi or n; res=[]
    for i,side in signals.items():
        if i<lo or i>=hi or i+1>=n: continue
        entry=o[i+1]; tp=entry+tp_pt*side; sl=entry-sl_pt*side; win=None; end=min(n,i+1+horizon)
        for k in range(i+1,end):
            if side>0:
                if l[k]<=sl: win=False;break
                if h[k]>=tp: win=True;break
            else:
                if h[k]>=sl: win=False;break
                if l[k]<=tp: win=True;break
        if win is None: continue
        res.append((win,(tp_pt-SPREAD) if win else (-sl_pt-SPREAD)))
    return res
def stat(res):
    if len(res)<15: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res),tot=sum(p))

RETR=[0.236,0.382,0.5,0.618,0.705,0.786,0.886]   # incl unconventional 0.705,0.886
EXT =[1.13,1.272,1.414,1.618,2.0,2.168,2.618]     # incl unconventional 1.13,1.414,2.168
LOOK=80   # bars to wait for price to reach a level

def build_fib(df):
    n=len(df);o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values
    a=atr(h,l,c); df['atr']=a; mean_atr=np.nanmean(a)
    sh,sl=swings(h,l,n); shi=np.where(sh)[0];sli=np.where(sl)[0]
    piv=sorted([(i,'H') for i in shi]+[(i,'L') for i in sli])
    # retracement-continuation signals per level; extension-fade per level
    retr_sig={r:{} for r in RETR}
    ext_sig={e:{} for e in EXT}
    for a_ in range(len(piv)-1):
        i1,t1=piv[a_]; i2,t2=piv[a_+1]
        start=i2+2
        if t1=='L' and t2=='H':   # bullish leg up
            A=l[i1]; B=h[i2]; size=B-A
            if size< max(8,2*mean_atr): continue
            # retracement: pullback DOWN to level -> LONG continuation
            for r in RETR:
                Lr=B-r*size
                for k in range(start,min(n,start+LOOK)):
                    if c[k]<A: break               # full retrace -> leg dead
                    if l[k]<=Lr:                    # touched level
                        retr_sig[r][k]=1; break
            # extension: price pushes UP beyond B to A+e*size -> FADE short
            for e in EXT:
                Le=A+e*size
                for k in range(start,min(n,start+LOOK)):
                    if c[k]<A: break
                    if h[k]>=Le:
                        ext_sig[e][k]=-1; break
        elif t1=='H' and t2=='L': # bearish leg down
            A=h[i1]; B=l[i2]; size=A-B
            if size< max(8,2*mean_atr): continue
            for r in RETR:
                Lr=B+r*size
                for k in range(start,min(n,start+LOOK)):
                    if c[k]>A: break
                    if h[k]>=Lr:
                        retr_sig[r][k]=-1; break    # SHORT continuation
            for e in EXT:
                Le=A-e*size
                for k in range(start,min(n,start+LOOK)):
                    if c[k]>A: break
                    if l[k]<=Le:
                        ext_sig[e][k]=1; break       # LONG fade
    return retr_sig,ext_sig

def run(tf,path,horizon):
    df=load(path); n=len(df); mid=n//2
    retr,ext=build_fib(df)
    print(f"\n{'#'*86}\n# {tf}  ({n} bars)   [SL3/TP8 = RR2.67, breakeven WR=27%]\n{'#'*86}")
    print("RETRACEMENT (continuation entry @ pullback level)")
    print(f"{'fib':>7}{'WR%':>7}{'EV':>7}{'n':>6}{'totPt':>8}  half-split(EV)")
    for r in RETR:
        s=stat(bracket(df,retr[r],3,8,horizon))
        if not s:
            print(f"{r:>7}   (n<15)"); continue
        r1=stat(bracket(df,retr[r],3,8,horizon,0,mid)); r2=stat(bracket(df,retr[r],3,8,horizon,mid,n))
        hh=f"{r1['ev']:+.2f}/{r2['ev']:+.2f}" if r1 and r2 else "-"
        flag="  <<STABLE" if (r1 and r2 and r1['ev']>0 and r2['ev']>0 and s['ev']>0) else ""
        print(f"{r:>7}{s['wr']:>7.1f}{s['ev']:>+7.2f}{s['n']:>6}{s['tot']:>8.0f}  {hh}{flag}")
    print("\nEXTENSION (fade/reversal entry @ projection level)  [breakeven WR=27%]")
    print(f"{'fib':>7}{'WR%':>7}{'EV':>7}{'n':>6}{'totPt':>8}  half-split(EV)")
    for e in EXT:
        s=stat(bracket(df,ext[e],3,8,horizon))
        if not s:
            print(f"{e:>7}   (n<15)"); continue
        r1=stat(bracket(df,ext[e],3,8,horizon,0,mid)); r2=stat(bracket(df,ext[e],3,8,horizon,mid,n))
        hh=f"{r1['ev']:+.2f}/{r2['ev']:+.2f}" if r1 and r2 else "-"
        flag="  <<STABLE" if (r1 and r2 and r1['ev']>0 and r2['ev']>0 and s['ev']>0) else ""
        print(f"{e:>7}{s['wr']:>7.1f}{s['ev']:>+7.2f}{s['n']:>6}{s['tot']:>8.0f}  {hh}{flag}")

run('H1', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24)
run('M15',f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv",48)
run('M5', f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72)

# ── Trend-filtered retest: only continuation in MA200 direction ──
def run_filtered(tf,path,horizon):
    df=load(path); n=len(df); mid=n//2
    c=df['c'].values; ma200=pd.Series(c).rolling(200).mean().values
    retr,ext=build_fib(df)
    print(f"\n{'='*86}\n= {tf} FIB retracement, TREND-FILTERED (short only when c<MA200) [BE WR=27%]\n{'='*86}")
    print(f"{'fib':>7}{'WR%':>7}{'EV':>7}{'n':>6}{'totPt':>8}  half(EV)")
    for r in RETR:
        sig={i:s for i,s in retr[r].items() if s<0 and not np.isnan(ma200[i]) and c[i]<ma200[i]}
        s=stat(bracket(df,sig,3,8,horizon))
        if not s: print(f"{r:>7}  (n<15)"); continue
        r1=stat(bracket(df,sig,3,8,horizon,0,mid)); r2=stat(bracket(df,sig,3,8,horizon,mid,n))
        hh=f"{r1['ev']:+.2f}/{r2['ev']:+.2f}" if r1 and r2 else "-"
        flag="  <<STABLE" if (r1 and r2 and r1['ev']>0 and r2['ev']>0 and s['ev']>0) else ""
        print(f"{r:>7}{s['wr']:>7.1f}{s['ev']:>+7.2f}{s['n']:>6}{s['tot']:>8.0f}  {hh}{flag}")

run_filtered('H1', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24)
run_filtered('M15',f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv",48)
run_filtered('M5', f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72)
