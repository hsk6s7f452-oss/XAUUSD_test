"""
SWING RULE SWEEP — how does swing-strength change the edge?
Parametrize swing window w: pivot = extreme of (2w+1) bars, confirmed at i+w.
Re-test the proven SMC edge (SwingHigh sweep + MA200 filter short) and sweep-long,
plus MSB, across w in {1,2,3,4,5,7}. No look-ahead. Bracket eval + half-split.
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
    """pivot = extreme of window [i-w, i+w], strictly greater/less than neighbors."""
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(w,n-w):
        wh=h[i-w:i+w+1]; wl=l[i-w:i+w+1]
        if h[i]==wh.max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==wl.min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl
def bracket(df,signals,sl_pt,tp_pt,horizon,lo=0,hi=None,use_atr=False):
    o=df['o'].values;h=df['h'].values;l=df['l'].values;c=df['c'].values;a=df['atr'].values
    n=len(df); hi=hi or n; res=[]
    for i,side in signals.items():
        if i<lo or i>=hi or i+1>=n: continue
        if use_atr and (np.isnan(a[i]) or a[i]<=0): continue
        entry=o[i+1]
        slp=a[i]*sl_pt if use_atr else sl_pt
        tpp=a[i]*tp_pt if use_atr else tp_pt
        tp=entry+tpp*side; sl=entry-slp*side; win=None; end=min(n,i+1+horizon)
        for k in range(i+1,end):
            if side>0:
                if l[k]<=sl: win=False;break
                if h[k]>=tp: win=True;break
            else:
                if h[k]>=sl: win=False;break
                if l[k]<=tp: win=True;break
        if win is None: continue
        res.append((win,(tpp-SPREAD) if win else (-slp-SPREAD)))
    return res
def stat(res):
    if len(res)<15: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res),tot=sum(p))

def sweep_signals(df,w):
    n=len(df);h=df['h'].values;l=df['l'].values;c=df['c'].values
    ma200=pd.Series(c).rolling(200).mean().values
    sh,sl=swings_w(h,l,n,w); shi=np.where(sh)[0];sli=np.where(sl)[0]
    sw_short={}; sw_short_f={}; sw_long={}; sw_long_f={}
    for i in range(w+1,n):
        ps=shi[shi<i-1]
        if len(ps):
            lv=h[ps[-1]]
            if h[i]>lv and c[i]<lv:
                sw_short[i]=-1
                if not np.isnan(ma200[i]) and c[i]<ma200[i]: sw_short_f[i]=-1
        pl=sli[sli<i-1]
        if len(pl):
            lv=l[pl[-1]]
            if l[i]<lv and c[i]>lv:
                sw_long[i]=1
                if not np.isnan(ma200[i]) and c[i]>ma200[i]: sw_long_f[i]=1
    return sw_short,sw_short_f,sw_long,sw_long_f

def run(tf,path,horizon):
    df=load(path); n=len(df); mid=n//2
    df['atr']=atr(df['h'].values,df['l'].values,df['c'].values)
    print(f"\n{'#'*92}\n# {tf} ({n} bars)  SWING-RULE SWEEP   [eval SL3/TP8 fixed-pt, breakeven WR=27%]\n{'#'*92}")
    print(f"{'setup':<34}{'w(bars)':>8}{'WR%':>7}{'EV':>7}{'n':>6}{'totPt':>8}  half(EV)")
    for w in (1,2,3,4,5,7):
        sw_s,sw_sf,sw_l,sw_lf=sweep_signals(df,w)
        for name,sig in [('SwingHigh sweep short',sw_s),
                         ('SwingHigh sweep+belowMA200',sw_sf),
                         ('SwingLow sweep long',sw_l),
                         ('SwingLow sweep+aboveMA200',sw_lf)]:
            s=stat(bracket(df,sig,3,8,horizon))
            if not s:
                continue
            r1=stat(bracket(df,sig,3,8,horizon,0,mid)); r2=stat(bracket(df,sig,3,8,horizon,mid,n))
            hh=f"{r1['ev']:+.2f}/{r2['ev']:+.2f}" if r1 and r2 else "-"
            flag="  <<STABLE" if (r1 and r2 and r1['ev']>0 and r2['ev']>0 and s['ev']>0) else ""
            star=" ***" if s['ev']>0.5 else ""
            print(f"{name:<34}{(2*w+1):>6}b {'':>1}{s['wr']:>7.1f}{s['ev']:>+7.2f}{s['n']:>6}{s['tot']:>8.0f}  {hh}{flag}{star}")
        print()

run('H1', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24)
run('M15',f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv",48)
run('M5', f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72)

# ── Same sweep, but ORIGINAL 1:1 ATR lens (the profile where sweep worked) ──
def run_1to1(tf,path,horizon):
    df=load(path); n=len(df); mid=n//2
    df['atr']=atr(df['h'].values,df['l'].values,df['c'].values)
    print(f"\n{'='*92}\n= {tf} ({n})  SWEEP under 1:1 ATR lens (TP=+1ATR/SL=-1ATR, breakeven WR=50%)\n{'='*92}")
    print(f"{'setup':<32}{'w':>5}{'WR%':>7}{'EV':>7}{'n':>6}  half(WR)")
    for w in (1,2,3,4,5,7):
        sw_s,sw_sf,sw_l,sw_lf=sweep_signals(df,w)
        for name,sig in [('SwingHigh sweep+belowMA200 short',sw_sf),
                         ('SwingHigh sweep short',sw_s)]:
            s=stat(bracket(df,sig,1,1,horizon,use_atr=True))
            if not s: continue
            r1=stat(bracket(df,sig,1,1,horizon,0,mid,use_atr=True)); r2=stat(bracket(df,sig,1,1,horizon,mid,n,use_atr=True))
            hh=f"{r1['wr']:.0f}/{r2['wr']:.0f}" if r1 and r2 else "-"
            flag="  <<STABLE" if (r1 and r2 and r1['wr']>=53 and r2['wr']>=53 and s['wr']>=55) else ""
            print(f"{name:<32}{(2*w+1):>4}b{s['wr']:>7.1f}{s['ev']:>+7.2f}{s['n']:>6}  {hh}{flag}")
        print()

run_1to1('H1', f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv",24)
run_1to1('M15',f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv",48)
run_1to1('M5', f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv",72)
