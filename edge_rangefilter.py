"""
RANGE-EXCLUSION FILTER — how much does sitting out chop improve each edge?
Compare each proven edge: BASELINE (all)  vs  SLOPING (regime != RANGE)  vs  DOWN-only.
Regime = causal MA200 slope over slope_k bars + price side.
Edges: H1 swing-high sweep short (1:1 ATR), M5 sweep short (1:1),
       M15/M5 MA-overextension short (3pt/10pt), H1 hour10 short (1:1).
Half-split shown for the filtered winner.
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
def regime(df,slope_k):
    n=len(df);c=df['c'].values
    ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        sl=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if sl>0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif sl<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg
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
    if len(res)<10: return None
    w=[r[0] for r in res]; p=[r[1] for r in res]
    return dict(wr=100*np.mean(w),ev=np.mean(p),n=len(res),tot=sum(p))
def fmt(s): return f"WR={s['wr']:4.0f}% EV={s['ev']:+5.2f} n={s['n']:4d} tot={s['tot']:+5.0f}" if s else "n<10"

def line(label, df, sig, slp,tpp,horizon,reg,use_atr):
    base=sig
    slope={i:v for i,v in sig.items() if reg[i] in ('UP','DOWN')}
    down ={i:v for i,v in sig.items() if reg[i]=='DOWN'}
    sb=stat(bracket(df,base,slp,tpp,horizon,use_atr=use_atr))
    ss=stat(bracket(df,slope,slp,tpp,horizon,use_atr=use_atr))
    sd=stat(bracket(df,down,slp,tpp,horizon,use_atr=use_atr))
    print(f"{label}")
    print(f"   ALL    : {fmt(sb)}")
    print(f"   SLOPING: {fmt(ss)}   (RANGE除外)")
    print(f"   DOWN   : {fmt(sd)}")
    # half-split for SLOPING
    if ss:
        n=len(df);mid=n//2
        r1=stat(bracket(df,slope,slp,tpp,horizon,0,mid,use_atr=use_atr))
        r2=stat(bracket(df,slope,slp,tpp,horizon,mid,n,use_atr=use_atr))
        if r1 and r2:
            tag="STABLE" if r1['ev']>0 and r2['ev']>0 else "unstable"
            print(f"   SLOPING half-split EV: {r1['ev']:+.2f} / {r2['ev']:+.2f}  [{tag}]")
    print()

# ---- H1 ----
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
h1['atr']=atr(h1['h'].values,h1['l'].values,h1['c'].values)
reg=regime(h1,50)
h=h1['h'].values;l=h1['l'].values;c=h1['c'].values
hr=h1['Date'].dt.hour.values
print("#"*70+"\n# H1  (RANGE除外フィルタ効果)\n"+"#"*70)
for w,nm in [(1,'3bar'),(2,'5bar')]:
    sh,sl=swings_w(h,l,n,w); shi=np.where(sh)[0]
    sw={}
    for i in range(w+1,n):
        ps=shi[shi<i-1]
        if len(ps) and h[i]>h[ps[-1]] and c[i]<h[ps[-1]]: sw[i]=-1
    line(f"swing-high sweep short [{nm}, 1:1 ATR]", h1, sw, 1,1,24, reg, True)
# hour10
h10={i:-1 for i in range(n) if hr[i]==10}
line("hour10(UTC) short [1:1 ATR]", h1, h10, 1,1,24, reg, True)

# ---- M5 ----
m5=load(f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv"); n5=len(m5)
m5['atr']=atr(m5['h'].values,m5['l'].values,m5['c'].values)
reg5=regime(m5,200)
h5=m5['h'].values;l5=m5['l'].values;c5=m5['c'].values
ma50=pd.Series(c5).rolling(50).mean().values; a5=m5['atr'].values
print("#"*70+"\n# M5  (RANGE除外フィルタ効果)\n"+"#"*70)
sh,sl=swings_w(h5,l5,n5,2); shi=np.where(sh)[0]
sw5={}
for i in range(3,n5):
    ps=shi[shi<i-1]
    if len(ps) and h5[i]>h5[ps[-1]] and c5[i]<h5[ps[-1]]: sw5[i]=-1
line("swing-high sweep short [5bar, 1:1 ATR]", m5, sw5, 1,1,72, reg5, True)
# MA overextension short
ovx={}; prev=False
for i in range(n5):
    if np.isnan(ma50[i]) or np.isnan(a5[i]) or a5[i]<=0: continue
    dev=(c5[i]-ma50[i])/a5[i]
    if dev>=2.5 and not prev: ovx[i]=-1
    prev=dev>=2.5
line("MA50 +2.5ATR overextension short [3pt/10pt]", m5, ovx, 3,10,72, reg5, False)
