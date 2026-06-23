"""
BROAD EDGE SCREEN — XAUUSD
Test many hypotheses, no look-ahead, triple-barrier 1:1 ATR evaluation.
Entry = next-bar open - spread(0.3 for long, +0.3 for short cost).
Win = TP(+1ATR) touched before SL(-1ATR) within horizon.
Report WR / EV(pt) / n / MaxConsecLoss for everything with n>=20.
"""
import pandas as pd, numpy as np, warnings, calendar
warnings.filterwarnings('ignore')
UP = "/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD = 0.3

def load(p):
    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'}, inplace=True)
    return df

def atr(h,l,c,p=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    a = np.full(len(c), np.nan)
    a[1:] = pd.Series(tr).rolling(p).mean().values
    return a

def swings(h,l,n):
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(2,n-2):
        if h[i]==h[i-2:i+3].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh[i]=True
        if l[i]==l[i-2:i+3].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl[i]=True
    return sh,sl

def evaluate(df, signals, horizon, atr_mult=1.0):
    """signals: dict sig_bar -> +1 long / -1 short. Entry next bar open. 1:1 ATR bracket."""
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values; a=df['atr'].values
    n=len(df); results=[]
    for i,side in signals.items():
        if i+1>=n or np.isnan(a[i]): continue
        entry=o[i+1] + (SPREAD if side>0 else -SPREAD)*0  # spread applied below as cost
        rng=a[i]*atr_mult
        if rng<=0: continue
        if side>0:
            tp=entry+rng; sl=entry-rng
        else:
            tp=entry-rng; sl=entry+rng
        win=None
        end=min(n, i+1+horizon)
        for k in range(i+1,end):
            hi=h[k]; lo=l[k]
            if side>0:
                if lo<=sl: win=False; break
                if hi>=tp: win=True; break
            else:
                if hi>=sl: win=False; break
                if lo<=tp: win=True; break
        if win is None:
            # close at horizon end
            exitp=c[end-1]
            pnl=(exitp-entry) if side>0 else (entry-exitp)
            win = pnl>0
            pnl-=SPREAD
        else:
            pnl=(rng-SPREAD) if win else (-rng-SPREAD)
        results.append((win,pnl))
    return results

def stats(res):
    if len(res)<20: return None
    wins=[r[0] for r in res]; pnls=[r[1] for r in res]
    wr=100*np.mean(wins); ev=np.mean(pnls); n=len(res)
    mcl=0; cur=0
    for w in wins:
        if not w: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    return dict(wr=wr,ev=ev,n=n,mcl=mcl,tot=sum(pnls))

REPORT=[]
def test(name, df, signals, horizon, atr_mult=1.0):
    res=evaluate(df,signals,horizon,atr_mult)
    s=stats(res)
    if s: REPORT.append((name,s))

# ════════════════════════════════════════════════════════════════════
def run_tf(df, tfname, horizon):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    df['atr']=atr(h,l,c)
    a=df['atr'].values
    sh,sl=swings(h,l,n)
    shi=np.where(sh)[0]; sli=np.where(sl)[0]
    hr=df['Date'].dt.hour.values
    dow=df['Date'].dt.dayofweek.values
    ma20=pd.Series(c).rolling(20).mean().values
    ma50=pd.Series(c).rolling(50).mean().values
    ma200=pd.Series(c).rolling(200).mean().values
    rng_hi=pd.Series(h).rolling(50).max().values
    rng_lo=pd.Series(l).rolling(50).min().values

    # Causal MSB + OB + FVG (signal available at bar i, using only <=i data)
    msb_bull={}; msb_bear={}; ob_bull_tap=[]; ob_bear_tap=[]
    lsh=None; lsl=None
    obs_bull=[]; obs_bear=[]  # list of (lo,hi,formed_bar)
    for i in range(n):
        # confirmed swings available at i (i-2 was pivot)
        if i>=2 and sh[i-2]: lsh=h[i-2]
        if i>=2 and sl[i-2]: lsl=l[i-2]
        if lsh is not None and c[i]>lsh and c[i-1]<=lsh:
            msb_bull[i]=1
            for j in range(i-1,max(0,i-30)-1,-1):
                if c[j]<o[j]: obs_bull.append((l[j],h[j],i)); break
            lsh=None
        if lsl is not None and c[i]<lsl and c[i-1]>=lsl:
            msb_bear[i]=-1
            for j in range(i-1,max(0,i-30)-1,-1):
                if c[j]>o[j]: obs_bear.append((l[j],h[j],i)); break
            lsl=None

    # OB tap entries (price returns into OB zone after formation; first touch)
    ob_bull_sig={}; ob_bear_sig={}
    for (lo,hi,fb) in obs_bull:
        for k in range(fb+1,min(n,fb+horizon*3)):
            if l[k]<=hi and h[k]>=lo:  # entered zone
                ob_bull_sig[k]=1; break
    for (lo,hi,fb) in obs_bear:
        for k in range(fb+1,min(n,fb+horizon*3)):
            if h[k]>=lo and l[k]<=hi:
                ob_bear_sig[k]=-1; break

    # FVG fill entries (confirmed at i+1, price returns into gap)
    fvg_bull_sig={}; fvg_bear_sig={}
    for i in range(1,n-1):
        gb=l[i+1]-h[i-1]
        if gb>0.5:
            top=l[i+1]; bot=h[i-1]
            for k in range(i+2,min(n,i+2+horizon*3)):
                if l[k]<=top and h[k]>=bot:
                    fvg_bull_sig[k]=1; break
        gs=l[i-1]-h[i+1]
        if gs>0.5:
            top=l[i-1]; bot=h[i+1]
            for k in range(i+2,min(n,i+2+horizon*3)):
                if h[k]>=bot and l[k]<=top:
                    fvg_bear_sig[k]=-1; break

    # Liquidity sweep: break prior swing low intrabar then close back above -> long
    sweep_lo={}; sweep_hi={}
    for i in range(3,n):
        prevsl=[s for s in sli if s<i-1]
        if prevsl:
            lv=l[prevsl[-1]]
            if l[i]<lv and c[i]>lv: sweep_lo[i]=1
        prevsh=[s for s in shi if s<i-1]
        if prevsh:
            lv=h[prevsh[-1]]
            if h[i]>lv and c[i]<lv: sweep_hi[i]=-1

    # ---- TESTS ----
    test(f"{tfname} MSB-bull long", df, msb_bull, horizon)
    test(f"{tfname} MSB-bear short", df, msb_bear, horizon)
    test(f"{tfname} OB-bull tap long", df, ob_bull_sig, horizon)
    test(f"{tfname} OB-bear tap short", df, ob_bear_sig, horizon)
    test(f"{tfname} FVG-bull fill long", df, fvg_bull_sig, horizon)
    test(f"{tfname} FVG-bear fill short", df, fvg_bear_sig, horizon)
    test(f"{tfname} SwingLow sweep long", df, sweep_lo, horizon)
    test(f"{tfname} SwingHigh sweep short", df, sweep_hi, horizon)

    # Sweep + trend filter (only with HTF MA200 alignment)
    sweep_lo_up={i:s for i,s in sweep_lo.items() if not np.isnan(ma200[i]) and c[i]>ma200[i]}
    sweep_hi_dn={i:s for i,s in sweep_hi.items() if not np.isnan(ma200[i]) and c[i]<ma200[i]}
    test(f"{tfname} SwingLow sweep + above MA200 long", df, sweep_lo_up, horizon)
    test(f"{tfname} SwingHigh sweep + below MA200 short", df, sweep_hi_dn, horizon)

    # Sweep in discount/premium (range position)
    def rngpos(i):
        if np.isnan(rng_hi[i]) or rng_hi[i]==rng_lo[i]: return 0.5
        return (c[i]-rng_lo[i])/(rng_hi[i]-rng_lo[i])
    sweep_lo_disc={i:s for i,s in sweep_lo.items() if rngpos(i)<0.4}
    sweep_hi_prem={i:s for i,s in sweep_hi.items() if rngpos(i)>0.6}
    test(f"{tfname} SwingLow sweep @discount long", df, sweep_lo_disc, horizon)
    test(f"{tfname} SwingHigh sweep @premium short", df, sweep_hi_prem, horizon)

    # MSB + trend
    msb_bull_t={i:s for i,s in msb_bull.items() if not np.isnan(ma200[i]) and c[i]>ma200[i]}
    msb_bear_t={i:s for i,s in msb_bear.items() if not np.isnan(ma200[i]) and c[i]<ma200[i]}
    test(f"{tfname} MSB-bull + above MA200 long", df, msb_bull_t, horizon)
    test(f"{tfname} MSB-bear + below MA200 short", df, msb_bear_t, horizon)

    # ---- Simple PA / technical (broad) ----
    # N consecutive same-color reversal
    for ncons in (3,4,5):
        rev_up={}; rev_dn={}
        for i in range(ncons,n):
            if all(c[i-k]<o[i-k] for k in range(ncons)): rev_up[i]=1   # n red -> long
            if all(c[i-k]>o[i-k] for k in range(ncons)): rev_dn[i]=-1  # n green -> short
        test(f"{tfname} {ncons}red reversal long", df, rev_up, horizon)
        test(f"{tfname} {ncons}green reversal short", df, rev_dn, horizon)

    # RSI extremes
    delta=np.diff(c, prepend=c[0])
    up=pd.Series(np.where(delta>0,delta,0)).rolling(14).mean().values
    dn=pd.Series(np.where(delta<0,-delta,0)).rolling(14).mean().values
    rsi=100-100/(1+up/(dn+1e-9))
    rsi_os={}; rsi_ob={}
    for i in range(1,n):
        if rsi[i-1]<30 and rsi[i]>=30: rsi_os[i]=1   # exit oversold -> long
        if rsi[i-1]>70 and rsi[i]<=70: rsi_ob[i]=-1  # exit overbought -> short
    test(f"{tfname} RSI exit oversold long", df, rsi_os, horizon)
    test(f"{tfname} RSI exit overbought short", df, rsi_ob, horizon)

    # MA cross
    ma_x_up={}; ma_x_dn={}
    for i in range(1,n):
        if not np.isnan(ma50[i]):
            if c[i-1]<=ma50[i-1] and c[i]>ma50[i]: ma_x_up[i]=1
            if c[i-1]>=ma50[i-1] and c[i]<ma50[i]: ma_x_dn[i]=-1
    test(f"{tfname} close cross>MA50 long", df, ma_x_up, horizon)
    test(f"{tfname} close cross<MA50 short", df, ma_x_dn, horizon)

    # Big bar (>1.5 ATR) continuation
    bigup={}; bigdn={}
    for i in range(n):
        if np.isnan(a[i]) or a[i]<=0: continue
        body=c[i]-o[i]
        if body>1.5*a[i]: bigup[i]=1
        if body<-1.5*a[i]: bigdn[i]=-1
    test(f"{tfname} bigbar-up continuation long", df, bigup, horizon)
    test(f"{tfname} bigbar-down continuation short", df, bigdn, horizon)
    # Big bar reversal (fade)
    test(f"{tfname} bigbar-up fade short", df, {i:-1 for i in bigup}, horizon)
    test(f"{tfname} bigbar-down fade long", df, {i:1 for i in bigdn}, horizon)

    # Hour-of-day directional bias (long every bar at hour H) - screen which hours trend
    for H in range(0,24):
        sig={i:1 for i in range(n) if hr[i]==H}
        if len(sig)>=30: test(f"{tfname} hour{H:02d} long-bias", df, sig, horizon)
        sig2={i:-1 for i in range(n) if hr[i]==H}
        if len(sig2)>=30: test(f"{tfname} hour{H:02d} short-bias", df, sig2, horizon)

    # Round-number reaction ($50 levels): price within 1pt of x50/x00 -> fade toward mean
    rn_long={}; rn_short={}
    for i in range(1,n):
        nearest=round(c[i]/50)*50
        if abs(c[i]-nearest)<=1.0:
            if c[i]<nearest: rn_short[i]=-1  # approaching from below, fade
            else: rn_long[i]=1
    test(f"{tfname} round# bounce long", df, rn_long, horizon)
    test(f"{tfname} round# reject short", df, rn_short, horizon)

print("Loading...")
h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv")
m5=load(f"{UP}/a8c2b4e5-XAUUSD_M5_Mar_Jun.csv")

print("Screening H1 (horizon 24)..."); run_tf(h1,'H1',24)
print("Screening M15 (horizon 32)..."); run_tf(m15,'M15',32)
print("Screening M5 (horizon 48)..."); run_tf(m5,'M5',48)

# Sort: edges = WR>=55 & EV>0 & n>=20, rank by EV*sqrt(n)
print("\n"+"="*78)
print("ALL RESULTS (n>=20), sorted by win rate")
print("="*78)
rows=sorted(REPORT, key=lambda x:-x[1]['wr'])
print(f"{'EDGE':<46}{'WR%':>6}{'EV':>8}{'n':>6}{'MCL':>5}{'totPt':>9}")
for name,s in rows:
    print(f"{name:<46}{s['wr']:>6.1f}{s['ev']:>8.2f}{s['n']:>6}{s['mcl']:>5}{s['tot']:>9.0f}")

print("\n"+"="*78)
print(">>> EDGES: WR>=55% AND EV>0 (n>=25) <<<")
print("="*78)
edges=[(nm,s) for nm,s in REPORT if s['wr']>=55 and s['ev']>0 and s['n']>=25]
edges.sort(key=lambda x:-x[1]['wr'])
if not edges:
    print("(none cleared WR>=55 & EV>0 & n>=25)")
for name,s in edges:
    print(f"  {name:<44} WR={s['wr']:.1f}%  EV={s['ev']:+.2f}pt  n={s['n']}  MCL={s['mcl']}  tot={s['tot']:+.0f}pt")
