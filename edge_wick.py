"""
WICK RESISTANCE/SUPPORT TEST
H1 and M15 wicks: do upper wicks act as resistance and lower wicks as support?
Method:
  - Wick level = high beyond body (upper wick) or low below body (lower wick)
  - Future price approaches that level within tolerance (0.1 ATR)
  - Measure rejection: short from upper wick touch, long from lower wick touch
  - 1:1 ATR bracket, horizon 24(H1) or 48(M15), RANGE excluded
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

def regime(c, slope_k):
    n=len(c); ma200=pd.Series(c).rolling(200).mean().values
    reg=np.array(['NA']*n,dtype=object)
    for i in range(n):
        if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
        s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
        if s>0.003 and c[i]>ma200[i]: reg[i]='UP'
        elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
        else: reg[i]='RANGE'
    return reg

def run(label, df, horizon, slope_k, min_wick_atr=0.3, tol_atr=0.15, lookback=50):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    a=atr(h,l,c); df['atr']=a
    reg=regime(c, slope_k)
    mon=df['Date'].dt.strftime('%Y-%m').values

    upper_wicks={}  # bar_idx -> wick_top level
    lower_wicks={}  # bar_idx -> wick_bot level
    for i in range(n):
        if np.isnan(a[i]) or a[i]<=0: continue
        body_top=max(o[i],c[i]); body_bot=min(o[i],c[i])
        uw=h[i]-body_top
        lw=body_bot-l[i]
        if uw >= min_wick_atr*a[i]: upper_wicks[i]=h[i]
        if lw >= min_wick_atr*a[i]: lower_wicks[i]=l[i]

    def test_wick_reaction(wicks, side, filter_reg=None):
        """
        For each wick level, scan future bars to see if price approaches within tol_atr,
        then take a trade (short for upper wick, long for lower wick).
        Only first touch per wick. Wick expires after lookback bars.
        """
        results=[]; used=set()
        for src, level in sorted(wicks.items()):
            end=min(n-1, src+lookback)
            for i in range(src+1, end+1):
                if i in used: continue
                if np.isnan(a[i]) or a[i]<=0: continue
                if filter_reg and reg[i] not in filter_reg: continue
                tol=tol_atr*a[i]
                hit=False
                if side<0 and abs(h[i]-level)<=tol: hit=True  # price touched upper wick
                if side>0 and abs(l[i]-level)<=tol: hit=True   # price touched lower wick
                if not hit: continue
                # trade from next bar open
                if i+1>=n: break
                used.add(i)
                entry=o[i+1]; rng=a[i]
                tp=entry+rng*side; slv=entry-rng*side
                win=None
                for k in range(i+1, min(n, i+1+horizon)):
                    if side<0:
                        if h[k]>=slv: win=False; break
                        if l[k]<=tp: win=True; break
                    else:
                        if l[k]<=slv: win=False; break
                        if h[k]>=tp: win=True; break
                if win is None: continue
                pnl=(rng-SPREAD) if win else (-rng-SPREAD)
                results.append((win,pnl,i,mon[i]))
                break  # one trade per wick
        return results

    def report(tag, res, idxs_mon=None):
        if len(res)<10:
            print(f"  {tag:<50} n={len(res)} too few"); return None
        wr=100*np.mean([r[0] for r in res]); ev=np.mean([r[1] for r in res])
        tot=sum(r[1] for r in res)
        mcl=cur=0
        for r in res:
            if not r[0]: cur+=1; mcl=max(mcl,cur)
            else: cur=0
        months=len(set(r[3] for r in res)); tpm=len(res)/max(months,1)
        mid=n//2
        h1_=[r for r in res if r[2]<mid]
        h2_=[r for r in res if r[2]>=mid]
        def hw(x): return f"{100*np.mean([r[0] for r in x]):.0f}%/{len(x)}" if len(x)>=8 else "-"
        stable="STABLE" if len(h1_)>=8 and len(h2_)>=8 and np.mean([r[1] for r in h1_])>0 and np.mean([r[1] for r in h2_])>0 else ""
        print(f"  {tag:<50} WR={wr:.0f}% EV={ev:+.2f} n={len(res):4d} tot={tot:+.0f} MCL={mcl} ~{tpm:.0f}/mo  half={hw(h1_)},{hw(h2_)}  {stable}")
        return res

    print(f"\n{'='*90}")
    print(f"# {label}  (wick>={min_wick_atr}ATR, tol={tol_atr}ATR, lookback={lookback}bars, horizon={horizon}bars)")
    print(f"{'='*90}")

    r_uw_all=test_wick_reaction(upper_wicks, -1)
    r_lw_all=test_wick_reaction(lower_wicks, +1)
    r_uw_slope=test_wick_reaction(upper_wicks, -1, ('UP','DOWN'))
    r_lw_slope=test_wick_reaction(lower_wicks, +1, ('UP','DOWN'))
    r_uw_down=test_wick_reaction(upper_wicks, -1, ('DOWN',))
    r_lw_up=test_wick_reaction(lower_wicks, +1, ('UP',))

    print(f"\n  Upper wick → SHORT (resistance test):")
    report("ALL regimes", r_uw_all)
    report("SLOPING only (no RANGE)", r_uw_slope)
    report("DOWN regime only", r_uw_down)

    print(f"\n  Lower wick → LONG (support test):")
    report("ALL regimes", r_lw_all)
    report("SLOPING only (no RANGE)", r_lw_slope)
    report("UP regime only", r_lw_up)

    # Sensitivity: wick size threshold
    print(f"\n  Sensitivity — upper wick size threshold (SLOPING):")
    for thresh in [0.2, 0.3, 0.5, 0.7]:
        uw2={i:v for i,v in upper_wicks.items() if (h[i]-max(o[i],c[i]))>=thresh*a[i]}
        r=test_wick_reaction(uw2, -1, ('UP','DOWN'))
        tag=f"upper wick>={thresh}ATR short SLOPING"
        report(tag, r)

print("WICK AS RESISTANCE/SUPPORT ANALYSIS")
print("="*90)

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
run("H1", h1, horizon=24, slope_k=50)

m15=load(f"{UP}/df162149-XAUUSD_M15_Mar_Jun.csv")
run("M15", m15, horizon=48, slope_k=200)
