"""
WFA Part2 — 新エッジ (Fib78.6%, RSI>70, ATR中ボラ×旗艦, hour10, 水曜+旗艦)
IS=600bar, OOS=200bar, step=200bar (ローリング)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

df=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
df=df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
n=len(df)
o=df['o'].values; h=df['h'].values; l=df['l'].values
c=df['c'].values; v=df['v'].values
hour=df['Date'].dt.hour.values; dow=df['Date'].dt.dayofweek.values

tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
at=np.full(n,np.nan); at[1:]=pd.Series(tr).rolling(14).mean().values
ma200=pd.Series(c).rolling(200).mean().values
ma50 =pd.Series(c).rolling(50).mean().values

slope_k=50
reg=np.array(['NA']*n,dtype=object)
for i in range(n):
    if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
    s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
    if   s> 0.003 and c[i]>ma200[i]: reg[i]='UP'
    elif s<-0.003 and c[i]<ma200[i]: reg[i]='DOWN'
    else: reg[i]='RANGE'

def calc_rsi(c,p=14):
    delta=np.diff(c); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=pd.Series(gain).ewm(alpha=1/p,adjust=False).mean().values
    al=pd.Series(loss).ewm(alpha=1/p,adjust=False).mean().values
    rsi=np.full(n,np.nan); rsi[1:]=100-100/(1+ag/(al+1e-9)); return rsi
rsi=calc_rsi(c)

sh1=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh1[i]=True
shi1=np.where(sh1)[0]

sh2=np.zeros(n,bool); sl2=np.zeros(n,bool)
for i in range(2,n-2):
    if all(h[i]>=h[i-j] for j in range(1,3)) and all(h[i]>=h[i+j] for j in range(1,3)): sh2[i]=True
    if all(l[i]<=l[i-j] for j in range(1,3)) and all(l[i]<=l[i+j] for j in range(1,3)): sl2[i]=True
shi2=np.where(sh2)[0]; sli2=np.where(sl2)[0]

atr_pct=np.full(n,np.nan)
for i in range(200,n):
    w=at[i-200:i]; w=w[~np.isnan(w)]
    if len(w)>0: atr_pct[i]=np.sum(w<=at[i])/len(w)

def trade(i, direction='short'):
    if i+1>=n: return None
    e=o[i+1]; r=at[i]
    if np.isnan(r) or r<=0: return None
    tp=e-r if direction=='short' else e+r
    sl=e+r if direction=='short' else e-r
    for k in range(i+1,min(n,i+25)):
        if direction=='short':
            if h[k]>=sl: return -r-SPREAD
            if l[k]<=tp: return  r-SPREAD
        else:
            if l[k]<=sl: return -r-SPREAD
            if h[k]>=tp: return  r-SPREAD
    return None

# 旗艦バー
flagship=[]
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]; psh=shi1[shi1<i-1]
    if not len(psh): continue
    res=h[psh]; res=res[res>c[i-1]]
    if not len(res): continue
    lv=res.min()
    if abs(h[i]-lv)<=tol or (h[i]>lv-tol and h[i]<lv+tol):
        if c[i]<lv and reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
            flagship.append(i)
fl=set(flagship)

# Fib78.6%バー
fib786=[]
for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi2[shi2<i]; psl=sli2[sli2<i]
    if not len(psh) or not len(psl): continue
    lsh=psh[-1]; lsl=psl[-1]
    if lsh<=lsl: continue
    top=h[lsh]; bot=l[lsl]
    if top<=bot: continue
    if abs((c[i]-bot)/(top-bot)-0.786)<=0.06 and reg[i] in ('UP','DOWN'):
        fib786.append(i)
f786=set(fib786)

# エッジ定義 {name: condition_fn}
edges = {
    "E1 Fib78.6%+SLOPING":        lambda i: i in f786,
    "E2 Fib78.6%+UTC07-19":       lambda i: i in f786 and 7<=hour[i]<=19,
    "E3 Fib78.6%+MA200下":        lambda i: i in f786 and not np.isnan(ma200[i]) and c[i]<ma200[i],
    "E4 RSI>70+SLOPING":           lambda i: reg[i] in ('UP','DOWN') and not np.isnan(rsi[i]) and rsi[i]>70,
    "E5 旗艦×ATR中ボラ":           lambda i: i in fl and not np.isnan(atr_pct[i]) and 0.50<=atr_pct[i]<=0.80,
    "E6 hour10+SLOPING":           lambda i: hour[i]==10 and reg[i] in ('UP','DOWN'),
    "E7 水曜+旗艦+UTC07-19":      lambda i: i in fl and dow[i]==2 and 7<=hour[i]<=19,
    "E8 旗艦+ロンドン(07-12)":    lambda i: i in fl and 7<=hour[i]<=12,
}

IS=600; OOS=200; STEP=200

print("="*105)
print("# WFA Part2 — 新エッジ ウォークフォワード検証")
print(f"  IS={IS}bar / OOS={OOS}bar / STEP={STEP}bar (ローリング)")
print("="*105)

results={}
for name, cond in edges.items():
    windows=[]
    start=0
    while start+IS+OOS<=n:
        is_end=start+IS; oos_end=start+IS+OOS
        # IS
        is_p=[trade(i) for i in range(start,is_end) if cond(i)]
        is_p=[p for p in is_p if p is not None]
        # OOS
        oos_p=[trade(i) for i in range(is_end,oos_end) if cond(i)]
        oos_p=[p for p in oos_p if p is not None]
        if is_p and oos_p:
            windows.append((np.mean(is_p), np.mean(oos_p), len(is_p), len(oos_p)))
        start+=STEP

    if not windows:
        print(f"  {name:30s}  窓なし"); continue

    is_evs =[w[0] for w in windows]
    oos_evs=[w[1] for w in windows]
    is_ns  =[w[2] for w in windows]
    oos_ns =[w[3] for w in windows]

    mean_is =np.mean(is_evs)
    mean_oos=np.mean(oos_evs)
    eff = mean_oos/mean_is if mean_is!=0 else 0
    oos_pos=sum(1 for e in oos_evs if e>0)
    oos_wr =100*oos_pos/len(oos_evs)

    # 有意性: OOS EVの平均が0より上かt検定
    from scipy import stats as scs
    if len(oos_evs)>=3:
        t,p=scs.ttest_1samp(oos_evs,0)
        p_str=f"p={p:.3f}"
    else:
        p_str="(窓少)"

    verdict = "✅実戦" if oos_wr>=60 and mean_oos>0 else ("△継続" if mean_oos>0 else "❌棄却")
    if oos_wr>=70 and mean_oos>2: verdict+="🔥"

    print(f"\n  {name}")
    print(f"    窓数={len(windows)}  IS平均EV={mean_is:+.2f}  OOS平均EV={mean_oos:+.2f}  "
          f"効率比={eff:.2f}  OOS勝率={oos_wr:.0f}%  {p_str}  → {verdict}")
    print(f"    OOS内訳: {[f'{e:+.1f}(n={nn})' for e,nn in zip(oos_evs,oos_ns)]}")
    results[name]=(mean_oos, oos_wr, verdict)

print("\n" + "="*105)
print("# WFA まとめ")
print("="*105)
print(f"  {'エッジ':30s}  OOS_EV  OOS勝率  判定")
print("-"*70)
for name,(ev,wr,v) in sorted(results.items(), key=lambda x:-x[1][0]):
    print(f"  {name:30s}  {ev:+6.2f}  {wr:5.0f}%   {v}")
