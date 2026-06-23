"""
コンフルエンス検証 — ダウ構造 × MA200 × フィボ
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

def load(p):
    df=pd.read_csv(p)
    df['Date']=pd.to_datetime(df['Date'],format='%Y.%m.%d %H:%M')
    df=df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
    return df

def calc_atr(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    a=np.full(len(c),np.nan); a[1:]=pd.Series(tr).rolling(p).mean().values; return a

h1=load(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv"); n=len(h1)
o=h1['o'].values; h=h1['h'].values; l=h1['l'].values; c=h1['c'].values
at=calc_atr(h,l,c)
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

sh_arr=np.zeros(n,bool); sl_arr=np.zeros(n,bool)
for i in range(2,n-2):
    if all(h[i]>=h[i-j] for j in range(1,3)) and all(h[i]>=h[i+j] for j in range(1,3)): sh_arr[i]=True
    if all(l[i]<=l[i-j] for j in range(1,3)) and all(l[i]<=l[i+j] for j in range(1,3)): sl_arr[i]=True
shi=np.where(sh_arr)[0]; sli=np.where(sl_arr)[0]

def eval_trade(entry_bar, direction, rr=1.0, hold=24):
    if entry_bar+1>=n: return None
    entry=o[entry_bar+1]; rng=at[entry_bar]
    if np.isnan(rng) or rng<=0: return None
    tp = entry-rng*rr if direction=='short' else entry+rng*rr
    sl = entry+rng    if direction=='short' else entry-rng
    for k in range(entry_bar+1, min(n, entry_bar+1+hold)):
        if direction=='short':
            if h[k]>=sl: return -rng-SPREAD
            if l[k]<=tp: return  rng*rr-SPREAD
        else:
            if l[k]<=sl: return -rng-SPREAD
            if h[k]>=tp: return  rng*rr-SPREAD
    return None

def report(name, pnls):
    if not pnls:
        print(f"  {name:72s} n=0"); return
    n_=len(pnls); wins=sum(1 for p in pnls if p>0)
    wr=100*wins/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    ev1=np.mean(h1p) if h1p else 0; ev2=np.mean(h2p) if h2p else 0
    stable=ev1>0 and ev2>0
    flag='✅STABLE' if stable else '❌'
    if stable and ev>5 and n_>=20: flag+='🔥'
    print(f"  {name:72s} WR={wr:4.0f}% EV={ev:+6.2f} n={n_:4d}  [{ev1:+.1f}/{ev2:+.1f}]  {flag}")

# ダウ構造フラグ
dow_down=np.zeros(n,bool)
dow_up  =np.zeros(n,bool)
for i in range(10,n):
    psh=shi[shi<i]; psl=sli[sli<i]
    if len(psh)<2 or len(psl)<2: continue
    sh1=h[psh[-2]]; sh2=h[psh[-1]]
    sl1=l[psl[-2]]; sl2=l[psl[-1]]
    if sh2<sh1 and sl2<sl1: dow_down[i]=True
    if sh2>sh1 and sl2>sl1: dow_up[i]=True

# フィボゾーン (ショート用: 下降SH→SLからの戻り率)
fib_s=np.zeros(n,float)
for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if not len(psh) or not len(psl): continue
    last_sh_idx=psh[-1]; last_sl_idx=psl[-1]
    if last_sh_idx<=last_sl_idx: continue  # 下降スイングのみ
    top=h[last_sh_idx]; bot=l[last_sl_idx]
    if top<=bot: continue
    retrace=(c[i]-bot)/(top-bot)
    for fv in [0.382,0.500,0.618,0.786]:
        if abs(retrace-fv)<=0.06:
            fib_s[i]=fv; break

# フィボゾーン (ロング用: 上昇SL→SHからの押し率)
fib_l=np.zeros(n,float)
for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi[shi<i]; psl=sli[sli<i]
    if not len(psh) or not len(psl): continue
    last_sh_idx=psh[-1]; last_sl_idx=psl[-1]
    if last_sl_idx<=last_sh_idx: continue  # 上昇スイングのみ
    pre_sh=psh[psh<last_sl_idx]
    if not len(pre_sh): continue
    top=h[pre_sh[-1]]; bot=l[last_sl_idx]
    if top<=bot: continue
    retrace=(top-c[i])/(top-bot)
    for fv in [0.382,0.500,0.618,0.786]:
        if abs(retrace-fv)<=0.06:
            fib_l[i]=fv; break

print("="*100)
print("# コンフルエンス — 高安切り下げ × MA200 × フィボ")
print("="*100)

print("\n### ベースライン ###")
base_slop=[eval_trade(i,'short') for i in range(10,n-1) if reg[i] in ('UP','DOWN')]
base_slop=[p for p in base_slop if p is not None]
base_dow =[eval_trade(i,'short') for i in range(10,n-1) if dow_down[i]]
base_dow =[p for p in base_dow  if p is not None]
base_both=[eval_trade(i,'short') for i in range(10,n-1) if reg[i] in ('UP','DOWN') and dow_down[i]]
base_both=[p for p in base_both if p is not None]
report("SLOPING毎足ショート (ベータ基準)", base_slop)
report("ダウ切り下げ毎足ショート", base_dow)
report("SLOPING + ダウ切り下げ毎足ショート", base_both)

print("\n### フィボ × フィルター (ショート) ###")
for fv in [0.382,0.500,0.618,0.786]:
    raw=[]; sl=[]; dd=[]; ma=[]; sl_dd=[]; sl_ma=[]; dd_ma=[]; all3=[]
    for i in range(10,n-1):
        if fib_s[i]!=fv: continue
        p=eval_trade(i,'short')
        if p is None: continue
        raw.append(p)
        is_sl = reg[i] in ('UP','DOWN')
        is_dd = dow_down[i]
        is_ma = not np.isnan(ma200[i]) and c[i]<ma200[i]
        if is_sl:           sl.append(p)
        if is_dd:           dd.append(p)
        if is_ma:           ma.append(p)
        if is_sl and is_dd: sl_dd.append(p)
        if is_sl and is_ma: sl_ma.append(p)
        if is_dd and is_ma: dd_ma.append(p)
        if is_sl and is_dd and is_ma: all3.append(p)
    print(f"\n  ─ Fib{fv:.1%}戻り ─")
    report(f"素", raw)
    report(f"+ SLOPING", sl)
    report(f"+ ダウ切り下げ", dd)
    report(f"+ MA200下", ma)
    report(f"+ SLOPING × ダウ切り下げ", sl_dd)
    report(f"+ SLOPING × MA200下", sl_ma)
    report(f"+ ダウ切り下げ × MA200下", dd_ma)
    report(f"+ SLOPING × ダウ切り下げ × MA200下 [全部]", all3)

print("\n### フィボ × フィルター (ロング) ###")
for fv in [0.382,0.500,0.618]:
    raw=[]; sl=[]; du=[]; ma=[]; sl_du=[]; all3=[]
    for i in range(10,n-1):
        if fib_l[i]!=fv: continue
        p=eval_trade(i,'long')
        if p is None: continue
        raw.append(p)
        is_sl = reg[i] in ('UP','DOWN')
        is_du = dow_up[i]
        is_ma = not np.isnan(ma200[i]) and c[i]>ma200[i]
        if is_sl: sl.append(p)
        if is_du: du.append(p)
        if is_ma: ma.append(p)
        if is_sl and is_du: sl_du.append(p)
        if is_sl and is_du and is_ma: all3.append(p)
    print(f"\n  ─ Fib{fv:.1%}押し (ロング) ─")
    report(f"素", raw)
    report(f"+ SLOPING", sl)
    report(f"+ ダウ切り上げ", du)
    report(f"+ MA200上", ma)
    report(f"+ SLOPING × ダウ切り上げ", sl_du)
    report(f"+ SLOPING × ダウ切り上げ × MA200上 [全部]", all3)

print("\n" + "="*100)
