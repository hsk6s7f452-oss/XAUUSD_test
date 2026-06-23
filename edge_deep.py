"""
エッジ深掘り — 既存STABLE × 新フィルター
1. 時間フィルター (UTC07-19のみ = ロンドン+NY)
2. ローソク足の実体フィルター (拒絶の強さ)
3. RSIフィルター (50/60超えでショート)
4. セッション × 旗艦スイープ
5. 水曜 × 各エッジ
6. 大陰線エッジの拡張
7. 連続下落 N本後のショート
8. Fib78.6% × 時間 × 水曜
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
hour=df['Date'].dt.hour.values
dow =df['Date'].dt.dayofweek.values  # 0=Mon, 2=Wed

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

# RSI
def calc_rsi(c,p=14):
    delta=np.diff(c); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=pd.Series(gain).ewm(alpha=1/p,adjust=False).mean().values
    al=pd.Series(loss).ewm(alpha=1/p,adjust=False).mean().values
    rs=np.where(al==0,100,ag/(al+1e-9))
    rsi=np.full(n,np.nan); rsi[1:]=100-100/(1+rs)
    return rsi
rsi=calc_rsi(c)

# スイング w=1
sh_arr=np.zeros(n,bool); sl_arr=np.zeros(n,bool)
for i in range(1,n-1):
    if h[i]==h[i-1:i+2].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh_arr[i]=True
    if l[i]==l[i-1:i+2].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl_arr[i]=True
shi=np.where(sh_arr)[0]; sli=np.where(sl_arr)[0]

# スイング w=2
sh2_arr=np.zeros(n,bool); sl2_arr=np.zeros(n,bool)
for i in range(2,n-2):
    if all(h[i]>=h[i-j] for j in range(1,3)) and all(h[i]>=h[i+j] for j in range(1,3)): sh2_arr[i]=True
    if all(l[i]<=l[i-j] for j in range(1,3)) and all(l[i]<=l[i+j] for j in range(1,3)): sl2_arr[i]=True
shi2=np.where(sh2_arr)[0]; sli2=np.where(sl2_arr)[0]

def trade(entry_bar, direction='short', rr=1.0, hold=24):
    if entry_bar+1>=n: return None
    e=o[entry_bar+1]; r=at[entry_bar]
    if np.isnan(r) or r<=0: return None
    tp=e-r*rr if direction=='short' else e+r*rr
    sl=e+r    if direction=='short' else e-r
    for k in range(entry_bar+1, min(n,entry_bar+1+hold)):
        if direction=='short':
            if h[k]>=sl: return -r-SPREAD
            if l[k]<=tp: return  r*rr-SPREAD
        else:
            if l[k]<=sl: return -r-SPREAD
            if h[k]>=tp: return  r*rr-SPREAD
    return None

def rep(name, pnls, indent=2):
    sp=' '*indent
    if not pnls: print(f"{sp}{name:68s} n=0"); return False
    n_=len(pnls); wr=100*sum(1 for p in pnls if p>0)/n_; ev=np.mean(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    ok=e1>0 and e2>0
    flag='✅STABLE' if ok else '❌'
    if ok and ev>5 and n_>=15: flag+='🔥'
    if ok and ev>10 and n_>=20: flag+='🔥'
    print(f"{sp}{name:68s} WR={wr:4.0f}% EV={ev:+6.2f} n={n_:4d} [{e1:+.1f}/{e2:+.1f}] {flag}")
    return ok

# ─────────────────────────────────────────────────────────────
# 1. 旗艦エッジ(SH resist reject+MA50上+SLOPING) × 時間・曜日
# ─────────────────────────────────────────────────────────────
print("="*100)
print("# 1. 旗艦エッジ × 時間フィルター / 曜日")
print("="*100)

flagship=[]
touch_count={}
for i in range(5,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    tol=0.20*at[i]
    past_sh=shi[shi<i-1]
    if not len(past_sh): continue
    res=h[past_sh]; res=res[res>c[i-1]]
    if not len(res): continue
    lv=res.min(); key=round(lv,1)
    if abs(h[i]-lv)<=tol or (h[i]>lv-tol and h[i]<lv+tol):
        touch_count[key]=touch_count.get(key,0)+1
        if c[i]<lv and reg[i] in ('UP','DOWN') and not np.isnan(ma50[i]) and c[i]>ma50[i]:
            flagship.append(i)

fl_all =[trade(i) for i in flagship]; fl_all=[p for p in fl_all if p is not None]
fl_day =[trade(i) for i in flagship if 7<=hour[i]<=19]; fl_day=[p for p in fl_day if p is not None]
fl_wed =[trade(i) for i in flagship if dow[i]==2];      fl_wed=[p for p in fl_wed if p is not None]
fl_wed_day=[trade(i) for i in flagship if dow[i]==2 and 7<=hour[i]<=19]
fl_wed_day=[p for p in fl_wed_day if p is not None]
fl_lon=[trade(i) for i in flagship if 7<=hour[i]<=12]
fl_lon=[p for p in fl_lon if p is not None]
fl_ny =[trade(i) for i in flagship if 13<=hour[i]<=19]
fl_ny =[p for p in fl_ny  if p is not None]
fl_rsi=[trade(i) for i in flagship if not np.isnan(rsi[i]) and rsi[i]>55]
fl_rsi=[p for p in fl_rsi if p is not None]

rep("旗艦 (既存ベース)", fl_all)
rep("旗艦 + 時間UTC07-19", fl_day)
rep("旗艦 + ロンドン(07-12)", fl_lon)
rep("旗艦 + NY(13-19)", fl_ny)
rep("旗艦 + 水曜", fl_wed)
rep("旗艦 + 水曜 + UTC07-19", fl_wed_day)
rep("旗艦 + RSI>55", fl_rsi)

# ─────────────────────────────────────────────────────────────
# 2. Fib78.6% × 時間・曜日
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 2. Fib78.6%戻りショート × 時間フィルター / 曜日")
print("="*100)

fib786=[]
for i in range(10,n):
    if np.isnan(at[i]) or at[i]<=0: continue
    psh=shi2[shi2<i]; psl=sli2[sli2<i]
    if not len(psh) or not len(psl): continue
    last_sh=psh[-1]; last_sl=psl[-1]
    if last_sh<=last_sl: continue
    top=h[last_sh]; bot=l[last_sl]
    if top<=bot: continue
    retrace=(c[i]-bot)/(top-bot)
    if abs(retrace-0.786)<=0.06 and reg[i] in ('UP','DOWN'):
        fib786.append(i)

f786_all=[trade(i) for i in fib786]; f786_all=[p for p in f786_all if p is not None]
f786_day=[trade(i) for i in fib786 if 7<=hour[i]<=19]; f786_day=[p for p in f786_day if p is not None]
f786_wed=[trade(i) for i in fib786 if dow[i]==2];      f786_wed=[p for p in f786_wed if p is not None]
f786_wd =[trade(i) for i in fib786 if dow[i]==2 and 7<=hour[i]<=19]
f786_wd =[p for p in f786_wd if p is not None]
f786_rsi=[trade(i) for i in fib786 if not np.isnan(rsi[i]) and rsi[i]>55]
f786_rsi=[p for p in f786_rsi if p is not None]
f786_dow=[trade(i) for i in fib786 if not np.isnan(ma200[i]) and c[i]<ma200[i]]
f786_dow=[p for p in f786_dow if p is not None]

rep("Fib78.6%+SLOPING (既存ベース)", f786_all)
rep("Fib78.6% + UTC07-19", f786_day)
rep("Fib78.6% + ロンドン(07-12)", [trade(i) for i in fib786 if 7<=hour[i]<=12 if trade(i) is not None])
rep("Fib78.6% + NY(13-19)",       [trade(i) for i in fib786 if 13<=hour[i]<=19 if trade(i) is not None])
rep("Fib78.6% + 水曜", f786_wed)
rep("Fib78.6% + 水曜 + UTC07-19", f786_wd)
rep("Fib78.6% + RSI>55", f786_rsi)
rep("Fib78.6% + MA200下", f786_dow)

# ─────────────────────────────────────────────────────────────
# 3. 大陰線後の戻りショート 拡張
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 3. 大陰線(≥2ATR)後の戻り — 拡張分析")
print("="*100)

disp_bars=[]
for i in range(1,n-2):
    if np.isnan(at[i]) or at[i]<=0: continue
    body=o[i]-c[i]  # 陰線なら正
    if body>=2*at[i] and reg[i] in ('UP','DOWN'):
        disp_bars.append(i)

def disp_entry(disp_i, window=10, fib_ret=None, require_time=None, require_wed=False):
    results=[]
    for di in disp_i:
        disp_bot=min(o[di],c[di])
        disp_top=max(o[di],c[di])
        disp_range=disp_top-disp_bot
        for j in range(di+1, min(n-1, di+window)):
            if np.isnan(at[j]) or at[j]<=0: continue
            # 戻り条件
            retrace=(c[j]-disp_bot)/disp_range if disp_range>0 else 0
            if fib_ret:
                flo,fhi=fib_ret
                if not (flo<=retrace<=fhi): continue
            else:
                if not (0.2<=retrace<=0.9): continue
            if c[j]>disp_top: continue  # 全戻しNG
            if require_time and not (require_time[0]<=hour[j]<=require_time[1]): continue
            if require_wed and dow[j]!=2: continue
            p=trade(j,'short')
            if p is not None:
                results.append(p); break  # 1陰線につき1エントリー
    return results

d_all   =disp_entry(disp_bars)
d_day   =disp_entry(disp_bars, require_time=(7,19))
d_fib50 =disp_entry(disp_bars, fib_ret=(0.45,0.65))
d_fib618=disp_entry(disp_bars, fib_ret=(0.55,0.70))
d_fib786=disp_entry(disp_bars, fib_ret=(0.70,0.85))
d_lon   =disp_entry(disp_bars, require_time=(7,12))
d_ny    =disp_entry(disp_bars, require_time=(13,19))

rep("大陰線後の戻りショート (既存ベース)", d_all)
rep("大陰線後 + UTC07-19", d_day)
rep("大陰線後 + ロンドン(07-12)", d_lon)
rep("大陰線後 + NY(13-19)", d_ny)
rep("大陰線後 + Fib50%戻り", d_fib50)
rep("大陰線後 + Fib61.8%戻り", d_fib618)
rep("大陰線後 + Fib78.6%戻り", d_fib786)

# ─────────────────────────────────────────────────────────────
# 4. ローソク足パターン × 旗艦
# (上ヒゲの長さ・実体の位置で拒絶の強さを測る)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 4. 拒絶ローソク足の強さフィルター")
print("="*100)

fl_strong_rej=[]  # 上ヒゲが実体の2倍以上 (強い拒絶)
fl_close_low =[]  # 終値が足の下半分 (弱気終値)
fl_both      =[]

for i in flagship:
    body=abs(c[i]-o[i])
    upper_wick=h[i]-max(c[i],o[i])
    lower_wick=min(c[i],o[i])-l[i]
    total_range=h[i]-l[i]
    close_pct=(c[i]-l[i])/total_range if total_range>0 else 0.5  # 0=底, 1=天井

    strong_rej = upper_wick >= body*1.5 or upper_wick >= at[i]*0.5
    close_low  = close_pct <= 0.40  # 足の下40%で終値

    p=trade(i)
    if p is None: continue
    if strong_rej: fl_strong_rej.append(p)
    if close_low:  fl_close_low.append(p)
    if strong_rej and close_low: fl_both.append(p)

rep("旗艦 + 上ヒゲ強拒絶(ヒゲ≥実体1.5倍)", fl_strong_rej)
rep("旗艦 + 弱気終値(足の下40%)", fl_close_low)
rep("旗艦 + 強拒絶 × 弱気終値", fl_both)

# ─────────────────────────────────────────────────────────────
# 5. 連続下落N本後 × SLOPING (モメンタム確認)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 5. 連続N本下落後のショート (モメンタム)")
print("="*100)

for n_bars in [2,3,4,5]:
    res_all=[]; res_sl=[]; res_sl_day=[]
    for i in range(n_bars,n-1):
        if any(np.isnan(at[i-k]) for k in range(n_bars)): continue
        # 直近n_bars本がすべて陰線
        if not all(c[i-k]<o[i-k] for k in range(n_bars)): continue
        p=trade(i,'short')
        if p is None: continue
        res_all.append(p)
        if reg[i] in ('UP','DOWN'): res_sl.append(p)
        if reg[i] in ('UP','DOWN') and 7<=hour[i]<=19: res_sl_day.append(p)
    rep(f"連続{n_bars}本陰線 + SLOPING", res_sl)
    rep(f"連続{n_bars}本陰線 + SLOPING + UTC07-19", res_sl_day)

# ─────────────────────────────────────────────────────────────
# 6. 最強エッジ同士の掛け合わせ
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 6. 最強エッジ 掛け合わせ")
print("="*100)

# 水曜 × Fib78.6%
res=[]; 
for i in fib786:
    if dow[i]!=2: continue
    p=trade(i)
    if p is not None: res.append(p)
rep("水曜 × Fib78.6%+SLOPING", res)

# 旗艦 × Fib78.6% (同バーに両方ヒット)
fl_set=set(flagship); f786_set=set(fib786)
combo=fl_set & f786_set
res=[p for i in sorted(combo) for p in [trade(i)] if p is not None]
rep("旗艦 × Fib78.6% (同時ヒット)", res)

# 水曜 × 旗艦 × UTC07-19
res=[trade(i) for i in flagship if dow[i]==2 and 7<=hour[i]<=19]
res=[p for p in res if p is not None]
rep("水曜 × 旗艦 × UTC07-19", res)

# 水曜 × Fib78.6% × UTC07-19
res=[trade(i) for i in fib786 if dow[i]==2 and 7<=hour[i]<=19]
res=[p for p in res if p is not None]
rep("水曜 × Fib78.6% × UTC07-19", res)

# 旗艦 × UTC07-19 × RSI>55
res=[trade(i) for i in flagship if 7<=hour[i]<=19 and not np.isnan(rsi[i]) and rsi[i]>55]
res=[p for p in res if p is not None]
rep("旗艦 × UTC07-19 × RSI>55", res)

# ─────────────────────────────────────────────────────────────
# 7. RSI × 相場環境
# ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("# 7. RSI 単体 / 組み合わせ")
print("="*100)

for rsi_th in [50,55,60,65,70]:
    res_s=[]; res_l=[]
    for i in range(20,n-1):
        if np.isnan(rsi[i]) or np.isnan(at[i]) or at[i]<=0: continue
        if reg[i] not in ('UP','DOWN'): continue
        if rsi[i]>rsi_th:
            p=trade(i,'short')
            if p is not None: res_s.append(p)
        if rsi[i]<(100-rsi_th):
            p=trade(i,'long')
            if p is not None: res_l.append(p)
    rep(f"RSI>{rsi_th} → ショート + SLOPING", res_s)
    rep(f"RSI<{100-rsi_th} → ロング + SLOPING", res_l)

print("\n" + "="*100)
print("# まとめ — STABLEのみ抽出")
print("="*100)
