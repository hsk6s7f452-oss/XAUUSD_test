"""
前日POC/VAH/VAL スキャルピング検証
固定TP/SL (pt単位) でM1スキャルを想定
TP=3,5,8,10,15pt × SL=3,5,8,10,15pt の全組み合わせ
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

# M1
m1=pd.read_csv(f"{UP}/9c76cb47-XAUUSD_M1_Mar_Jun.csv")
m1['Date']=pd.to_datetime(m1['Date'],format='%Y.%m.%d %H:%M')
m1=m1.sort_values('Date').reset_index(drop=True)
m1.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
m1['date_only']=m1['Date'].dt.date

# H1 SLOPING
h1=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
h1['Date']=pd.to_datetime(h1['Date'],format='%Y.%m.%d %H:%M')
h1=h1.sort_values('Date').reset_index(drop=True)
h1.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
ch=h1['c'].values; hh=h1['h'].values; lh=h1['l'].values
nh=len(h1)
tr=np.maximum(hh[1:]-lh[1:],np.maximum(np.abs(hh[1:]-ch[:-1]),np.abs(lh[1:]-ch[:-1])))
at=np.full(nh,np.nan); at[1:]=pd.Series(tr).rolling(14).mean().values
ma200=pd.Series(ch).rolling(200).mean().values
slope_k=50
reg=np.array(['NA']*nh,dtype=object)
for i in range(nh):
    if i<200+slope_k or np.isnan(ma200[i]) or np.isnan(ma200[i-slope_k]): continue
    s=(ma200[i]-ma200[i-slope_k])/ma200[i-slope_k]
    if   s> 0.003 and ch[i]>ma200[i]: reg[i]='UP'
    elif s<-0.003 and ch[i]<ma200[i]: reg[i]='DOWN'
    else: reg[i]='RANGE'
h1['date_only']=h1['Date'].dt.date
h1_reg={}
for i in range(nh): h1_reg[h1['date_only'].iloc[i]]=reg[i]

# VP計算
def calc_vp(df_day, n_bins=120):
    lo=df_day['l'].min(); hi=df_day['h'].max()
    if hi<=lo or hi-lo<0.5: return None,None,None
    bins=np.linspace(lo,hi,n_bins+1)
    bin_vol=np.zeros(n_bins)
    for _,row in df_day.iterrows():
        mask=(bins[1:]>=row['l'])&(bins[:-1]<=row['h'])
        span=max(mask.sum(),1)
        bin_vol[mask]+=row['v']/span
    poc_bin=np.argmax(bin_vol)
    poc=(bins[poc_bin]+bins[poc_bin+1])/2
    total_vol=bin_vol.sum(); target=total_vol*0.70
    up=poc_bin; dn=poc_bin; va_vol=bin_vol[poc_bin]
    while va_vol<target:
        up_v=bin_vol[up+1] if up+1<n_bins else 0
        dn_v=bin_vol[dn-1] if dn>0 else 0
        if up_v>=dn_v and up+1<n_bins: up+=1; va_vol+=bin_vol[up]
        elif dn>0: dn-=1; va_vol+=bin_vol[dn]
        else: break
    vah=(bins[up]+bins[up+1])/2; val=(bins[dn]+bins[dn+1])/2
    return poc,vah,val

dates_unique=sorted(m1['date_only'].unique())
daily_vp={}
for d in dates_unique:
    dd=m1[m1['date_only']==d]
    if len(dd)<60: continue
    poc,vah,val=calc_vp(dd)
    if poc is not None: daily_vp[d]=dict(poc=poc,vah=vah,val=val)

TOUCH=0.0015   # ±0.15%
MAX_BARS=120   # 最大2時間

# シグナル収集 (固定TP/SL用に生データ保存)
raw_signals=[]  # {lv, direction, entry, time, sloping, ...}

for idx in range(len(dates_unique)-1):
    prev_d=dates_unique[idx]; curr_d=dates_unique[idx+1]
    if prev_d not in daily_vp: continue
    vp=daily_vp[prev_d]
    poc=vp['poc']; vah=vp['vah']; val=vp['val']
    sloping=h1_reg.get(curr_d,'NA')
    curr=m1[m1['date_only']==curr_d].reset_index(drop=True)
    if len(curr)<30: continue

    touched={'POC':False,'VAH':False,'VAL':False}
    for j in range(len(curr)-1):
        row=curr.iloc[j]
        hr=row['Date'].hour
        if not (7<=hr<=19): continue
        mid=(row['h']+row['l'])/2
        entry=curr.iloc[j+1]['o']

        for lv_name, level, direction in [
            ('VAH', vah, 'short'),
            ('VAL', val, 'long'),
            ('POC', poc, 'short' if sloping=='DOWN' else 'long'),
        ]:
            if touched[lv_name]: continue
            if abs(mid-level)/level>TOUCH: continue

            # 次120本のhigh/lowシーケンスを保存
            future_h=curr.iloc[j+1:j+1+MAX_BARS]['h'].values
            future_l=curr.iloc[j+1:j+1+MAX_BARS]['l'].values

            raw_signals.append(dict(
                lv=lv_name, direction=direction, entry=entry,
                time=row['Date'], sloping=sloping, hour=hr,
                future_h=future_h, future_l=future_l,
                level=level
            ))
            touched[lv_name]=True

print(f"シグナル数: {len(raw_signals)}")

# TP/SL グリッド評価
tp_vals=[3,5,8,10,15]
sl_vals=[3,5,8,10,15]

def eval_grid(signals, tp_pt, sl_pt):
    results=[]
    for s in signals:
        fh=s['future_h']; fl=s['future_l']; e=s['entry']; d=s['direction']
        hit=None
        for k in range(len(fh)):
            if d=='short':
                if fh[k]>=e+sl_pt: hit=-(sl_pt+SPREAD); break
                if fl[k]<=e-tp_pt: hit=+(tp_pt-SPREAD); break
            else:
                if fl[k]<=e-sl_pt: hit=-(sl_pt+SPREAD); break
                if fh[k]>=e+tp_pt: hit=+(tp_pt-SPREAD); break
        if hit is not None: results.append(hit)
    return results

def st(pnls):
    if len(pnls)<5: return None,None,None,None
    wr=100*sum(1 for p in pnls if p>0)/len(pnls)
    ev=np.mean(pnls)
    h1p=pnls[:len(pnls)//2]; h2p=pnls[len(pnls)//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    stable=e1>0 and e2>0
    return wr,ev,stable,len(pnls)

print()
print("="*75)
print("  全シグナル × TP/SL グリッド (スキャルピング)")
print("="*75)
print(f"  {'TP':>4} {'SL':>4} | {'n':>4} {'WR':>5} {'EV':>7} {'Total':>7} | STABLE")
print("  "+"-"*55)

best=[]
for tp in tp_vals:
    for sl in sl_vals:
        r=eval_grid(raw_signals, tp, sl)
        wr,ev,stable,n_=st(r)
        if wr is None: continue
        flag='✅' if stable else '  '
        total=sum(r)
        print(f"  TP={tp:2d} SL={sl:2d} | {n_:4d} {wr:4.0f}%  {ev:+6.2f}  {total:+7.1f} | {flag}")
        if stable: best.append((tp,sl,n_,wr,ev,total))

# VAH+DOWN, VAL+UP 別
print()
print("="*75)
print("  VAHショート (SLOPING=DOWN) × TP/SL グリッド")
print("="*75)
vah_down=[s for s in raw_signals if s['lv']=='VAH' and s['sloping']=='DOWN']
print(f"  シグナル数: {len(vah_down)}")
print(f"  {'TP':>4} {'SL':>4} | {'n':>4} {'WR':>5} {'EV':>7} {'Total':>7} | STABLE")
print("  "+"-"*55)
for tp in tp_vals:
    for sl in sl_vals:
        r=eval_grid(vah_down, tp, sl)
        wr,ev,stable,n_=st(r)
        if wr is None: continue
        flag='✅' if stable else '  '
        total=sum(r)
        print(f"  TP={tp:2d} SL={sl:2d} | {n_:4d} {wr:4.0f}%  {ev:+6.2f}  {total:+7.1f} | {flag}")

print()
print("="*75)
print("  POCショート (SLOPING=DOWN) × TP/SL グリッド")
print("="*75)
poc_down=[s for s in raw_signals if s['lv']=='POC' and s['sloping']=='DOWN']
print(f"  シグナル数: {len(poc_down)}")
print(f"  {'TP':>4} {'SL':>4} | {'n':>4} {'WR':>5} {'EV':>7} {'Total':>7} | STABLE")
print("  "+"-"*55)
for tp in tp_vals:
    for sl in sl_vals:
        r=eval_grid(poc_down, tp, sl)
        wr,ev,stable,n_=st(r)
        if wr is None: continue
        flag='✅' if stable else '  '
        total=sum(r)
        print(f"  TP={tp:2d} SL={sl:2d} | {n_:4d} {wr:4.0f}%  {ev:+6.2f}  {total:+7.1f} | {flag}")

# ── 反発の「深さ分布」をもっと詳しく ──────────────────────────
print()
print("="*75)
print("  VAHタッチ後の値動き詳細 (シグナル足起点, 最初の30分)")
print("  → 「何ptまで戻ったか」の分布")
print("="*75)
vah_sigs=[s for s in raw_signals if s['lv']=='VAH']
bounces=[]
for s in vah_sigs:
    fh=s['future_h']; fl=s['future_l']; e=s['entry']
    # 最初の30本(30分)での最大下落
    max_fall=e-min(fl[:30]) if len(fl)>=30 else e-min(fl)
    max_rise=max(fh[:30])-e if len(fh)>=30 else max(fh)-e
    bounces.append(dict(fall=max_fall, rise=max_rise, sloping=s['sloping']))

bdf=pd.DataFrame(bounces)
print(f"\n  VAHショート目線 (タッチ後30分の最大下落pt):")
print(f"  全体  n={len(bdf)}  "
      f"平均={bdf['fall'].mean():.1f}  中央={bdf['fall'].median():.1f}  "
      f"p25={bdf['fall'].quantile(0.25):.1f}  p75={bdf['fall'].quantile(0.75):.1f}")

for sl in ['DOWN','UP','RANGE']:
    sub=bdf[bdf['sloping']==sl]
    if len(sub)<3: continue
    print(f"  {sl:6s} n={len(sub):3d}  "
          f"平均={sub['fall'].mean():.1f}  中央={sub['fall'].median():.1f}  "
          f"3pt超={100*sum(sub['fall']>3)/len(sub):.0f}%  "
          f"5pt超={100*sum(sub['fall']>5)/len(sub):.0f}%  "
          f"8pt超={100*sum(sub['fall']>8)/len(sub):.0f}%  "
          f"10pt超={100*sum(sub['fall']>10)/len(sub):.0f}%")

print(f"\n  VAHの逆方向リスク (タッチ後30分の最大上昇pt = SL方向):")
print(f"  全体  3pt超={100*sum(bdf['rise']>3)/len(bdf):.0f}%  "
      f"5pt超={100*sum(bdf['rise']>5)/len(bdf):.0f}%  "
      f"8pt超={100*sum(bdf['rise']>8)/len(bdf):.0f}%  "
      f"10pt超={100*sum(bdf['rise']>10)/len(bdf):.0f}%")

# VAL
val_sigs=[s for s in raw_signals if s['lv']=='VAL']
bounces_v=[]
for s in val_sigs:
    fh=s['future_h']; fl=s['future_l']; e=s['entry']
    max_rise=max(fh[:30])-e if len(fh)>=30 else max(fh)-e
    max_fall=e-min(fl[:30]) if len(fl)>=30 else e-min(fl)
    bounces_v.append(dict(rise=max_rise,fall=max_fall,sloping=s['sloping']))
bdfv=pd.DataFrame(bounces_v)
print(f"\n  VALロング目線 (タッチ後30分の最大上昇pt):")
print(f"  全体  n={len(bdfv)}  "
      f"平均={bdfv['rise'].mean():.1f}  中央={bdfv['rise'].median():.1f}  "
      f"3pt超={100*sum(bdfv['rise']>3)/len(bdfv):.0f}%  "
      f"5pt超={100*sum(bdfv['rise']>5)/len(bdfv):.0f}%  "
      f"8pt超={100*sum(bdfv['rise']>8)/len(bdfv):.0f}%")
print(f"  VALの逆方向リスク (下落 = SL方向):")
print(f"  全体  3pt超={100*sum(bdfv['fall']>3)/len(bdfv):.0f}%  "
      f"5pt超={100*sum(bdfv['fall']>5)/len(bdfv):.0f}%  "
      f"8pt超={100*sum(bdfv['fall']>8)/len(bdfv):.0f}%  "
      f"10pt超={100*sum(bdfv['fall']>10)/len(bdfv):.0f}%")
