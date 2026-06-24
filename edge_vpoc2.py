"""
前日POC/VAH/VAL 反発エッジ v2
─ 正しい方向ロジック ─
  VAH touch → SHORT (価値域上限 = 上抵抗)
  VAL touch → LONG  (価値域下限 = 下支持)
  POC touch → アプローチ方向で判断 (上から→short, 下から→long)

─ TP目標3パターン ─
  (A) TP = 対岸レベル (VAH→POC距離, VAL→POC距離)
  (B) TP = 0.5×H1 ATR
  (C) TP = 1.0×H1 ATR  SL = 0.5×H1 ATR (RR=1:2逆張り)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

# ── M1 ──────────────────────────────────────────────────────
m1=pd.read_csv(f"{UP}/9c76cb47-XAUUSD_M1_Mar_Jun.csv")
m1['Date']=pd.to_datetime(m1['Date'],format='%Y.%m.%d %H:%M')
m1=m1.sort_values('Date').reset_index(drop=True)
m1.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
m1['date_only']=m1['Date'].dt.date

# ── H1 ──────────────────────────────────────────────────────
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
h1_atr={}; h1_reg={}
for i in range(nh):
    d=h1['date_only'].iloc[i]
    if not np.isnan(at[i]): h1_atr[d]=at[i]
    h1_reg[d]=reg[i]

# ── Volume Profile ───────────────────────────────────────────
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
print(f"VP計算完了: {len(daily_vp)}日分\n")

# ── トレード評価 ─────────────────────────────────────────────
TOUCH=0.0015   # ±0.15% (少し広げた)
MAX_BARS=240   # 最大4時間保有

def eval_trade(m1_df, j, direction, sl, tp):
    """M1でエントリー後の評価"""
    entry=m1_df.iloc[j+1]['o'] if j+1<len(m1_df) else None
    if entry is None: return None
    for k in range(j+1, min(len(m1_df), j+MAX_BARS+1)):
        r=m1_df.iloc[k]
        if direction=='short':
            if r['h']>=sl: return -(sl-entry)-SPREAD
            if r['l']<=tp: return (entry-tp)-SPREAD
        else:
            if r['l']<=sl: return -(entry-sl)-SPREAD
            if r['h']>=tp: return (tp-entry)-SPREAD
    return None

results=[]

for idx in range(len(dates_unique)-1):
    prev_d=dates_unique[idx]
    curr_d=dates_unique[idx+1]
    if prev_d not in daily_vp: continue
    vp=daily_vp[prev_d]
    poc=vp['poc']; vah=vp['vah']; val=vp['val']
    va_range=vah-val  # 価値域幅

    atr=h1_atr.get(curr_d); sloping=h1_reg.get(curr_d,'NA')
    if not atr or atr<=0: continue

    curr=m1[m1['date_only']==curr_d].reset_index(drop=True)
    if len(curr)<30: continue

    touched={'POC':False,'VAH':False,'VAL':False}

    for j in range(len(curr)-1):
        row=curr.iloc[j]
        hr=row['Date'].hour
        if not (7<=hr<=19): continue
        mid=(row['h']+row['l'])/2

        for lv_name, level, default_dir, tp_target in [
            # VAH → SHORT (価値域上限で売り), TP=POCまで
            ('VAH', vah, 'short', poc),
            # VAL → LONG  (価値域下限で買い), TP=POCまで
            ('VAL', val, 'long',  poc),
            # POC → アプローチ方向: SLOPINGがDOWNならshort優先
            ('POC', poc, None,    None),
        ]:
            if touched[lv_name]: continue
            dist=abs(mid-level)/level
            if dist>TOUCH: continue

            # POCは方向判断
            if lv_name=='POC':
                # 直近5本の終値トレンドで判断
                prev5=curr.iloc[max(0,j-5):j]['c'].values
                if len(prev5)<2: continue
                direction='short' if prev5[-1]>prev5[0] else 'long'
                # SLOPINGヒント
                if sloping=='DOWN': direction='short'
                elif sloping=='UP': direction='long'
                tp_target=val if direction=='short' else vah
            else:
                direction=default_dir

            # TP/SL 3パターン
            if direction=='short':
                sl_A=level+atr*0.5; tp_A=max(tp_target, mid-atr*0.3)
                sl_B=mid+atr*0.5; tp_B=mid-atr*0.5
                sl_C=mid+atr*0.5; tp_C=mid-atr*1.0
            else:
                sl_A=level-atr*0.5; tp_A=min(tp_target, mid+atr*0.3)
                sl_B=mid-atr*0.5; tp_B=mid+atr*0.5
                sl_C=mid-atr*0.5; tp_C=mid+atr*1.0

            r_A=eval_trade(curr,j,direction,sl_A,tp_A)
            r_B=eval_trade(curr,j,direction,sl_B,tp_B)
            r_C=eval_trade(curr,j,direction,sl_C,tp_C)

            results.append(dict(
                date=curr_d, time=row['Date'], lv=lv_name,
                level=level, poc=poc, vah=vah, val=val,
                direction=direction, atr=atr, sloping=sloping,
                hour=hr, dow=row['Date'].weekday(),
                r_A=r_A, r_B=r_B, r_C=r_C
            ))
            touched[lv_name]=True

dt=pd.DataFrame(results)
print(f"シグナル総数: {len(dt)}")

# ── 統計ヘルパー ─────────────────────────────────────────────
def st(pnls, label=''):
    p=[x for x in pnls if x is not None]
    if len(p)<5: return
    n_=len(p); wr=100*sum(1 for x in p if x>0)/n_; ev=np.mean(p)
    h1p=p[:n_//2]; h2p=p[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    mcl=0; cur=0
    for x in p:
        if x<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    stable='✅STABLE' if e1>0 and e2>0 else '❌'
    nmo=n_/3.3
    print(f"  {label:35s}: n={n_:3d}({nmo:.0f}/月) WR={wr:.0f}% EV={ev:+.2f} "
          f"前半={e1:+.2f} 後半={e2:+.2f} {stable}")

# ── メイン結果 ───────────────────────────────────────────────
print("="*85)
print("  パターンA: SL=0.5ATR / TP=対岸VP水準 (VAHタッチ→POCまでTP)")
print("="*85)
for lv in ['VAH','VAL','POC']:
    sub=dt[dt['lv']==lv]['r_A'].tolist()
    st(sub, f"全体 {lv}")
for lv in ['VAH','VAL','POC']:
    for sl in ['DOWN','UP']:
        sub=dt[(dt['lv']==lv)&(dt['sloping']==sl)]['r_A'].tolist()
        st(sub, f"{lv}+SLOPING={sl}")

print()
print("="*85)
print("  パターンB: SL=0.5ATR / TP=0.5ATR (対称1:1)")
print("="*85)
for lv in ['VAH','VAL','POC']:
    sub=dt[dt['lv']==lv]['r_B'].tolist()
    st(sub, f"全体 {lv}")
for lv in ['VAH','VAL']:
    for sl in ['DOWN','UP']:
        sub=dt[(dt['lv']==lv)&(dt['sloping']==sl)]['r_B'].tolist()
        st(sub, f"{lv}+SLOPING={sl}")

print()
print("="*85)
print("  パターンC: SL=0.5ATR / TP=1.0ATR (RR=1:2 有利)")
print("="*85)
for lv in ['VAH','VAL','POC']:
    sub=dt[dt['lv']==lv]['r_C'].tolist()
    st(sub, f"全体 {lv}")
for lv in ['VAH','VAL']:
    for sl in ['DOWN','UP']:
        sub=dt[(dt['lv']==lv)&(dt['sloping']==sl)]['r_C'].tolist()
        st(sub, f"{lv}+SLOPING={sl}")

# ── 時間帯別 ─────────────────────────────────────────────────
print()
print("="*85)
print("  時間帯 × レベル (パターンB)")
print("="*85)
for sess,h_r in [('London 07-12',range(7,13)),('NY 13-19',range(13,20))]:
    for lv in ['VAH','VAL','POC']:
        sub=dt[(dt['lv']==lv)&(dt['hour'].isin(h_r))]['r_B'].tolist()
        st(sub, f"{sess} {lv}")

# ── VAHショート × SLOPING DOWN 詳細 ────────────────────────────
print()
print("="*85)
print("  VAHショート+DOWN 全トレード詳細 (パターンB)")
print("="*85)
sub_dt=dt[(dt['lv']=='VAH')&(dt['sloping']=='DOWN')].reset_index(drop=True)
print(f"  {'日時':22s} {'VAH':8s} {'POC':8s} {'ATR':6s} {'結果':>8}")
print("  "+"-"*60)
for _,row in sub_dt.iterrows():
    r=row['r_B']
    if r is None: continue
    sign='✅' if r>0 else '❌'
    print(f"  {sign} {str(row['time']):22s} {row['vah']:8.2f} {row['poc']:8.2f} {row['atr']:6.2f} {r:+8.2f}pt")

# ── 反発幅の実測 (実際に何pt反発するか) ─────────────────────────
print()
print("="*85)
print("  タッチ後の実際の反発幅 (最初の60分=60本)")
print("="*85)
bounce_vah=[]; bounce_val=[]
for idx in range(len(dates_unique)-1):
    prev_d=dates_unique[idx]; curr_d=dates_unique[idx+1]
    if prev_d not in daily_vp: continue
    vp=daily_vp[prev_d]
    vah_l=vp['vah']; val_l=vp['val']
    curr=m1[m1['date_only']==curr_d].reset_index(drop=True)
    for j in range(len(curr)-61):
        row=curr.iloc[j]
        if not (7<=row['Date'].hour<=19): continue
        mid=(row['h']+row['l'])/2
        # VAHタッチ
        if abs(mid-vah_l)/vah_l<=TOUCH:
            next60=curr.iloc[j:j+60]
            min_price=next60['l'].min()
            bounce=vah_l-min_price  # 下方向反発
            bounce_vah.append(bounce)
        # VALタッチ
        if abs(mid-val_l)/val_l<=TOUCH:
            next60=curr.iloc[j:j+60]
            max_price=next60['h'].max()
            bounce=max_price-val_l  # 上方向反発
            bounce_val.append(bounce)

if bounce_vah:
    print(f"  VAHタッチ後60分の下落幅: "
          f"平均={np.mean(bounce_vah):.1f}pt  中央={np.median(bounce_vah):.1f}pt  "
          f"最大={np.max(bounce_vah):.1f}pt  n={len(bounce_vah)}")
    pcts=[10,25,50,75,90]
    print(f"  パーセンタイル: {' '.join(f'p{p}={np.percentile(bounce_vah,p):.1f}' for p in pcts)}")
if bounce_val:
    print(f"  VALタッチ後60分の上昇幅: "
          f"平均={np.mean(bounce_val):.1f}pt  中央={np.median(bounce_val):.1f}pt  "
          f"最大={np.max(bounce_val):.1f}pt  n={len(bounce_val)}")
    pcts=[10,25,50,75,90]
    print(f"  パーセンタイル: {' '.join(f'p{p}={np.percentile(bounce_val,p):.1f}' for p in pcts)}")

# 最適SL/TPを示唆
if bounce_vah and bounce_val:
    print()
    avg_bounce=(np.mean(bounce_vah)+np.mean(bounce_val))/2
    med_bounce=(np.median(bounce_vah)+np.median(bounce_val))/2
    print(f"  → 反発は平均{avg_bounce:.1f}pt / 中央値{med_bounce:.1f}pt")
    print(f"  → H1 ATR(平均{np.mean(list(h1_atr.values())):.1f}pt)の約{med_bounce/np.mean(list(h1_atr.values()))*100:.0f}%の反発")
    print(f"  → 推奨TP: {med_bounce*0.5:.0f}〜{med_bounce*0.7:.0f}pt固定 / SL: {med_bounce*1.5:.0f}pt")
