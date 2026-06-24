"""
前日POC / VAH / VAL レベル反発エッジ検証
M1データ + H1データ組み合わせ
Volume Profile: 各価格帯のボリューム集計
POC = Point of Control (最大出来高価格)
VAH = Value Area High (出来高上位70%上限)
VAL = Value Area Low  (出来高上位70%下限)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

UP="/root/.claude/uploads/d3b4dd6d-4c2a-5492-a1b3-6ef4295aea59"
SPREAD=0.3

# ─── M1データ ────────────────────────────────────────────────
m1=pd.read_csv(f"{UP}/9c76cb47-XAUUSD_M1_Mar_Jun.csv")
m1['Date']=pd.to_datetime(m1['Date'],format='%Y.%m.%d %H:%M')
m1=m1.sort_values('Date').reset_index(drop=True)
m1.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c','Volume':'v'},inplace=True)
m1['date_only']=m1['Date'].dt.date
m1['hour']=m1['Date'].dt.hour

# ─── H1データ (MA200, SLOPING, ATR用) ───────────────────────
h1=pd.read_csv(f"{UP}/cb66c61a-XAUUSD_H1_Feb_Jun.csv")
h1['Date']=pd.to_datetime(h1['Date'],format='%Y.%m.%d %H:%M')
h1=h1.sort_values('Date').reset_index(drop=True)
h1.rename(columns={'Open':'o','High':'h','Low':'l','Close':'c'},inplace=True)
nh=len(h1)
ch=h1['c'].values; oh=h1['o'].values; hh=h1['h'].values; lh=h1['l'].values

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

# H1 → 日付ごとのATR & SLOPING辞書
h1['date_only']=h1['Date'].dt.date
h1_reg={}; h1_atr={}
for i in range(nh):
    d=h1['date_only'].iloc[i]
    h1_reg[d]=reg[i]
    if not np.isnan(at[i]): h1_atr[d]=at[i]

# ─── 日別Volume Profile計算 ─────────────────────────────────
def calc_vp(df_day, n_bins=100):
    """1日分M1データからPOC/VAH/VALを計算"""
    lo=df_day['l'].min(); hi=df_day['h'].max()
    if hi<=lo: return None,None,None
    bins=np.linspace(lo,hi,n_bins+1)
    bin_vol=np.zeros(n_bins)
    for _,row in df_day.iterrows():
        # 各バーの出来高をハイロー範囲内のビンに均等配布
        mask=(bins[1:]>=row['l'])&(bins[:-1]<=row['h'])
        span=max(mask.sum(),1)
        bin_vol[mask]+=row['v']/span

    poc_bin=np.argmax(bin_vol)
    poc=(bins[poc_bin]+bins[poc_bin+1])/2

    # Value Area: 全出来高の70%を含む最もPOC周辺の領域
    total_vol=bin_vol.sum()
    target=total_vol*0.70
    up=poc_bin; dn=poc_bin; va_vol=bin_vol[poc_bin]
    while va_vol<target:
        up_vol=bin_vol[up+1] if up+1<n_bins else 0
        dn_vol=bin_vol[dn-1] if dn>0 else 0
        if up_vol>=dn_vol and up+1<n_bins:
            up+=1; va_vol+=bin_vol[up]
        elif dn>0:
            dn-=1; va_vol+=bin_vol[dn]
        else: break

    vah=(bins[up]+bins[up+1])/2
    val=(bins[dn]+bins[dn+1])/2
    return poc,vah,val

# 日別VP計算
dates_unique=sorted(m1['date_only'].unique())
daily_vp={}  # {date: (poc, vah, val)}

print("日別VP計算中...")
for d in dates_unique:
    day_data=m1[m1['date_only']==d]
    if len(day_data)<60: continue  # 1時間未満はスキップ
    poc,vah,val=calc_vp(day_data)
    if poc is not None:
        daily_vp[d]=dict(poc=poc,vah=vah,val=val,
                         day_high=day_data['h'].max(),
                         day_low=day_data['l'].min())

print(f"VP計算完了: {len(daily_vp)}日分")

# ─── 前日VP水準でのM1反発テスト ──────────────────────────────
# ルール:
#   前日のPOC/VAH/VALを計算
#   翌日、価格がその水準の±0.1%(=XAUUSD約2-5pt)以内に来たら
#   → 反発方向にエントリー (水準より上→ロング、下→ショート)
#   → SL: ATR(14)H1, TP: ATR(14)H1 (1:1)
#   → 07:00-19:00 UTCのみ

TOUCH_RANGE=0.001  # ±0.1% (XAU5000なら±5pt)

trades_poc=[]; trades_vah=[]; trades_val=[]
trades_poc_sl=[]; trades_vah_sl=[]; trades_val_sl=[]

detail_trades=[]  # 全トレード詳細

for idx in range(len(dates_unique)-1):
    prev_d=dates_unique[idx]
    curr_d=dates_unique[idx+1]

    if prev_d not in daily_vp: continue
    vp=daily_vp[prev_d]
    poc=vp['poc']; vah=vp['vah']; val=vp['val']

    # 当日のM1バー
    curr_m1=m1[m1['date_only']==curr_d].reset_index(drop=True)
    if len(curr_m1)<30: continue

    # 当日のH1 ATR & SLOPING
    atr=h1_atr.get(curr_d, None)
    if atr is None or atr<=0: continue
    sloping=h1_reg.get(curr_d,'NA')

    # タッチ検出 (重複回避: 同水準は1日1回)
    touched_poc=False; touched_vah=False; touched_val=False

    for j in range(len(curr_m1)-1):
        row=curr_m1.iloc[j]
        if not (7<=row['Date'].hour<=19): continue

        mid=(row['h']+row['l'])/2
        entry_price=curr_m1.iloc[j+1]['o']  # 次バー始値エントリー

        for level_name, level, touched_flag in [
            ('POC', poc, touched_poc),
            ('VAH', vah, touched_vah),
            ('VAL', val, touched_val)
        ]:
            if touched_flag: continue

            dist=abs(mid-level)/level
            if dist > TOUCH_RANGE: continue

            # 反発方向
            if mid >= level:  # 水準より上 → ショート (上抵抗)
                direction='short'
                tp_price=entry_price-atr
                sl_price=entry_price+atr
            else:             # 水準より下 → ロング (下支持)
                direction='long'
                tp_price=entry_price+atr
                sl_price=entry_price-atr

            # M1で結果判定 (最大4時間=240本)
            result=None
            for k in range(j+1, min(len(curr_m1), j+241)):
                r=curr_m1.iloc[k]
                if direction=='short':
                    if r['h']>=sl_price: result=-(atr+SPREAD); break
                    if r['l']<=tp_price: result=+(atr-SPREAD); break
                else:
                    if r['l']<=sl_price: result=-(atr+SPREAD); break
                    if r['h']>=tp_price: result=+(atr-SPREAD); break

            if result is None: continue

            pnl_pt=result/atr  # ATR正規化
            pnl_raw=result

            detail_trades.append(dict(
                date=curr_d, time=row['Date'], level_name=level_name,
                level=level, direction=direction, atr=atr,
                sloping=sloping, result=pnl_raw, pnl_norm=pnl_pt,
                prev_poc=poc, prev_vah=vah, prev_val=val
            ))

            if level_name=='POC':
                trades_poc.append(pnl_raw); touched_poc=True
            elif level_name=='VAH':
                trades_vah.append(pnl_raw); touched_vah=True
            elif level_name=='VAL':
                trades_val.append(pnl_raw); touched_val=True

# ─── 統計 ─────────────────────────────────────────────────────
def stats(pnls, label):
    if not pnls: print(f"  {label}: データなし"); return
    n_=len(pnls)
    wr=100*sum(1 for p in pnls if p>0)/n_
    ev=np.mean(pnls)
    total=sum(pnls)
    h1p=pnls[:n_//2]; h2p=pnls[n_//2:]
    e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
    stable='✅STABLE' if e1>0 and e2>0 else '❌UNSTABLE'
    mcl=0; cur=0
    for p in pnls:
        if p<0: cur+=1; mcl=max(mcl,cur)
        else: cur=0
    nmo=n_/(len(dates_unique)/22)  # 月あたり
    print(f"  {label:30s}: n={n_:4d} ({nmo:.0f}/月)  WR={wr:.0f}%  EV={ev:+.2f}pt  Total={total:+.0f}pt  MaxCL={mcl}  {stable}")
    print(f"    前半EV={e1:+.2f}  後半EV={e2:+.2f}")

print()
print("="*80)
print("  前日VP水準 (POC/VAH/VAL) 反発エッジ検証")
print("  M1足タッチ検出 → M1足次バー始値エントリー | SL/TP=H1 ATR(14)")
print("  タッチ範囲: ±0.1% | 07:00-19:00 UTC")
print("="*80)

stats(trades_poc, "POC (Point of Control)")
print()
stats(trades_vah, "VAH (Value Area High)")
print()
stats(trades_val, "VAL (Value Area Low)")

# ─── フィルター別分析 ──────────────────────────────────────────
print()
print("="*80)
print("  フィルター別分析")
print("="*80)

dt_df=pd.DataFrame(detail_trades)
if len(dt_df)==0:
    print("  トレードなし"); exit()

print(f"\n  全体: n={len(dt_df)}  WR={100*sum(dt_df['result']>0)/len(dt_df):.0f}%  EV={dt_df['result'].mean():+.2f}pt")

# SLOPING別
for sl in ['UP','DOWN','RANGE','NA']:
    sub=dt_df[dt_df['sloping']==sl]['result'].tolist()
    if not sub: continue
    wr=100*sum(1 for p in sub if p>0)/len(sub)
    ev=np.mean(sub)
    print(f"  SLOPING={sl:5s}: n={len(sub):4d}  WR={wr:.0f}%  EV={ev:+.2f}pt")

print()
# レベル×SLOPING
for sl in ['DOWN','UP']:
    for lv in ['POC','VAH','VAL']:
        sub=dt_df[(dt_df['sloping']==sl)&(dt_df['level_name']==lv)]['result'].tolist()
        if len(sub)<5: continue
        wr=100*sum(1 for p in sub if p>0)/len(sub)
        ev=np.mean(sub)
        h1p=sub[:len(sub)//2]; h2p=sub[len(sub)//2:]
        e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
        stable='✅' if e1>0 and e2>0 else '❌'
        print(f"  {lv}+SLOPING={sl}: n={len(sub):3d}  WR={wr:.0f}%  EV={ev:+.2f}pt  {stable}")

# 方向別
print()
for direction in ['short','long']:
    sub=dt_df[dt_df['direction']==direction]['result'].tolist()
    if not sub: continue
    wr=100*sum(1 for p in sub if p>0)/len(sub)
    ev=np.mean(sub)
    print(f"  direction={direction}: n={len(sub):4d}  WR={wr:.0f}%  EV={ev:+.2f}pt")

# 時間帯別
print()
dt_df['hour']=pd.to_datetime(dt_df['time']).dt.hour
for session,h_range in [('London(7-12)',range(7,13)),('NY(13-19)',range(13,20))]:
    sub=dt_df[dt_df['hour'].isin(h_range)]['result'].tolist()
    if not sub: continue
    wr=100*sum(1 for p in sub if p>0)/len(sub)
    ev=np.mean(sub)
    print(f"  {session}: n={len(sub):4d}  WR={wr:.0f}%  EV={ev:+.2f}pt")

# ─── 最強コンビ探索 ─────────────────────────────────────────────
print()
print("="*80)
print("  最強コンビネーション探索")
print("="*80)

combos=[]
for sl in ['DOWN','UP']:
    for lv in ['POC','VAH','VAL']:
        for sess,h_r in [('London',range(7,13)),('NY',range(13,20)),('All',range(7,20))]:
            sub=dt_df[(dt_df['sloping']==sl)&(dt_df['level_name']==lv)&
                      (dt_df['hour'].isin(h_r))]['result'].tolist()
            if len(sub)<8: continue
            wr=100*sum(1 for p in sub if p>0)/len(sub)
            ev=np.mean(sub)
            h1p=sub[:len(sub)//2]; h2p=sub[len(sub)//2:]
            e1=np.mean(h1p) if h1p else 0; e2=np.mean(h2p) if h2p else 0
            stable=e1>0 and e2>0
            combos.append(dict(label=f"{lv}+{sl}+{sess}",n=len(sub),wr=wr,ev=ev,stable=stable,
                               e1=e1,e2=e2,pnls=sub))

combos.sort(key=lambda x: -x['ev'])
for c in combos:
    st='✅STABLE' if c['stable'] else '❌'
    print(f"  {c['label']:28s}: n={c['n']:3d}  WR={c['wr']:.0f}%  EV={c['ev']:+.2f}pt  "
          f"前半={c['e1']:+.2f} 後半={c['e2']:+.2f}  {st}")

# ─── 日別サンプル表示 (タッチ具体例) ────────────────────────────
print()
print("="*80)
print("  具体的タッチ例 (最初の20件)")
print("="*80)
print(f"  {'日時':22s} {'レベル':5s} {'水準':8s} {'方向':7s} {'ATR':7s} {'結果(pt)':>10}")
print("  " + "-"*70)
for t in detail_trades[:20]:
    sign='✅' if t['result']>0 else '❌'
    print(f"  {sign} {str(t['time']):22s} {t['level_name']:5s} {t['level']:8.2f} {t['direction']:7s} "
          f"{t['atr']:6.2f}  {t['result']:+8.2f}pt")

# ─── POC/VAH/VAL前日水準一覧 (3月最初の5日) ─────────────────
print()
print("="*80)
print("  前日VP水準サンプル (最初の5日)")
print("="*80)
for d in dates_unique[1:6]:
    prev_d=dates_unique[dates_unique.index(d)-1]
    if prev_d in daily_vp:
        vp=daily_vp[prev_d]
        print(f"  {d} 前日({prev_d}) → POC:{vp['poc']:.2f}  VAH:{vp['vah']:.2f}  VAL:{vp['val']:.2f}")
        print(f"    前日レンジ: {vp['day_low']:.2f} 〜 {vp['day_high']:.2f}")
