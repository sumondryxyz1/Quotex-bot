# Advanced Market Intelligence & Predictive Signal Engine — Research & Design Plan

**তারিখ:** 2026-11-07
**ভিত্তি:** `AUDIT_REPORT.md` (commit `0ec531c`) — যেখানে প্রমাণিত: বর্তমান বটে কোনো
statistically validated edge নেই (সব পরীক্ষা ~৫০%, breakeven ৫৪.১%)।
**এই প্ল্যানের লক্ষ্য:** বেশি signal নয় — **statistically validated predictive edge**
খুঁজে বের করা, অথবা সৎভাবে প্রমাণ করা যে এই ফিডে তা নেই।
**কোনো কোড পরিবর্তন করা হয়নি।** এটি শুধু design + roadmap।

---

## ০. মূল দর্শন (Non-negotiable Principles)

এই প্ল্যানের প্রতিটি সিদ্ধান্ত নিচের ৯টি নিয়ম মেনে চলবে:

1. **Edge first, signal second.** কোনো strategy "কাজ করছে" বলা হবে না যতক্ষণ না
   out-of-sample payout-adjusted EV > 0 প্রমাণ না হয়।
2. **Indicator যোগ করা মানে উন্নতি নয়।** নতুন feature যোগ করার আগে প্রমাণ করতে হবে
   সেটা *কিছু* নতুন তথ্য বহন করে (mutual information > 0 on held-out data)।
3. **OTC ও Real Forex আলাদা।** দুটো আলাদা universe — আলাদা DB, আলাদা পরীক্ষা,
   কখনো pooled করা হবে না (pooling raw returns আগেই প্রমাণিতভাবে ভুয়া significance বানায়)।
4. **Look-ahead ও leakage শিকারি-পরীক্ষা (hunt) বাধ্যতামূলক।** প্রতিটি নতুন module-এর
   জন্য automated audit।
5. **Walk-forward + out-of-sample বাধ্যতামূলক।** Single train/test split যথেষ্ট নয়;
   purged walk-forward লাগবে।
6. **Accuracy একা অর্থহীন।** রিপোর্টে থাকবে: payout-adjusted expectancy, Wilson CI,
   max drawdown, profit factor, per-regime ও per-pair breakdown।
7. **কোনো fabricated win-rate নেই।** যা মাপা হয়নি, লেখা হবে না।
8. **Private credentials বা অননুমোদিত endpoint নেই।** SSID দিয়ে authenticated history/
   payout আনা হবে **না**। Payout হবে user-সরবরাহকৃত parameter।
9. **Kill criteria স্পষ্ট।** কোনো strategy ব্যর্থ হলে → `research-only` ট্যাগ বা বাদ।

> **সৎ প্রত্যাশা:** এই ফিডে (বিশেষত OTC) edge না-ও থাকতে পারে। সেক্ষেত্রে
> "কোনো tradable edge নেই" — এটাও একটি **সফল, মূল্যবান ফলাফল**। প্ল্যান সেভাবেই লেখা।

---

## A. নতুন Architecture

### A.1 মূল পরিবর্তন: "Signal Factory" → "Research Pipeline"

বর্তমান bot সরাসরি signal বানায়। নতুন architecture-এ **প্রতিটি candidate signal
একটি validation gate পার হতে হবে** — gate pass না করলে সেটা live নয়, `research-only`।

```
┌─────────────────────────────────────────────────────────────────────┐
│ L0  DATA LAYER  (provenance-tagged)                                  │
│   feed ──► candle_store (source=otc|real) + ticks (alag DB)          │
│   provenance.py: source tag, gap/spike validation, quality score     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  (শুধু validated, tagged data)
┌───────────────────────────────▼─────────────────────────────────────┐
│ L1  STRUCTURE / INTELLIGENCE LAYER  (pure functions, no look-ahead)  │
│   structure.py   → swing, BOS, CHoCH, MSS, internal/external         │
│   liquidity.py   → EQH/EQL, pools, sweep+reclaim, displacement       │
│   zones.py       → Order Block / FVG + VALIDITY (react কিনা মাপে)     │
│   regime.py      → trend/range/chop + vol regime + session           │
│   microstructure.py → tick intensity, clustering, realised vol       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  (feature vectors + structural context)
┌───────────────────────────────▼─────────────────────────────────────┐
│ L2  HYPOTHESIS / VALIDATION LAYER   ◄── এটাই নতুন কেন্দ্র            │
│   validation.py → purged walk-forward, embargo, metrics, power       │
│   প্রতিটি strategy এখানে জমা দেয়; pass না হলে promote হবে না         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  (শুধু validated strategies)
┌───────────────────────────────▼─────────────────────────────────────┐
│ L3  MODEL LAYER                                                      │
│   calibrate.py → calibrated P(up), reliability diagram               │
│   ML: HistGBM / RF / LogReg, walk-forward re-fit                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  (calibrated probability)
┌───────────────────────────────▼─────────────────────────────────────┐
│ L4  DECISION LAYER                                                   │
│   decision.py → ensemble (measured-reliability weighted)             │
│               → NO TRADE gate (EV≤0, regime undefined, conflict)     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ L5  REPORTING LAYER                                                  │
│   dashboard.py + golive.py → payout-adjusted EV, CI, DD, per-regime  │
│   payout.py → payout user-parameter (SSID ছাড়া)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### A.2 Signal Contract (সব module-এর অভিন্ন ইন্টারফেস)

প্রতিটি strategy module একটি uniform object ফিরিয়ে দেবে, যাতে `validation.py`
সবাইকে একইভাবে মাপতে পারে:

```python
@dataclass
class Candidate:
    asset: str
    period: int
    source: str            # "otc" | "real"     ← provenance বাধ্যতামূলক
    time: int
    action: str            # CALL | PUT | ABSTAIN
    features: dict         # raw features (audit-এর জন্য)
    context: dict          # regime, structure, liquidity state
    p_up: float | None     # calibrated probability (থাকলে)
    strategy_id: str       # কোন module + version
```

**কেন:** বর্তমান কোডে প্রতিটি engine আলাদা shape ফেরায় (`Signal`, `SmartSignal`,
`EnsembleSignal`, `Analysis`)। একই validation harness-এ সব মাপা যায় না — এটাই
আগের audit-এ "rule আর ensemble মিশে যায়" সমস্যার মূল।

### A.3 Provenance বাধ্যতামূলক (Goal-এর বিশেষ নির্দেশনা)

প্রতিটি row-এ `source` থাকবে:
- `otc` = `_otc` suffix (broker synthetic, 24/7)
- `real` = non-OTC (আসল market, session-এ চলে)

`candle_store`-এ নতুন column `source` + `feed_id` (feed version)। `edge.py`,
`research.py`, `backtest.py` — সব শুধু একই source-এর মধ্যে pool করবে।

---

## B. নতুন Strategy Modules (১০টি Goal ম্যাপ করে)

### B.1 `liquidity.py` — Advanced Liquidity Detection Engine (Goal 1, 4)

| Function | কাজ |
|---|---|
| `equal_highs_lows(h, l, i)` | EQH/EQL cluster (stop-loss pool) খোঁজা (tolerance-aware) |
| `liquidity_pools(h, l, i)` | clustered swing level = potential stop zone |
| `sweep_and_reclaim(h, l, c, i)` | sweep + close-back classification (current `smc.liquidity_sweep()`-এর উন্নত রূপ) |
| `displacement(o, c, h, l, i, atr)` | বড় body + imbalance রেখে যাওয়া move (institutional footprint) |
| `session_levels(h, l, t, session)` | Asian/London/NY high-low (real pair-এর জন্য) |

**উন্নতির মূল:** বর্তমান `smc.liquidity_sweep()` (line 66) শুধু "wick through prior swing
then close back" দেখে। নতুন module যোগ করবে: (ক) pool size/age, (খ) sweep **সাথে
displacement** আছে কিনা (নাহলে সেটা দুর্বল sweep), (গ) reclaim strength।

### B.2 `structure.py` — Proper BOS & CHoCH (Goal 2, 3)

বর্তমান `smc.structure()` (line 46) শুধু HH+HL/LH+LL দেখে → "up/down/range"।
নতুন module সত্যিকারের BOS ও CHoCH implement করবে:

| Function | কাজ |
|---|---|
| `swing_series(h, l, left, right)` | confirmed fractal pivots (strength সহ) |
| `classify_swings()` | প্রতিটি swing কে HH/HL/LH/LL লেবেল |
| `bos(h, l, c, i)` | **Break of Structure** — trend-এর দিকে confirmed swing break, close beyond |
| `choch(h, l, c, i)` | **Change of Character** — trend-এর বিপরীতে প্রথম break |
| `mss(h, l, c, i)` | Market Structure Shift = CHoCH + displacement |
| `internal_external()` | multi-degree structure (minor vs major) |

**BOS vs CHoCH সংজ্ঞা (স্পষ্ট):**
- **BOS:** prevailing trend-এর দিকেই একটি swing break → trend continue
- **CHoCH:** prevailing trend-এর বিপরীতে প্রথম break → সম্ভাব্য reversal
- উভয়ের জন্য **close beyond the level** লাগবে (wick-only নয়)

**⚠️ সতর্কতা:** "ICT MSP/breaker block" — ICT-এর নির্দিষ্ট proprietary নাম ব্যবহার করা হবে
না; শুধু প্রমাণযোগ্য structural concept implement করা হবে।

### B.3 `zones.py` — Order Block & FVG Validity (Goal 5)

বর্তমান `smc.order_block()` (line 91) ও `smc.fair_value_gap()` (line 116) শুধু zone
**খুঁজে** দেয় — zone **কাজ করে কিনা মাপে না**। নতুন module:

| Function | কাজ |
|---|---|
| `find_order_block(...)` | OB + **displacement requirement** (impulse না হলে OB নয়) |
| `find_fvg(...)` | 3-candle imbalance + gap size filter |
| `zone_state(zone, future_candles)` | mitigated / respected / violated |
| `zone_validity_stats(zones, candles)` | **zone respect rate** — বাস্তবে price react করে কত % |
| `freshness(zone, i)` | zone কত নতুন, price কত দূরে |

**এটাই Goal 5-এর মূল:** zone-এর "validity" অনুমান নয় — **মাপা** হবে।
`zone_validity_stats()` রিপোর্ট করবে: এই ধরনের OB/FVG-তে price কত % বার সত্যিই
বounced। যদি respect rate ~random হয় → zone concept এই ফিডে বাতিল।

### B.4 `regime.py` — Market Regime Detection (Goal 6)

বর্তমান smart engine শুধু ADX ব্যবহার করে (`smart_engine.py:170`)। নতুন module
**multi-dimensional** regime:

| Dimension | Measure |
|---|---|
| Trend vs Range vs Chop | ADX + EMA slope + efficiency ratio (Kaufman) |
| Volatility regime | ATR percentile (rolling) + realised vs implied proxy |
| Mean-reversion vs Momentum | **variance ratio** (`randomwalk.py`-এ আছে) |
| Persistence | Hurst exponent (R/S analysis) |
| Session / time-of-day | hour buckets (real pair-এর জন্য) |

**Output:** `regime_state` = {trend_score, vol_bucket, hurst, vr, session}।
প্রতিটি strategy **শুধু নির্দিষ্ট regime-এ** সক্রিয় হবে (regime-conditioned)।

### B.5 `microstructure.py` — Tick-Level Analysis (Goal 7)

`tickflow.py`-এর সম্প্রসারণ। **সৎ ভিত্তি:** আগের audit-এ প্রমাণিত — tick sign
autocorrelation ≈ 0 (mean z = −0.12), direction flag-এ কোনো তথ্য নেই (49.66%, z=−1.96)।

তাই এই module-এর কাজ: **এই সিদ্ধান্ত re-verify করা এবং নতুন tick-dimension খোঁজা**:

| Function | কাজ |
|---|---|
| `tick_intensity(ticks, window)` | প্রতি সময়-ইউনিটে tick count (activity bursts) |
| `tick_size_distribution(ticks)` | tick size histogram (large-tick clustering) |
| `realised_vs_candle_vol()` | tick-থেকে মাপা vol vs candle-এর vol |
| `tick_autocorr(ticks, lags)` | lag-k autocorrelation (momentum/reversal) |
| `quote_gap_analysis()` | price jump/gap frequency |
| `burst_detection()` | হঠাৎ activity spike (news/stop-run সূচক) |

**প্রত্যাশা:** OTC-তে probable কিছুই নেই। Real pair-এ session-এ হয়তো সামান্য।
মাপা হবে — দাবি করা হবে না।

### B.6 `decision.py` — Ensemble + Intelligent NO TRADE (Goal 9, 10)

| Function | কাজ |
|---|---|
| `weighted_vote(candidates, reliability)` | প্রতিটি strategy-র **মাপা reliability** দিয়ে weighted vote |
| `calibrated_prob(models)` | calibrated P(up) (isolated, `calibrate.py` থেকে) |
| `no_trade_gate(...)` | নিচের যেকোনো হলে ABSTAIN |
| `expected_value(p_up, payout)` | `EV = p_up*(1+payout) - 1` (CALL); ABSTAIN যদি EV ≤ threshold |

**NO TRADE শর্ত (Goal 10):**
1. Regime undefined (transitional ADX / low Hurst confidence)
2. Strategy-গুলো পরস্পরবিরোধী (disagreement > threshold)
3. Calibrated probability breakeven-এর কাছে (EV ≤ 0 + margin)
4. Feature distribution out-of-range (regime shift / data anomaly)
5. Insufficient history for the active strategy
6. Data quality score پایین (gap/spike detected)

**মূল পার্থক্য:** বর্তমান ensemble `min_agree=2` (গণনা-ভিত্তিক)। নতুন gate
**EV-ভিত্তিক** — অর্থাৎ probability ও payout থেকে সরাসরি সিদ্ধান্ত।

---

## C. ML Model Selection

### C.1 কোন model — এবং কেন

| Model | ব্যবহার | কারণ |
|---|---|---|
| `HistGradientBoostingClassifier` | primary | দ্রুত, বড় ডেটায় ভালো, missing value handle করে |
| `RandomForestClassifier` | secondary | ভিন্ন bias, feature importance |
| `LogisticRegression` (regularized) | linear baseline | সরল baseline; ML আসলে কিছু শিখছে কিনা যাচাই |
| `CalibratedClassifierCV` | probability wrapper | raw proba অবিশ্বাসযোগ্য; calibration বাধ্যতামূলক |

**❌ বাদ:** Deep learning (LSTM/Transformer) — এখনই নয়। কারণ: ডেটা ভলিউম কম
(কয়েক দিন), torch install নেই, এবং overfit-ঝুঁকি বেশি। "দিন/সপ্তাহ 5m+" ডেটা
জমলে Phase 9-এ পুনর্বিবেচনা।

### C.2 Target — শুধু direction নয়, payout-adjusted

বর্তমান ML target: "next candle up?" (binary)। নতুন approach:
- **Primary target:** direction (CALL/PUT) — তবে evaluation **EV-ভিত্তিক**
- **Secondary target:** move size class (big/small) → filter low-EV trades
- **মেট্রিক:** accuracy নয়, **payout-adjusted expectancy**

### C.3 Validation (ML)

- `TimeSeriesSplit` (expanding) — বর্তমান `train_model.py:74 walk_forward()`-এর উন্নত রূপ
- **Purged + embargo** split (overlapping labels-এর leakage বন্ধ) — `validation.py`
- **Calibration check:** reliability diagram + Brier score
- **Feature pruning:** permutation importance; শুধু প্রমাণিত-তথ্যবাহী feature রাখা
- **Baseline comparison:** majority-class + random control সবসময় পাশে

### C.4 Model Acceptance Criterion

একটি ML model "গ্রহণযোগ্য" যদি:
1. Calibrated probability-র Brier score baseline-এর চেয়ে ভালো, **এবং**
2. Purged walk-forward-এ payout-adjusted EV-র CI lower bound > 0 (test set-এ), **এবং**
3. OTC ও real আলাদাভাবে দুটোতেই (বা সৎভাবে বলবে কোনটায় নেই)

---

## D. Data Requirements

### D.1 Provenance ও Storage

| পরিবর্তন | কারণ |
|---|---|
| `candles` ও `ticks` টেবিলে `source` column | OTC/real আলাদা রাখতে |
| `data/otc.db` ও `data/real.db` আলাদা (optional) | সম্পূর্ণ বিচ্ছিন্নতা |
| `feed_id` / `collector_version` column | কোন feed version-এর ডেটা জানতে |
| Data quality score per row | gap/spike flag |

### D.2 Statistical Power — কত ডেটা দরকার? (এটাই সবচেয়ে গুরুত্বপূর্ণ সংখ্যা)

Breakeven (৮৫% payout) = ৫৪.১%। একটা edge **প্রমাণ** করতে (α=0.05, power=0.80):

| সত্যিকারের hit-rate | ভুয়া নয় বলে ধরতে দরকার (decisive signals) |
|---|---|
| 54% | ~1,225 |
| 55% | ~785 |
| 56% | ~545 |
| 58% | ~330 |
| 60% | ~195 |

**অর্থাৎ:** একটি strategy-কে "validated" বলতে হলে **প্রতি strategy-প্রতি pair-প্রতি
regime-এ ~1,000+ decisive signal** লাগবে। বর্তমানে মোট আছে মাত্র **৪২ settled**।

**সিদ্ধান্ত:** edge খোঁজার আগে **data collection সবচেয়ে বড় bottleneck**।
- OTC: 5s timeframe → দ্রুত জমে (৯১ pair × প্রতি ৫s)
- Real: শুধু market hours-এ; session-based collection
- **সময়:** ~1,000 decisive signals/pair পেতে → সপ্তাহ-স্তরের continuous collection

### D.3 Payout (SSID ছাড়া)

**নিয়ম মেনে:** authenticated session ব্যবহার করা হবে না। তাই:
- `payout.py`: per-pair payout **user manually input** করবে (config file)
- Default: conservative 0.85, কিন্তু UI-তে editable
- **সব expectancy calculation payout-parameterized** — user সঠিক payout দিলে
  ফলাফল সাথে সাথে আপডেট

### D.4 Data Quality

| Check | কাজ |
|---|---|
| gap detection | পরপর candle-এর সময়-পার্থক্য अस्वाभाविक হলে flag |
| spike detection | tick return > k×σ হলে flag |
| frozen-pair detection | শেষ N candle flat (real pair বন্ধ) |
| duplicate detection | একই (asset, period, time) |

---

## E. Backtesting Methodology

### E.1 Split কৌশল (তিন-স্তর + purge)

```
|← train →|← embargo →|← validation →|← embargo →|← test →|
   60%          gap          20%           gap        20%
```
- **কখনো shuffle নয়** (time-series)
- **Embargo:** train ও test-এর মাঝে gap — overlapping label leakage বন্ধ
- **Per-pair split** (pooled split নয়)

### E.2 Walk-Forward (বাধ্যতামূলক)

- Expanding বা rolling window
- প্রতিটি block-এ model **re-fit** (বর্তমান `ai_ensemble.py` একবার fit করে — ভুল)
- প্রতিটি fold-এ সম্পূর্ণ মেট্রিক set

### E.3 Multiple Testing Correction

- `research.py` ইতিমধ্যে Bonferroni করে (ভালো) — প্রতিটি নতুন hypothesis-তেও লাগবে
- **Deflated Sharpe Ratio** (Bailey & López de Prado) — কত hypothesis test করা হলো তা হিসাব করে
- একটা hypothesis ভালো দেখালে → **কতগুলো test করা হয়েছিল** সেটা রিপোর্টে থাকবে

### E.4 মেট্রিক Set (প্রত্যেক strategy-র জন্য)

| মেট্রিক | সূত্র |
|---|---|
| Hit-rate + Wilson CI | `(wins, n)` |
| **Payout-adjusted EV** | `hit*(1+payout) − 1` (per trade) |
| Expectancy (ATR units) | `mean(signed_move/atr) − cost` |
| Max drawdown | equity curve |
| Profit factor | gross_win / gross_loss |
| Calmar | return / max_dd |
| p-value | binomial vs breakeven |
| Per-regime breakdown | trend/range আলাদা |
| Per-pair breakdown | consistency |

### E.5 Look-ahead & Leakage Audit (automated)

| Audit | পদ্ধতি |
|---|---|
| **Future-permutation test** | 미래 data shuffle করে দেখব feature বদলায় কিনা — বদলালে look-ahead |
| **Time-disjointness test** | train ∩ test = ∅ (time-এ) assert |
| **Overlap test** | label horizon vs feature window overlap পরীক্ষা |
| **Pooling test** | raw return pool করা নিষিদ্ধ; normalize-first যাচাই |

### E.6 Cost Model

- Spread proxy: ATR-ভিত্তিক (বর্তমান `spread_atr=0.15`)
- **Payout** = parameter (D.3)
- Slippage: optional (OTC-তে কম প্রাসঙ্গিক)

### E.7 Settlement Definition একীভূত করা (bug fix)

**আগের audit-এ পাওয়া অসঙ্গতি:**
- `backtest.py:126`: `move = next.close − next.open`
- `edge.py:97` ও `journal.py:120`: `move = settle.close − entry.close`

→ নতুন `validation.py` **একটাই** definition ব্যবহার করবে (binary option যেভাবে
settle হয় সেভাবে: close-to-close)।

---

## F. Existing Code — কী কী পরিবর্তন হবে

| File | Function | পরিবর্তন | কারণ |
|---|---|---|---|
| `smc.py` | `structure()` (46) | `structure.py`-তে migrate; BOS/CHoCH যোগ | Goal 2,3 |
| `smc.py` | `liquidity_sweep()` (66) | `liquidity.py`-তে expand (pool+displacement) | Goal 1,4 |
| `smc.py` | `order_block()` (91), `fair_value_gap()` (116) | `zones.py`-তে validity সহ migrate | Goal 5 |
| `smart_engine.py` | ADX gate (170) | `regime.py` ব্যবহার করবে | Goal 6 |
| `ensemble.py` | `evaluate()` (123) | measured-reliability weighting | Goal 9 |
| `ensemble.py` | — | NO TRADE gate যোগ | Goal 10 |
| `features.py` | `FEATURES` (13), `compute_features()` (34) | liquidity/structure/regime/tick feature যোগ | Goal 7,8 |
| `candle_store.py` | schema (44) | `source`, `feed_id` column | Provenance |
| `candle_store.py` | `save()` (62) | source tag বাধ্যতামূলক | Provenance |
| `edge.py` | `run()` (75) | payout-adjusted EV + per-regime + purge | Goal E |
| `backtest.py` | `backtest_pair()` (118) | settlement definition unify | Bug fix |
| `research.py` | `build_panel()` (74) | new features + regime conditioning | Goal 6,9 |
| `journal.py` | `settle_open()` (107) | validation.py-এর সাথে definition মেলানো | Bug fix |
| `golive.py` | `gate()` (100) | payout-adjusted expectancy + per-regime | Goal E |
| `analysis.py` | `analyze_*()` | new modules-এর adapter | Integration |
| `dashboard.py` | API | regime/structure/liquidity/calibrated-prob দেখানো | UI |
| `train_model.py` | `walk_forward()` (74) | purged + embargo | Leakage |

**নীতি:** পুরনো engine **মুছে ফেলা হবে না** — বরং নতুন module পাশে থাকবে, এবং
`validation.py` দেখাবে নতুনটা আসলেই পুরনোটার চেয়ে ভালো কিনা।

---

## G. নতুন Python Files

| File | কাজ | Goal |
|---|---|---|
| `structure.py` | swing, BOS, CHoCH, MSS, internal/external | 2, 3 |
| `liquidity.py` | EQH/EQL, pools, sweep+reclaim, displacement, session levels | 1, 4 |
| `zones.py` | Order Block/FVG + validity statistics | 5 |
| `regime.py` | trend/range/chop + vol + Hurst + variance ratio + session | 6 |
| `microstructure.py` | tick intensity, size dist, realised vol, bursts | 7 |
| `decision.py` | weighted ensemble + NO TRADE gate + EV | 9, 10 |
| `validation.py` | **কেন্দ্রীয় harness** — purged walk-forward, metrics, power | E |
| `provenance.py` | otc/real tagging, data quality, gap/spike detection | D |
| `payout.py` | payout parameter (user-input), per-pair config | D.3 |
| `calibrate.py` | probability calibration + reliability diagram | C |
| `power.py` | sample-size / statistical power calculator | D.2 |
| `audit_leakage.py` | automated look-ahead/leakage test suite | E.5 |
| `test_structure.py` | unit tests | H |
| `test_liquidity.py` | unit tests | H |
| `test_zones.py` | unit tests | H |
| `test_regime.py` | unit tests | H |
| `test_microstructure.py` | unit tests | H |
| `test_decision.py` | unit tests | H |
| `test_validation.py` | known-answer tests (Wilson, EV, p-value) | H |
| `test_leakage.py` | leakage audit tests | H |

---

## H. Testing & Validation Plan

### H.1 প্রতি Module-এর ৫-স্তরের পরীক্ষা

প্রতিটি নতুন module-কে নিচের ৫টি gate পার হতে হবে (ক্রমে):

| Gate | পরীক্ষা | Pass মানে |
|---|---|---|
| **G1 Unit** | deterministic correctness tests | সব test pass |
| **G2 Look-ahead** | future-permutation test | feature অপরিবর্তিত |
| **G3 Synthetic-negative** | random-walk data-তে চালানো | hit-rate ~50% (ভুয়া edge নেই) |
| **G4 Synthetic-positive** | injected known pattern | detector সেটা খুঁজে পায় (প্রমাণ code শেখে) |
| **G5 Real-data** | purged walk-forward, OTC/real আলাদা | payout-adjusted EV CI > 0 |

**G3 + G4 একসাথে গুরুত্বপূর্ণ:** G3 বলে "মিথ্যা edge বানায় না", G4 বলে "আসল edge
থাকলে খুঁজে পায়"। বর্তমান `test_patterns.py`-এ এই ধারণা আছে (control test) —
সেটাই expand করা হবে।

### H.2 Cross-cutting Tests

| Test | কাজ |
|---|---|
| Settlement consistency | `backtest`/`edge`/`journal` একই definition ব্যবহার করে |
| Pooling guard | raw return pool করলে test fail করে |
| Provenance guard | OTC+real একসাথে pool করলে test fail করে |
| Metric known-answers | Wilson/p-value/EV হাতে-হিসাব করা মানের সাথে মেলে |
| Regression golden-file | frozen candle sample-এ output অপরিবর্তিত |

### H.3 Model Validation

- Reliability diagram (calibration curve) প্রতিটি model-এর জন্য
- Brier score vs baseline
- Permutation importance stability (fold-to-fold)

---

## I. Implementation-এর ধাপসমূহ (Phases)

প্রতিটি phase **আগের phase-এর gate pass** হলেই শুরু হবে।

| Phase | কাজ | Deliverable |
|---|---|---|
| **P0** | Provenance + data quality + payout parameter | `provenance.py`, `payout.py`, `candle_store` schema update |
| **P1** | Validation harness | `validation.py`, `power.py`, `audit_leakage.py` |
| **P2** | Structure engine (BOS/CHoCH/MSS) | `structure.py` + tests |
| **P3** | Liquidity + displacement | `liquidity.py` + tests |
| **P4** | Zones + validity | `zones.py` + tests |
| **P5** | Regime detection | `regime.py` + tests |
| **P6** | Microstructure (tick) | `microstructure.py` + tests |
| **P7** | Feature integration + ML calibration | `features.py`, `calibrate.py` |
| **P8** | Ensemble + NO TRADE | `decision.py` |
| **P9** | Full validation + decision | রিপোর্ট: কোন strategy validated, কোনটা বাদ |

**ক্রম কেন এমন:** P0-P1 আগে — কারণ validation ছাড়া নতুন strategy মাপা যায় না।
Structure আগে (P2), কারণ liquidity/zones structure-এর উপর দাঁড়ায়।

---

## J. প্রতি Phase-এর Success Criteria

| Phase | সফল হলে | ব্যর্থ হলে |
|---|---|---|
| **P0** | সব row-এ source tag; gap/spike detect; payout config কাজ করে | collection ঠিক নয় → data layer আবার |
| **P1** | harness purged walk-forward দেয়; leakage audit pass; power calc সঠিক | harness ভুল → **সব থামান**, এটাই ভিত্তি |
| **P2** | BOS/CHoCH synthetic pattern-এ ধরা পড়ে; random-walk-এ 50% | structure concept কাজ করে না → research-only |
| **P3** | sweep+displacement random-walk-এ 50%, injected-এ ধরা পড়ে | বাতিল বা research-only |
| **P4** | zone respect rate **মাপা**; random-এর চেয়ে আলাদা কিনা পরীক্ষা | zone ভুয়া → research-only |
| **P5** | regime persist করে (random-এর চেয়ে বেশি); ভবিষ্যৎ vol predict করে | regime অর্থহীন → বাদ |
| **P6** | tick feature কোনো তথ্য বহন করে (MI > 0, held-out) | (প্রত্যাশিত) তথ্য নেই → সৎভাবে বাদ |
| **P7** | calibrated prob-এর Brier baseline-এর চেয়ে ভালো | calibration ব্যর্থ → model বাদ |
| **P8** | NO TRADE false signal কমায়, EV বাড়ায় (held-out) | উন্নতি নেই → gate বাতিল |
| **P9** | ≥1 strategy payout-adjusted EV CI > 0 (test, ≥2 regime, ≥5 pair) | কোনোটা না → "no tradable edge" রিপোর্ট |

**Overarching success:** একটি strategy "validated" শুধু তখনই যদি — purged
walk-forward-এ **payout-adjusted EV-র CI lower bound > 0**, ≥5 pair, ≥2 regime,
drawdown limit-এর মধ্যে। নাহলে → `research-only` বা বাদ।

---

## Development Roadmap (বাস্তবসম্মত)

| Stage | সময় | কাজ | Milestone |
|---|---|---|---|
| **Stage 1** | ২–৩ দিন | P0 + P1 (foundation) | validation harness চালু |
| **Stage 2** | ১ সপ্তাহ | P2 + P3 (structure + liquidity) | structural detectors + tests |
| **Stage 3** | ১ সপ্তাহ | P4 + P5 (zones + regime) | validity stats + regime engine |
| **Stage 4** | ৩–৪ দিন | P6 (microstructure) | tick verdict (হ্যাঁ/না) |
| **Stage 5** | ১ সপ্তাহ | P7 + P8 (ML + decision) | calibrated model + NO TRADE |
| **Stage 6** | চলমান | P9 + **ডেটা সংগ্রহ** | প্রথম সৎ verdict |
| **Stage 7** | ২–৪ সপ্তাহ | পর্যাপ্ত ডেটা → পুনরায় validate | final verdict |

**⚠️ সময়ের সবচেয়ে বড় বাধা:** D.2-এর power calculation। ~1,000+ decisive
signal/strategy/pair দরকার — তাই **collection ব্যাকগ্রাউন্ডে সপ্তাহ-জুড়ে চলবে**,
আর code ওই সময়ে তৈরি হবে। Code আগে, ডেটা পরে।

---

## সৎ প্রত্যাশা ও Kill Criteria

### যা আশা করা যুক্তিসঙ্গত

- Structure/liquidity/regime module-গুলো **research quality** বাড়াবে
- NO TRADE gate সম্ভবত false signal কমাবে (কিন্তু edge বানাবে না)
- Real (non-OTC) pair-এ session-এ **হয়তো** সামান্য structure আছে

### যা সম্ভবত হবে (আগের প্রমাণের ভিত্তিতে)

- OTC pair-এ কোনো validated edge **নাও** পাওয়া যেতে পারে
- Tick microstructure-এ তথ্য **নাই** (আগেই মাপা: autocorr ≈ 0)
- Payout hardcoded-এর কারণে breakeven ~৫৪% — যা ছোঁয়া কঠিন

### Kill Criteria (স্পষ্ট বাতিলের শর্ত)

একটি strategy বাদ/research-only যদি:
1. Random-walk control-এ ~50%-এর বেশি দেখায় (মানে ভুয়া edge বানায়), **অথবা**
2. Injected synthetic pattern-এ ধরতে না পারে (মানে code কাজ করে না), **অথবা**
3. Purged walk-forward-এ payout-adjusted EV CI lower bound ≤ 0, **অথবা**
4. Validation-এ ভালো কিন্তু test-এ ভেঙে পড়ে (data-snooping — আগে `invert.py`-এ
   এই ফাঁদ ধরা পড়েছিল)

### চূড়ান্ত সিদ্ধান্ত হবে এই দুটোর একটি

| ফলাফল | অর্থ |
|---|---|
| **≥1 strategy validated** | একটা **small, controlled** paper-trade দিয়ে forward test |
| **কোনোটা validated নয়** | সৎভাবে রিপোর্ট: "এই ফিডে এই method-গুলো দিয়ে tradable edge নেই" |

দুটোই **সফল ফলাফল** — কারণ দুটোই সত্য, এবং দ্বিতীয়টা আপনাকে টাকা হারানো থেকে বাঁচায়।

---

## সংক্ষেপে: ১০টি Goal → সমাধান

| Goal | Module |
|---|---|
| 1. Liquidity Detection | `liquidity.py` |
| 2. SMC + ICT + Structure | `structure.py` + `zones.py` + `liquidity.py` |
| 3. BOS + CHoCH | `structure.py` |
| 4. Liquidity Sweep + Displacement | `liquidity.py` |
| 5. Order Block + FVG validity | `zones.py` |
| 6. Market Regime Detection | `regime.py` |
| 7. Tick-level analysis | `microstructure.py` |
| 8. ML next-candle probability | `features.py` + `calibrate.py` |
| 9. Multi-strategy ensemble | `decision.py` |
| 10. Intelligent NO TRADE | `decision.py` |
| (ভিত্তি) | `validation.py`, `provenance.py`, `payout.py`, `power.py` |

> **এক লাইনে:** বেশি signal নয় — **প্রমাণযোগ্য edge**। আর প্রমাণ না মিললে,
> সেটাও একটা প্রমাণ। 🤖

---

**সম্পন্ন।** এটি শুধু design ও roadmap — কোনো কোড পরিবর্তন করা হয়নি।
