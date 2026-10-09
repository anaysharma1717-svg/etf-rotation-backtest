# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

"""
Weekly adaptive momentum / reversal ETF rotation  (QuantConnect, Python)

Universe : SPY QQQ IWM EFA EEM XLK XLF XLE TLT GLD VNQ HYG     Defensive asset: BIL
Signals  : previous close only. Decisions are made after the close, orders are
           market-on-open for the next session. Fees = Interactive Brokers model,
           plus per-ETF slippage.

REGIMES (checked every day, highest priority first)
  EXTREME  : VIX/VIX3M > 1.20 AND VIX 20d z-score > 2
             -> 50% BIL, up to 50% in TLT and GLD (positive score only; 40% per-asset cap)
  REVERSAL : VIX/VIX3M > 1.05 OR (VIX z-score > 1.5 AND VIX/VIX3M > 1.0)
             -> up to 3 assets above their 200d SMA with positive 3m return that had the
                largest normalized 5d drop (5d return / (20d daily vol * sqrt 5))
  MOMENTUM : SPY > 200d SMA and no stress
             -> score = blended 1m/3m/6m returns / annualized 20d vol (ranking score, not Sharpe),
                only assets above 100d SMA,
                hold top 3
  BEAR     : SPY < 200d SMA and no stress (not covered by the brief, see MODIFICATIONS)
             -> same as MOMENTUM but only 50% invested

SIZING   : capped inverse 20d volatility with redistribution; infeasible residual -> BIL.
CLUSTER  : at most 2 picks from SPY/QQQ/XLK (switchable; not a full correlation model).
CHURN    : (momentum / bear) a holding is sold if it drops below its 100d SMA or falls out of
           the top 5; an existing holding is only swapped for a new candidate whose score is
           at least 10% better AND at least 0.02 score points better (switchable).
REBALANCE: weekly (decided after the first session of each week, traded next open).

MODIFICATIONS added on top of the brief (all switchable below)
  1. BEAR regime (BEAR_EXPOSURE): the brief does not say what to do when SPY is below its
     200d SMA but VIX is calm. Default runs momentum at 50% exposure. Set 0.0 for 100% BIL.
  2. REQUIRE_POS_SCORE: momentum picks must have a positive score (absolute momentum check),
     so the strategy does not buy "least bad" losers.
  3. DAILY_STRESS_CHECK: an EXTREME signal is acted on at the next open instead of waiting
     up to a week (crashes do not wait for Monday).
  4. KEEP_RANK = 5 defines the "top ranking group" a holding must stay inside.
  5. MIN_TRADE = 2%: weight changes smaller than this are skipped (fewer tiny trades).

BENCHMARKS (simulated inside the algorithm on the same daily data, signal at close,
fill at next open, costs per side): buy & hold SPY, buy & hold QQQ, and the TQQQ trend
strategy built earlier (QQQ 200d SMA with 1% hysteresis, vol target 50%, VIX/VIX3M stress
scale, RSI/ATR trim, BIL when off).

OUTPUT: the Logs tab prints CAGR, Sharpe, Sortino, max drawdown, volatility, turnover,
trade count and yearly returns for the strategy and every benchmark, plus regime counts.
Sharpe/Sortino use same-date adjusted BIL returns as a risk-free proxy in these custom logs.
QC headline statistics are separate and are NOT changed by metrics().
This patch has local tests only: run a QuantConnect smoke test before interpreting results.
"""

# ============================================================== settings

RUN = "full"        # "full" = 2010-03 to latest, "dev" = 2010-03 to 2023-12, "oos" = 2024-01 to latest
CASH = 100_000

RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLE", "TLT", "GLD", "VNQ", "HYG"]
SAFE = "BIL"
DEFENSIVE = ["TLT", "GLD"]
BENCH_ONLY = ["TQQQ"]

# momentum
W1, W3, W6 = 0.40, 0.40, 0.20
L1, L3, L6 = 21, 63, 126
VOL_N = 20
TREND_N = 100
LONG_N = 200
TOP_N = 3
KEEP_RANK = 5
REPLACE_EDGE = 0.10
REPLACE_FLOOR = 0.02   # absolute score points, not a return; 0.0 disables the floor
USE_CLUSTER_CAP = True
EQUITY_BETA = {"QQQ", "XLK", "SPY"}
MAX_CLUSTER_PICKS = 2
MAX_W = 0.40
REQUIRE_POS_SCORE = True

# regimes
Z_N = 20
RATIO_REV, Z_REV, RATIO_FLOOR = 1.05, 1.5, 1.0
RATIO_EXT, Z_EXT = 1.20, 2.0
REV_N, REV_LOOK = 3, 5
EXT_SAFE, DEF_N = 0.50, 2
BEAR_EXPOSURE = 0.50
DAILY_STRESS_CHECK = True

# Research protocol (NOT an optimizer; this helper only lists one-at-a-time trials):
# 1. Run RUN="dev" first and save baseline + sensitivity results after costs.
# 2. Run each sensitivity_cases() dict by editing ONLY those settings above.
#    Evaluate drawdown, turnover, regime days, and excess Sharpe; seek plateaus.
# 3. Freeze rules, then run RUN="oos" ONCE. Already-viewed 2024+ data is NOT pristine
#    OOS, so treat it as a holdout diagnostic; future unseen data is the real test.
# 4. Warmup may use pre-2024 prices in oos (needed for signals), but no pre-start trades.
# 5. Run ablations: USE_CLUSTER_CAP=False, REPLACE_FLOOR=0.0; change one rule at a time.
# These local changes are hypotheses, not evidence of higher returns.

def sensitivity_cases():
    cases = [{"label": "baseline"}]
    for name, values in {
        "RATIO_REV": (1.03, 1.05, 1.07), "RATIO_EXT": (1.15, 1.20, 1.25),
        "Z_REV": (1.25, 1.50, 1.75), "Z_EXT": (1.75, 2.00, 2.25),
        "KEEP_RANK": (4, 5, 6)
    }.items():
        for value in values:
            if value != globals()[name]:
                cases.append({"label": f"{name}={value}", name: value})
    return cases


# execution
MIN_TRADE = 0.02
CASH_BUFFER = 0.995
SLIPPAGE = {"SPY": 0.0001, "QQQ": 0.0001, "IWM": 0.0002, "EFA": 0.0002, "EEM": 0.0003,
            "XLK": 0.0002, "XLF": 0.0002, "XLE": 0.0002, "TLT": 0.0002, "GLD": 0.0002,
            "VNQ": 0.0003, "HYG": 0.0003, "BIL": 0.0001, "TQQQ": 0.0005}
SIM_COST_BPS = {"SPY": 2, "QQQ": 2, "TQQQ": 6, "BIL": 2}

# TQQQ trend benchmark (same settings as the earlier Nasdaq strategy)
TQ_SMA, TQ_HYST, TQ_TV = 200, 0.01, 0.50
TQ_VIX_LO, TQ_VIX_HI = 0.95, 1.10
TQ_RSI_N, TQ_RSI_HI, TQ_EMA_N, TQ_ATR_N, TQ_EXT_ATR, TQ_RSI_CUT = 10, 80, 20, 14, 2.0, 0.25
TQ_THRESH = 0.10

MODES = ["MOMENTUM", "BEAR", "REVERSAL", "EXTREME"]


# ============================================================== helpers (pure functions)

def sma(c, n):
    return float(np.mean(c[-n:])) if len(c) >= n else None


def ret(c, n):
    return c[-1] / c[-1 - n] - 1.0 if len(c) > n and c[-1 - n] > 0 else None


def daily_vol(c, n):
    if len(c) < n + 1:
        return None
    r = np.diff(np.log(np.asarray(c[-(n + 1):], dtype=float)))
    s = float(r.std(ddof=1))
    return s if s > 0 else None


def mom_score(c):
    r1, r3, r6, v = ret(c, L1), ret(c, L3), ret(c, L6), daily_vol(c, VOL_N)
    if None in (r1, r3, r6, v):
        return None
    # Annualization is constant across assets: ranks unchanged; this is not a Sharpe.
    return (W1 * r1 + W3 * r3 + W6 * r6) / (v * np.sqrt(252))


def inv_vol(picks, closes, gross):
    """Waterfall capped inverse-vol weights. Infeasible residual goes to BIL, logged later."""
    if gross <= 0 or MAX_W <= 0:
        return {}
    iv = {}
    for a in dict.fromkeys(picks):
        v = daily_vol(closes[a], VOL_N)
        if v is not None and np.isfinite(v) and v > 0:
            iv[a] = 1.0 / v
    remaining = min(float(gross), len(iv) * MAX_W)
    weights, active = {}, dict(iv)
    while active and remaining > 1e-12:
        total = sum(active.values())
        proposed = {a: remaining * x / total for a, x in active.items()}
        capped = [a for a, w in proposed.items() if w > MAX_W + 1e-12]
        if not capped:
            weights.update(proposed)
            break
        for a in capped:
            weights[a] = MAX_W
            remaining -= MAX_W
            del active[a]
    return weights


def vix_z(vix_hist):
    if len(vix_hist) < Z_N:
        return None
    x = np.asarray(list(vix_hist)[-Z_N:], dtype=float)
    s = x.std(ddof=1)
    return float((x[-1] - x.mean()) / s) if s > 0 else 0.0


def regime(spy_closes, ratio, z):
    r = ratio if ratio is not None else 1.0
    zz = z if z is not None else 0.0
    if r > RATIO_EXT and zz > Z_EXT:
        return "EXTREME"
    if r > RATIO_REV or (zz > Z_REV and r > RATIO_FLOOR):
        return "REVERSAL"
    s = sma(spy_closes, LONG_N)
    if s is not None and spy_closes[-1] > s:
        return "MOMENTUM"
    return "BEAR"


def better(new, old):
    return new >= old + max(REPLACE_EDGE * abs(old), REPLACE_FLOOR)


def cluster_ok(picks):
    return (not USE_CLUSTER_CAP or
            sum(a in EQUITY_BETA for a in picks) <= MAX_CLUSTER_PICKS)


def select_momentum(closes, held, gross):
    """Returns (weights, picks, scores). Retain eligible top-group holdings before filling."""
    scores = {}
    for a in RISK:
        c = closes[a]
        s, m = mom_score(c), sma(c, TREND_N)
        if s is None or m is None or c[-1] <= m:
            continue
        if REQUIRE_POS_SCORE and s <= 0:
            continue
        scores[a] = s
    ranked = sorted(scores, key=scores.get, reverse=True)
    group = ranked[:KEEP_RANK]
    picks = []
    # Keep the strongest eligible holdings; the cap overrides KEEP_RANK only when necessary.
    for a in ranked:
        if a in held and a in group and len(picks) < TOP_N and cluster_ok(picks + [a]):
            picks.append(a)
    cands = [a for a in ranked if a not in picks]
    for a in list(cands):
        if len(picks) >= TOP_N:
            break
        if cluster_ok(picks + [a]):
            picks.append(a)
            cands.remove(a)
    # Search feasible replacements, not just cands[0]: the best candidate may violate the cap.
    for a in cands:
        for old in sorted(picks, key=scores.get):
            trial = [a if x == old else x for x in picks]
            if old in held and better(scores[a], scores[old]) and cluster_ok(trial):
                picks = trial
                break
    return inv_vol(picks, closes, gross), picks, scores


def select_reversal(closes):
    drops = {}
    for a in RISK:
        c = closes[a]
        m, r3, r5, v = sma(c, LONG_N), ret(c, L3), ret(c, REV_LOOK), daily_vol(c, VOL_N)
        if None in (m, r3, r5, v) or c[-1] <= m or r3 <= 0:
            continue
        nd = r5 / (v * np.sqrt(REV_LOOK))
        if nd < 0:
            drops[a] = nd
    picks = []
    for a in sorted(drops, key=drops.get):
        if len(picks) < REV_N and cluster_ok(picks + [a]):
            picks.append(a)
    return inv_vol(picks, closes, 1.0), picks, drops


def select_extreme(closes):
    sc = {a: mom_score(closes[a]) for a in DEFENSIVE}
    sc = {a: s for a, s in sc.items() if s is not None and s > 0}
    picks = sorted(sc, key=sc.get, reverse=True)[:DEF_N]
    return inv_vol(picks, closes, 1.0 - EXT_SAFE), picks, sc


# ---------------------------------------------------------------- metrics

def metrics(rets, risk_free):
    r = np.asarray(rets, dtype=float)
    rf = np.asarray(risk_free, dtype=float)
    if r.shape != rf.shape or not np.all(np.isfinite(r)) or not np.all(np.isfinite(rf)):
        raise ValueError("Returns and BIL proxy must be finite and same-date aligned")
    if len(r) < 20:
        return None
    eq = np.cumprod(1 + r)
    yrs = len(r) / 252.0
    cagr = eq[-1] ** (1 / yrs) - 1
    sd = r.std(ddof=1)
    excess = r - rf
    ex_sd = excess.std(ddof=1)
    dn = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    mdd = float(((np.r_[1.0, eq] - peak) / peak).min())
    return {"cagr": cagr, "sharpe": excess.mean() / ex_sd * np.sqrt(252) if ex_sd > 0 else 0.0,
            "sortino": excess.mean() / dn * np.sqrt(252) if dn > 0 else 0.0,
            "mdd": mdd, "vol": sd * np.sqrt(252), "years": yrs, "total": eq[-1] - 1}


def yearly(dated):
    out = {}
    for d, x in dated:
        out.setdefault(d.year, []).append(x)
    return {y: float(np.prod(1 + np.array(v)) - 1) for y, v in out.items()}


# ---------------------------------------------------------------- TQQQ trend helpers

def ema(x, n):
    if len(x) < n:
        return None
    a, e = 2.0 / (n + 1), float(np.mean(x[:n]))
    for v in x[n:]:
        e = a * v + (1 - a) * e
    return e


def wilder_rsi(x, n):
    if len(x) < n + 1:
        return None
    d = np.diff(np.asarray(x, dtype=float))
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for u, w in zip(up[n:], dn[n:]):
        au, ad = (au * (n - 1) + u) / n, (ad * (n - 1) + w) / n
    return 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)


def wilder_atr(h, l, c, n):
    if len(c) < n + 1:
        return None
    h, l, c = (np.asarray(v, dtype=float) for v in (h, l, c))
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    a = tr[:n].mean()
    for t in tr[n:]:
        a = (a * (n - 1) + t) / n
    return a


# ---------------------------------------------------------------- benchmark simulator

class Sim:
    """Virtual portfolio: target set at a close, executed at the next open, costs per side."""

    def __init__(self, name):
        self.name = name
        self.h = {"CASH": 1.0}
        self.pending = None
        self.rets, self.prev = [], 1.0
        self.turnover, self.trades = 0.0, 0
        self.on = None

    def value(self):
        return sum(self.h.values())

    def weights(self):
        v = self.value()
        return {a: x / v for a, x in self.h.items()}

    def step(self, date, opens, closes, prev_closes, record):
        for a in self.h:
            if a != "CASH" and a in opens and prev_closes.get(a, 0) > 0:
                self.h[a] *= opens[a] / prev_closes[a]
        if self.pending is not None:
            self._execute(self.pending, opens)
            self.pending = None
        for a in self.h:
            if a != "CASH" and a in closes and opens.get(a, 0) > 0:
                self.h[a] *= closes[a] / opens[a]
        v = self.value()
        if record:
            self.rets.append((date, v / self.prev - 1.0))
        self.prev = v

    def _execute(self, target, opens):
        v = self.value()
        new = {a: w * v for a, w in target.items() if w > 0 and a in opens}
        new["CASH"] = new.get("CASH", 0.0) + max(0.0, v - sum(new.values()))
        cost = 0.0
        for a in set(new) | set(self.h):
            if a == "CASH":
                continue
            d = abs(new.get(a, 0.0) - self.h.get(a, 0.0))
            if d > 1e-9:
                self.trades += 1
            cost += d * SIM_COST_BPS.get(a, 3) / 1e4
            self.turnover += d / v    # gross buys + sells, same convention as strategy fills
        self.h = {a: x * (v - cost) / v for a, x in new.items()}

    def decide_tqqq(self, q, ratio):
        c = list(q["c"])
        s = sma(c, TQ_SMA)
        v = daily_vol(c, VOL_N)
        if s is None or v is None:
            return
        v *= np.sqrt(252)
        px, was = c[-1], self.on
        if self.on is None:
            self.on = px > s
        elif self.on and px < s * (1 - TQ_HYST):
            self.on = False
        elif not self.on and px > s * (1 + TQ_HYST):
            self.on = True
        w = 1.0 if self.on else 0.0
        w *= min(1.0, TQ_TV / (3.0 * v))
        if ratio is not None:
            w *= min(1.0, max(0.0, (TQ_VIX_HI - ratio) / (TQ_VIX_HI - TQ_VIX_LO)))
        tail = c[-260:]
        rsi, e = wilder_rsi(tail, TQ_RSI_N), ema(tail, TQ_EMA_N)
        atr = wilder_atr(list(q["h"])[-260:], list(q["l"])[-260:], tail, TQ_ATR_N)
        if self.on and rsi and e and atr and rsi > TQ_RSI_HI and (px - e) / atr > TQ_EXT_ATR:
            w *= 1.0 - TQ_RSI_CUT
        cur = self.weights().get("TQQQ", 0.0)
        major = (was is not None and was != self.on) or ((w == 0) != (cur < 1e-3))
        if major or abs(w - cur) > TQ_THRESH or self.weights().get("CASH", 0) > 0.5:
            self.pending = {"TQQQ": w, "BIL": 1.0 - w}


# ============================================================== algorithm

class SlippageInit(BrokerageModelSecurityInitializer):
    def __init__(self, model, seeder):
        super().__init__(model, seeder)

    def initialize(self, security: Security) -> None:
        super().initialize(security)
        if security.type == SecurityType.EQUITY:
            security.set_slippage_model(ConstantSlippageModel(SLIPPAGE.get(security.symbol.value, 0.0003)))
            security.set_leverage(2.0)     # lets same-open sells fund buys without rejections


class AdaptiveMomentumReversalRotation(QCAlgorithm):

    def initialize(self) -> None:
        if RUN not in ("full", "dev", "oos"):
            raise ValueError("RUN must be full, dev, or oos")
        if not (0 < MAX_W <= 1 and 0 <= EXT_SAFE <= 1 and 0 <= BEAR_EXPOSURE <= 1):
            raise ValueError("Invalid allocation limits")
        if KEEP_RANK < TOP_N or TOP_N < 1 or MAX_CLUSTER_PICKS < 1:
            raise ValueError("Invalid ranking/cluster limits")
        if RUN == "oos":
            self.set_start_date(2024, 1, 1)
        else:
            self.set_start_date(2010, 3, 1)
        if RUN == "dev":
            self.set_end_date(2023, 12, 31)
        self.set_cash(CASH)
        self.set_warm_up(timedelta(days=420))
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.set_security_initializer(SlippageInit(self.brokerage_model,
                                                   FuncSecuritySeeder(self.get_last_known_prices)))
        self.sym = {a: self.add_equity(a, Resolution.DAILY,
                                         data_normalization_mode=DataNormalizationMode.ADJUSTED).symbol for a in RISK + [SAFE] + BENCH_ONLY}
        self.set_benchmark(self.sym["SPY"])
        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.vix3m = self.add_data(CBOE, "VIX3M", Resolution.DAILY).symbol

        self.closes = {a: deque(maxlen=300) for a in self.sym}
        self.qqq = {"h": deque(maxlen=300), "l": deque(maxlen=300), "c": self.closes["QQQ"]}
        self.vix_hist = deque(maxlen=Z_N)
        self.prev_closes = {}
        self.sims = {"BH_SPY": Sim("BH_SPY"), "BH_QQQ": Sim("BH_QQQ"), "TQQQ_TREND": Sim("TQQQ_TREND")}

        self.last_date = None
        self.last_week = None
        self.mode = None
        self.picks = []
        self.started = False
        self.eq = []                 # (date, equity)
        self.rf = {}                 # same-date adjusted BIL close-to-close return
        self.cap_residuals = []       # (date, mode, requested gross, allocated gross)
        self.mode_days = {m: 0 for m in MODES}
        self.switches = 0
        self.n_trades = 0
        self.traded = 0.0
        self.trade_log = []

    # ---------------------------------------------------------------- data
    def on_data(self, data: Slice) -> None:
        spy = self.sym["SPY"]
        if not data.bars.contains_key(spy):
            return
        bt = data.bars[spy].time
        date = datetime(bt.year, bt.month, bt.day)
        if date == self.last_date:
            return
        self.last_date = date

        opens, today = {}, {}
        for a, s in self.sym.items():
            if data.bars.contains_key(s):
                b = data.bars[s]
                opens[a], today[a] = float(b.open), float(b.close)
                self.closes[a].append(float(b.close))
                if a == "QQQ":
                    self.qqq["h"].append(float(b.high))
                    self.qqq["l"].append(float(b.low))

        v, v3 = float(self.securities[self.vix].price), float(self.securities[self.vix3m].price)
        ratio = v / v3 if v > 0 and v3 > 0 else None
        if v > 0:
            self.vix_hist.append(v)
        z = vix_z(self.vix_hist)

        trading = not self.is_warming_up
        # benchmarks: move to today, then decide at today's close for tomorrow's open
        for name, s in self.sims.items():
            s.step(date, opens, today, self.prev_closes, trading and self.started)
        if trading and self.started and SAFE in today and self.prev_closes.get(SAFE, 0) > 0:
            self.rf[date] = today[SAFE] / self.prev_closes[SAFE] - 1.0
        self.prev_closes.update(today)
        if not trading:
            return
        if not self.started:
            self.sims["BH_SPY"].pending = {"SPY": 1.0}
            self.sims["BH_QQQ"].pending = {"QQQ": 1.0}
        if "TQQQ" in self.prev_closes:
            self.sims["TQQQ_TREND"].decide_tqqq(self.qqq, ratio)

        if self.started:
            self.eq.append((date, float(self.portfolio.total_portfolio_value)))
        if any(len(self.closes[a]) < LONG_N + 1 for a in RISK):
            return

        mode = regime(list(self.closes["SPY"]), ratio, z)
        week = date.isocalendar()[:2]
        weekly = week != self.last_week
        # Entry into EXTREME is urgent; exits remain weekly by design, not daily.
        urgent = DAILY_STRESS_CHECK and mode == "EXTREME" and self.mode != "EXTREME"
        if self.started:
            self.mode_days[mode] += 1
        if weekly or urgent or not self.started:
            self.last_week = week
            self._rebalance(date, mode, ratio, z)
        self.started = True

        if date.weekday() == 4:
            self.plot("Regime", "mode", MODES.index(self.mode) if self.mode else -1)
            for name, s in self.sims.items():
                self.plot("Benchmarks (x start cash)", name, s.value() * CASH)

    # ---------------------------------------------------------------- decisions
    def _rebalance(self, date, mode, ratio, z):
        closes = {a: list(self.closes[a]) for a in RISK}
        if mode == "EXTREME":
            w, picks, _ = select_extreme(closes)
        elif mode == "REVERSAL":
            w, picks, _ = select_reversal(closes)
        else:
            held = self.picks if self.mode in ("MOMENTUM", "BEAR") else []
            gross = 1.0 if mode == "MOMENTUM" else BEAR_EXPOSURE
            w, picks, _ = (select_momentum(closes, held, gross) if gross > 0 else ({}, [], {}))
        requested = 1.0 - EXT_SAFE if mode == "EXTREME" else (
            BEAR_EXPOSURE if mode == "BEAR" else 1.0)
        allocated = sum(w.values())
        if requested - allocated > 1e-8:
            self.cap_residuals.append((date, mode, requested, allocated))
            self.log(f"{date.date()} {mode}: requested gross={requested:.1%}, "
                     f"allocated={allocated:.1%}; residual={requested-allocated:.1%} to BIL "
                     "(too few eligible picks / per-asset cap)")
        if mode != self.mode:
            self.switches += 1
        self.mode, self.picks = mode, picks
        target = dict(w)
        target[SAFE] = max(0.0, 1.0 - sum(w.values()))
        if len(self.trade_log) < 40:
            self.trade_log.append(f"{date.date()} {mode:8s} " +
                                  " ".join(f"{a}:{x:.2f}" for a, x in sorted(target.items()) if x > 0.005))
        self._trade(target)

    def _trade(self, target):
        tpv = float(self.portfolio.total_portfolio_value)
        if tpv <= 0:
            return
        for a in RISK + [SAFE]:
            s = self.sym[a]
            px = float(self.securities[s].price)
            if px <= 0:
                continue
            w = target.get(a, 0.0) * CASH_BUFFER
            have = int(self.portfolio[s].quantity)
            cur = have * px / tpv
            if w == 0 and have != 0:
                self.market_on_open_order(s, -have, tag="exit")
            elif abs(w - cur) >= MIN_TRADE:
                q = int(w * tpv / px) - have
                if q != 0:
                    self.market_on_open_order(s, q, tag=f"{self.mode} w={w:.2f}")

    def on_order_event(self, e: OrderEvent) -> None:
        if e.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.traded += abs(float(e.fill_quantity) * float(e.fill_price))
        if e.status == OrderStatus.FILLED:
            self.n_trades += 1

    # ---------------------------------------------------------------- report
    def on_end_of_algorithm(self) -> None:
        if len(self.eq) < 30:
            self.log("Not enough data for a report.")
            return
        eq = np.array([x for _, x in self.eq])
        dates = [d for d, _ in self.eq]
        strat = [(dates[i], eq[i] / eq[i - 1] - 1) for i in range(1, len(eq))]
        rows = {"STRATEGY": (strat, self.traded / eq.mean(), self.n_trades)}
        for n, s in self.sims.items():
            rows[n] = (s.rets, s.turnover, s.trades)

        # Compare all custom rows over the same dates. Never silently assume rf=0.
        common_dates = set(d for d, _ in strat)
        for r, _, _ in rows.values():
            common_dates &= {d for d, _ in r}
        missing_rf = common_dates - set(self.rf)
        if missing_rf:
            self.log(f"BIL proxy missing on {len(missing_rf)} report dates; custom report skipped.")
            return
        rows = {n: ([(d, x) for d, x in r if d in common_dates], turn, tr)
                for n, (r, turn, tr) in rows.items()}
        L = [f"RUN={RUN}  {dates[0].date()} -> {dates[-1].date()}",
             "name           CAGR   Sharpe Sortino  MaxDD    Vol  Total   Turn/yr  Trades"]
        for n, (r, turn, tr) in rows.items():
            m = metrics([x for _, x in r], [self.rf[d] for d, _ in r])
            if not m:
                continue
            tpy = turn / m["years"]
            L.append(f"{n:12s} {m['cagr']*100:6.1f}% {m['sharpe']:6.2f} {m['sortino']:6.2f} "
                     f"{m['mdd']*100:6.1f}% {m['vol']*100:5.1f}% {m['total']*100:6.0f}% "
                     f"{tpy*100:6.0f}% {tr:7d}")
        yrs = sorted({d.year for d, _ in strat})
        L.append("Yearly %     " + " ".join(f"{y % 100:5d}" for y in yrs))
        for n, (r, _, _) in rows.items():
            yr = yearly(r)
            L.append(f"{n:12s} " + " ".join(f"{yr.get(y, 0)*100:5.0f}" for y in yrs))
        tot = max(1, sum(self.mode_days.values()))
        L.append("Regime days: " + ", ".join(f"{m} {self.mode_days[m]} ({self.mode_days[m]/tot*100:.0f}%)"
                                             for m in MODES) + f"; switches {self.switches}")
        L.append("Turnover = gross buys + sells / equity / year (no /2; strategy uses average equity). Sharpe/Sortino use BIL proxy.")
        L.append(f"Allocation residuals to BIL: {len(self.cap_residuals)} rebalances (details logged).")
        L.append("BIL proxy has small ETF costs/risk; custom stats do not change QC headline stats.")
        L.append("-- first rebalances --")
        L += self.trade_log[:20]
        text = "\n".join(L)
        for i in range(0, len(text), 3000):
            self.log(text[i:i + 3000])
