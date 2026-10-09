# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

"""
ETF rotation, VERSION 2  (QuantConnect, Python)

Universe : SPY QQQ IWM EFA EEM XLF XLE VNQ TLT IEF GLD DBC     Cash: BIL     Signal only (never held): HYG
Core     : weekly (first session of the week). Decide on the close, market-on-open order next session.
           score = (0.4 x 21d return + 0.4 x 63d return + 0.2 x 126d return) / annualized 20d volatility
           eligible = price above its 100d average AND score above 0
           hold the top 3 eligible; keep an existing holding while it stays eligible and in the top 5
           empty slots go to BIL
           weights = inverse 20d volatility, max 40% each, excess redistributed, remainder to BIL
           skip weight changes under 2%; always execute full exits
Switches : USE_VOL_TARGET, USE_CRASH_BRAKE, USE_CREDIT_SIGNAL (all evaluated daily, see settings below)
Costs    : Interactive Brokers fee model + 2 basis points slippage per side
Report   : custom table in the Logs tab (CAGR, vol, Sharpe, max drawdown, worst year, longest time
           underwater, yearly turnover, trades) for the strategy, SPY buy and hold, and 60/40 SPY/IEF
           rebalanced monthly. Sharpe uses same-date BIL returns as the risk-free proxy.
"""

# ============================================================== settings

RUN = "dev"                  # "dev" = 2008-01 to 2023-12 | "oos" = 2024-01 to latest | "full" = 2008-01 to latest
CASH = 100_000

USE_VOL_TARGET = False       # A: scale exposure to min(1, VOL_TARGET / estimated portfolio volatility)
USE_CRASH_BRAKE = False      # B: cap exposure when SPY is far below its recent high
USE_CREDIT_SIGNAL = False    # C: cap exposure when HYG/IEF is below its 50 day average

VOL_TARGET = 0.10            # annualized portfolio volatility target
VT_N = 20                    # days of returns used to estimate portfolio volatility
EXP_TRIGGER = 0.10           # between weekly rebalances, trade only if exposure moves by more than this
CRASH_PCT = 0.05             # crash brake: SPY close more than 5% below its highest close of the last CRASH_N days
CRASH_N = 20
CRASH_CAP = 0.50
CREDIT_N = 50                # HYG/IEF ratio compared with its 50 day average
CREDIT_CAP = 0.50

RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLF", "XLE", "VNQ", "TLT", "IEF", "GLD", "DBC"]
SAFE = "BIL"
SIGNAL_ONLY = "HYG"

W1, W3, W6 = 0.40, 0.40, 0.20
L1, L3, L6 = 21, 63, 126
VOL_N = 20
TREND_N = 100                # price must be above this many day average
TOP_N = 3
KEEP_RANK = 5
MAX_W = 0.40
MIN_TRADE = 0.02
CASH_BUFFER = 0.995
SLIPPAGE = 0.0002            # 2 basis points per side
SIM_COST_BPS = 2             # same cost for the simulated benchmarks


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
    return (W1 * r1 + W3 * r3 + W6 * r6) / (v * np.sqrt(252))


def inv_vol(picks, closes, gross):
    """Inverse volatility weights, capped at MAX_W each, excess redistributed. Leftover goes to BIL."""
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


def select_core(closes, held):
    """Top TOP_N eligible assets. A holding is kept while it is still eligible and inside the top KEEP_RANK."""
    scores = {}
    for a in RISK:
        c = closes[a]
        s, m = mom_score(c), sma(c, TREND_N)
        if s is None or m is None or c[-1] <= m or s <= 0:
            continue
        scores[a] = s
    ranked = sorted(scores, key=scores.get, reverse=True)
    group = ranked[:KEEP_RANK]
    picks = [a for a in ranked if a in held and a in group][:TOP_N]
    for a in ranked:
        if len(picks) >= TOP_N:
            break
        if a not in picks:
            picks.append(a)
    return inv_vol(picks, closes, 1.0), picks, scores


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
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    mdd = float(((np.r_[1.0, eq] - peak) / peak).min())
    return {"cagr": cagr, "sharpe": excess.mean() / ex_sd * np.sqrt(252) if ex_sd > 0 else 0.0,
            "mdd": mdd, "vol": sd * np.sqrt(252), "years": yrs, "total": eq[-1] - 1}


def yearly(dated):
    out = {}
    for d, x in dated:
        out.setdefault(d.year, []).append(x)
    return {y: float(np.prod(1 + np.array(v)) - 1) for y, v in out.items()}


def underwater_days(dated):
    """Longest stretch (calendar days) from an equity peak until a new peak. Open stretch at the end counts."""
    eq, peak, peak_d, longest = 1.0, 1.0, dated[0][0], 0
    for d, x in dated:
        eq *= 1 + x
        if eq >= peak:
            longest = max(longest, (d - peak_d).days)
            peak, peak_d = eq, d
    return max(longest, (dated[-1][0] - peak_d).days)


class Sim:
    """Virtual benchmark portfolio: target set at a close, executed at the next open, costs per side."""

    def __init__(self, name):
        self.name = name
        self.h = {"CASH": 1.0}
        self.pending = None
        self.rets, self.prev = [], 1.0
        self.turnover, self.trades = 0.0, 0

    def value(self):
        return sum(self.h.values())

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
            cost += d * SIM_COST_BPS / 1e4
            self.turnover += d / v
        self.h = {a: x * (v - cost) / v for a, x in new.items()}


# ============================================================== algorithm

class SlippageInit(BrokerageModelSecurityInitializer):
    def __init__(self, model, seeder):
        super().__init__(model, seeder)

    def initialize(self, security: Security) -> None:
        super().initialize(security)
        if security.type == SecurityType.EQUITY:
            security.set_slippage_model(ConstantSlippageModel(SLIPPAGE))
            security.set_leverage(2.0)     # lets same-open sells fund buys without rejections


class EtfRotationV2(QCAlgorithm):

    def initialize(self) -> None:
        if RUN not in ("full", "dev", "oos"):
            raise ValueError("RUN must be full, dev, or oos")
        if RUN == "oos":
            self.set_start_date(2024, 1, 1)
        else:
            self.set_start_date(2008, 1, 1)
        if RUN == "dev":
            self.set_end_date(2023, 12, 31)
        self.set_cash(CASH)
        self.set_warm_up(timedelta(days=420))
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.set_security_initializer(SlippageInit(self.brokerage_model,
                                                   FuncSecuritySeeder(self.get_last_known_prices)))
        self.sym = {a: self.add_equity(a, Resolution.DAILY,
                                       data_normalization_mode=DataNormalizationMode.ADJUSTED).symbol
                    for a in RISK + [SAFE, SIGNAL_ONLY]}
        self.set_benchmark(self.sym["SPY"])

        self.closes = {a: deque(maxlen=300) for a in self.sym}
        self.ratio_hist = deque(maxlen=CREDIT_N)
        self.prev_closes = {}
        self.sims = {"BH_SPY": Sim("BH_SPY"), "60_40": Sim("60_40")}

        self.last_date = None
        self.last_week = None
        self.last_month = None
        self.started = False
        self.picks = []
        self.base_w = {}            # core target weights (risk assets only, before any exposure scaling)
        self.cur_exp = 1.0
        self.eq = []                # (date, equity)
        self.rf = {}                # same-date BIL close-to-close return
        self.exp_hist = []
        self.bind = {"vol": 0, "crash": 0, "credit": 0}   # days each cap was the lowest one in force
        self.rebalances = 0
        self.exp_trades = 0
        self.n_trades = 0
        self.traded = 0.0

    # ---------------------------------------------------------------- daily data
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
        if SIGNAL_ONLY in today and "IEF" in today and today["IEF"] > 0:
            self.ratio_hist.append(today[SIGNAL_ONLY] / today["IEF"])

        trading = not self.is_warming_up
        for name, s in self.sims.items():
            s.step(date, opens, today, self.prev_closes, trading and self.started)
        if trading and self.started and SAFE in today and self.prev_closes.get(SAFE, 0) > 0:
            self.rf[date] = today[SAFE] / self.prev_closes[SAFE] - 1.0
        self.prev_closes.update(today)
        if not trading:
            return

        if not self.started:
            self.sims["BH_SPY"].pending = {"SPY": 1.0}
        month = (date.year, date.month)
        if month != self.last_month:
            self.last_month = month
            self.sims["60_40"].pending = {"SPY": 0.6, "IEF": 0.4}
        self.eq.append((date, float(self.portfolio.total_portfolio_value)))

        if any(len(self.closes[a]) < L6 + 2 for a in RISK):
            return

        week = date.isocalendar()[:2]
        weekly = week != self.last_week or not self.started
        if weekly:
            self.last_week = week
            closes = {a: list(self.closes[a]) for a in RISK}
            self.base_w, self.picks, _ = select_core(closes, self.picks)

        e = self._exposure()
        if self.started:
            self.exp_hist.append(e)
        if weekly:
            self.rebalances += 1
            self._apply(e)
        elif abs(e - self.cur_exp) > EXP_TRIGGER:
            self.exp_trades += 1
            self._apply(e)
        self.started = True

    # ---------------------------------------------------------------- exposure (daily)
    def _port_vol(self):
        w = dict(self.base_w)
        w[SAFE] = max(0.0, 1.0 - sum(self.base_w.values()))
        total = None
        for a, x in w.items():
            c = np.asarray(self.closes[a], dtype=float)
            if x <= 0 or len(c) < VT_N + 1:
                continue
            c = c[-(VT_N + 1):]
            r = x * (c[1:] / c[:-1] - 1.0)
            total = r if total is None else total + r
        if total is None:
            return None
        return float(total.std(ddof=1) * np.sqrt(252))

    def _exposure(self):
        caps = {}
        if USE_VOL_TARGET:
            v = self._port_vol()
            caps["vol"] = min(1.0, VOL_TARGET / v) if v and v > 1e-9 else 1.0
        if USE_CRASH_BRAKE:
            c = list(self.closes["SPY"])[-CRASH_N:]
            caps["crash"] = CRASH_CAP if c[-1] < max(c) * (1.0 - CRASH_PCT) else 1.0
        if USE_CREDIT_SIGNAL and len(self.ratio_hist) >= CREDIT_N:
            r = list(self.ratio_hist)
            caps["credit"] = CREDIT_CAP if r[-1] < float(np.mean(r)) else 1.0
        e = min(list(caps.values()) + [1.0])
        if e < 0.9999 and self.started:
            self.bind[min(caps, key=caps.get)] += 1
        return e

    # ---------------------------------------------------------------- trading
    def _apply(self, e):
        target = {a: w * e for a, w in self.base_w.items()}
        target[SAFE] = max(0.0, 1.0 - sum(target.values()))
        self.cur_exp = e
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
                    self.market_on_open_order(s, q, tag=f"w={w:.2f}")

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

        common = set(d for d, _ in strat)
        for r, _, _ in rows.values():
            common &= {d for d, _ in r}
        missing = common - set(self.rf)
        if missing:
            self.log(f"BIL proxy missing on {len(missing)} report dates; report skipped.")
            return
        rows = {n: ([(d, x) for d, x in r if d in common], t, k) for n, (r, t, k) in rows.items()}

        L = [f"RUN={RUN} | VOL_TARGET={USE_VOL_TARGET} CRASH_BRAKE={USE_CRASH_BRAKE} CREDIT={USE_CREDIT_SIGNAL}"
             f" | params: vol {VOL_TARGET} trend {TREND_N} crash {CRASH_PCT}",
             f"period {min(common).date()} -> {max(common).date()}",
             "name        CAGR   Vol  Sharpe  MaxDD  WorstYr(year)  Underwater(days)  Turn/yr  Trades"]
        for n, (r, turn, tr) in rows.items():
            m = metrics([x for _, x in r], [self.rf[d] for d, _ in r])
            if not m:
                continue
            yr = yearly(r)
            wy = min(yr, key=yr.get)
            L.append(f"{n:10s} {m['cagr']*100:5.1f}% {m['vol']*100:4.1f}% {m['sharpe']:6.4f} {m['mdd']*100:6.1f}% "
                     f"{yr[wy]*100:7.1f}% ({wy}) {underwater_days(r):10d} {turn/m['years']*100:9.0f}% {tr:7d}")
        yrs = sorted({d.year for d, _ in strat})
        L.append("Yearly %    " + " ".join(f"{y % 100:5d}" for y in yrs))
        for n, (r, _, _) in rows.items():
            yr = yearly(r)
            L.append(f"{n:10s} " + " ".join(f"{yr.get(y, 0)*100:5.0f}" for y in yrs))
        avg_e = float(np.mean(self.exp_hist)) if self.exp_hist else 1.0
        L.append(f"Weekly rebalances {self.rebalances}; extra exposure trades {self.exp_trades}; "
                 f"avg intended exposure {avg_e*100:.0f}%")
        L.append(f"Days each cap was the lowest in force: vol {self.bind['vol']}, crash {self.bind['crash']}, "
                 f"credit {self.bind['credit']}")
        L.append("QC total fees USD " + f"{float(self.portfolio.total_fees):,.0f}")
        L.append("Turnover = gross buys + sells / average equity / year (no halving). Sharpe uses BIL as risk free.")
        text = "\n".join(L)
        for i in range(0, len(text), 3000):
            self.log(text[i:i + 3000])
