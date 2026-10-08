# ETF Rotation Backtest

A rule-based ETF rotation strategy, backtested on QuantConnect (LEAN) from 1 March 2010 to 10 July 2026.

This is a research project, not a validated live strategy. The numbers below come from a full-sample backtest that was used while developing the rules, so they are likely flattering. See [Limitations](#limitations).

## What the strategy does

Once a week, the strategy looks at 12 ETFs and decides what to hold for the next week. It uses the previous close only, and orders go in as market-on-open for the next session.

**Universe:** SPY, QQQ, IWM, EFA, EEM, XLK, XLF, XLE, TLT, GLD, VNQ, HYG
**Cash proxy:** BIL (short-term Treasury bills), used for any weight that cannot be allocated.

The strategy switches between four modes. They are checked in this order, and the first one that applies wins. Two stress signals decide the mode: the VIX / VIX3M ratio (how nervous short-term volatility is compared with 3-month volatility) and the 20-day z-score of the VIX (how unusual today's VIX is compared with the last 20 days).

| Mode | When it applies | What it holds |
|---|---|---|
| Extreme | VIX/VIX3M above 1.20 and VIX z-score above 2 | 50% BIL, up to 50% in TLT and GLD (only if their momentum is positive) |
| Reversal | VIX/VIX3M above 1.05, or z-score above 1.5 with ratio above 1.0 | Up to 3 ETFs that are above their 200-day average, have a positive 3-month return, and had the biggest recent 5-day drop |
| Momentum | No stress and SPY above its 200-day average | Top 3 ETFs by momentum score, only if above their 100-day average |
| Bear | No stress and SPY below its 200-day average | Same as Momentum but only 50% invested, rest in BIL |

**Momentum score:** 40% of the 1-month return, 40% of the 3-month return and 20% of the 6-month return, divided by the 20-day annualized volatility.

**Position sizing:** inverse volatility weights (calmer ETFs get more), capped at 40% per ETF. Any weight that cannot be placed goes to BIL.

**Other rules:** at most 2 of SPY, QQQ and XLK at once. A holding is kept while it stays in the top 5, and is only swapped if the new candidate scores at least 10% and 0.02 points better. Weight changes under 2% are skipped.

**Costs:** Interactive Brokers fee model plus a small fixed slippage per ETF (1 to 3 basis points per side).

The code also simulates three benchmarks internally (buy and hold SPY, buy and hold QQQ, and a TQQQ trend strategy) and prints a comparison table in the QuantConnect logs.

## Results

| Metric | Result |
|---|---|
| Starting / ending equity | $100,000 / $376,259 |
| Total net return | 276.3% |
| CAGR | 8.43% |
| Maximum drawdown | 17.6% |
| Sharpe / Sortino | 0.458 / 0.474 |
| Probabilistic Sharpe ratio | 0.334% |
| Annual volatility | 9.3% |
| Total fees | $8,831.83 |
| Orders | 1,966 |
| Beta / alpha | 0.35 / 0.012 |
| Information ratio | -0.381 |

All figures are after modeled costs, taken from the QuantConnect statistics. The full breakdown with charts (equity curve, drawdown, exposure, turnover, rolling and calendar-year returns) is in the [PDF report](ETF-Rotation-Backtest-Report.pdf).

## Limitations

- **It did not beat SPY.** On the same $100,000 start, the QuantConnect SPY benchmark ended at roughly $913,748 against $376,259 for the strategy. It had lower volatility and a smaller drawdown, but the lower return is the honest headline.
- **No real out-of-sample test.** The thresholds were developed on this same period. The 2024 onward data has already been seen, so it is only a holdout check, not fresh evidence.
- **Low statistical confidence.** The probabilistic Sharpe ratio is 0.334%, so there is no proof of an edge.
- **One invalid order.** A BIL order on 24 August 2015 was rejected for insufficient buying power, so the real portfolio may differ slightly from the intended one.
- **Simulated costs only.** Fixed slippage and modeled fees cannot capture real spreads, liquidity or market impact.

The validation steps still to do (development vs holdout split, threshold sensitivity, ablations, higher cost stress tests and an execution audit) are listed in section 6 of the report.

## Run it yourself

1. Create a new Python algorithm project on QuantConnect.
2. Paste the contents of `main.py` into it.
3. Set `RUN` at the top of the file: `"full"` (2010 to latest), `"dev"` (2010 to 2023) or `"oos"` (2024 to latest).
4. Run the backtest. Results print in the Logs tab.

Note: the code has been checked locally, but run a QuantConnect smoke test before trusting any output.

## Files

- `main.py`: the full algorithm
- `ETF-Rotation-Backtest-Report.pdf`: the full report with charts

## Disclaimer

Historical simulation only. This is not investment advice and says nothing certain about future returns.

Author: Anay Sharma
