"""
KO / PEP pairs trading backtest - version 2.

Adds:
  1. Cointegration testing (Engle-Granger + ADF), full sample and split sample
  2. Risk management: a z-score stop-loss and a maximum holding period
  3. A side-by-side comparison of the original rules vs the risk-managed rules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from statsmodels.tsa.stattools import coint, adfuller

# ------------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------------
tickers = ["KO", "PEP"]
start_date = "2022-09-19"
end_date = date.today().isoformat()

print(f"Pulling {tickers} from {start_date} to {end_date}...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
data = data.dropna()
print(f"Trading days: {len(data)}  ({data.index[0].date()} to {data.index[-1].date()})\n")

log_ko = np.log(data["KO"])
log_pep = np.log(data["PEP"])
spread = log_ko - log_pep

# ------------------------------------------------------------------
# 2. IS THE SPREAD ACTUALLY MEAN-REVERTING? (cointegration testing)
# ------------------------------------------------------------------
# Correlation says "do they move together day to day".
# Cointegration says "is the GAP between them anchored, or does it wander
# off freely". Mean reversion needs the second one, and only the second
# one justifies betting that a stretched spread will come back.
#
# Both tests below share the same null hypothesis: "no mean reversion".
# A p-value below 0.05 means we can reject that and claim evidence of
# mean reversion. A high p-value means we cannot - the spread looks like
# it's free to wander, which is fatal for this strategy.

def cointegration_report(label, s_ko, s_pep):
    _, coint_p, _ = coint(s_ko, s_pep)
    spr = s_ko - s_pep
    adf_p = adfuller(spr.dropna())[1]
    verdict = "mean-reverting" if min(coint_p, adf_p) < 0.05 else "NOT mean-reverting"
    print(f"  {label:<22} Engle-Granger p={coint_p:.3f}   ADF p={adf_p:.3f}   -> {verdict}")

print("Cointegration tests (p < 0.05 = evidence of mean reversion):")
cointegration_report("Full sample", log_ko, log_pep)

# Split the sample in half to test whether the relationship changed
mid = len(data) // 2
cointegration_report("First half", log_ko.iloc[:mid], log_pep.iloc[:mid])
cointegration_report("Second half", log_ko.iloc[mid:], log_pep.iloc[mid:])

corr = log_ko.diff().corr(log_pep.diff())
print(f"\nFull-period daily return correlation: {corr:.3f}\n")

# ------------------------------------------------------------------
# 3. SIGNAL
# ------------------------------------------------------------------
window = 60
zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

# ------------------------------------------------------------------
# 4. BACKTEST ENGINE
# ------------------------------------------------------------------
def run_backtest(zscore, spread, entry=2.0, exit_th=0.0,
                 stop_z=None, max_hold=None):
    """
    Returns (positions series, list of trade records).

    stop_z   - exit if the z-score stretches this far AGAINST us
               (i.e. the spread kept diverging instead of reverting)
    max_hold - exit after this many trading days regardless

    After a stop-loss we block re-entry until the z-score calms back
    inside +/-1. Without that, the very next day would re-open the same
    losing trade and the stop would achieve nothing.
    """
    position = 0
    days_in_trade = 0
    blocked = False
    positions = []
    trades = []
    entry_date = None
    entry_idx = None

    spread_vals = spread.values
    dates = zscore.index

    for i, (d, z) in enumerate(zip(dates, zscore.values)):
        if np.isnan(z):
            positions.append(0)
            continue

        if position == 0:
            if blocked:
                if abs(z) < 1.0:      # spread has calmed - allow trading again
                    blocked = False
            else:
                if z > entry:
                    position, days_in_trade = -1, 0
                    entry_date, entry_idx = d, i
                elif z < -entry:
                    position, days_in_trade = 1, 0
                    entry_date, entry_idx = d, i
        else:
            days_in_trade += 1
            reason = None

            reverted = (position == 1 and z >= exit_th) or (position == -1 and z <= exit_th)
            stopped = stop_z is not None and (
                (position == 1 and z < -stop_z) or (position == -1 and z > stop_z))
            timed_out = max_hold is not None and days_in_trade >= max_hold

            if reverted:
                reason = "reverted"
            elif stopped:
                reason = "stop-loss"
            elif timed_out:
                reason = "time limit"

            if reason:
                pnl = position * (spread_vals[i] - spread_vals[entry_idx])
                trades.append({"entry": entry_date.date(), "exit": d.date(),
                               "dir": "long" if position == 1 else "short",
                               "days": days_in_trade, "pnl": pnl, "exit_reason": reason})
                position = 0
                if reason in ("stop-loss", "time limit"):
                    blocked = True

        positions.append(position)

    return pd.Series(positions, index=dates), trades


def summarise(name, positions, trades, spread):
    strat_ret = positions.shift(1) * spread.diff()
    cum = strat_ret.cumsum()
    total = cum.iloc[-1]
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() else np.nan
    wins = sum(1 for t in trades if t["pnl"] > 0)

    print(f"--- {name} ---")
    print(f"  Trades:            {len(trades)}")
    print(f"  Win rate:          {wins}/{len(trades)}" if trades else "  Win rate: n/a")
    print(f"  Days in position:  {(positions != 0).sum()} / {len(positions)}")
    print(f"  Avg days per trade:{np.mean([t['days'] for t in trades]):.0f}" if trades else "")
    print(f"  Cumulative PnL:    {total:.4f}  ({(np.exp(total)-1)*100:.1f}%)")
    print(f"  Sharpe:            {sharpe:.2f}")
    if trades:
        reasons = pd.Series([t["exit_reason"] for t in trades]).value_counts()
        print(f"  Exits by reason:   {dict(reasons)}")
    print()
    return cum


# Original rules: no risk management
pos_a, trades_a = run_backtest(zscore, spread)

# Risk-managed rules: stop out at |z| = 3, and never hold beyond 30 days
pos_b, trades_b = run_backtest(zscore, spread, stop_z=3.0, max_hold=30)

print("=" * 60)
cum_a = summarise("ORIGINAL (no stop-loss, no time limit)", pos_a, trades_a, spread)
cum_b = summarise("RISK-MANAGED (stop at |z|=3, max 30 days)", pos_b, trades_b, spread)
print("=" * 60)

print("\nPnL by calendar year (log-return units):")
yr = pd.DataFrame({
    "original": (pos_a.shift(1) * spread.diff()).groupby(spread.index.year).sum(),
    "risk-managed": (pos_b.shift(1) * spread.diff()).groupby(spread.index.year).sum(),
})
print(yr.round(4).to_string())

print("\nTrade log (risk-managed):")
print(pd.DataFrame(trades_b).to_string(index=False))

# ------------------------------------------------------------------
# 5. PLOT
# ------------------------------------------------------------------
plt.figure(figsize=(11, 5.5))
plt.plot(cum_a.index, cum_a, color="#2E5EAA", linewidth=2, label="Original rules")
plt.plot(cum_b.index, cum_b, color="#D1793A", linewidth=2, label="With stop-loss + time limit")
plt.axhline(0, color="#888888", linewidth=1, linestyle="--")
plt.title("KO / PEP Pairs Trade - Cumulative PnL")
plt.xlabel("Date")
plt.ylabel("Cumulative log return")
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pnl_comparison.png", dpi=120)
plt.show()

data["zscore"] = zscore
data["position_original"] = pos_a
data["position_riskmanaged"] = pos_b
data.to_csv("ko_pep_v2.csv")
print("\nSaved: ko_pep_v2.csv, pnl_comparison.png")
