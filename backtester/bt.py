"""
Honest backtester for well-known trading strategies, set up for a South African
retail investor trading JSE-listed ETFs (Satrix 40 / Satrix S&P 500) on
EasyEquities inside a tax-free savings account (TFSA).

What makes it "honest":
  * Every strategy is tuned ONLY on 2010-2019 (in-sample). The 2020-2026 data
    (out-of-sample) is never used to pick anything; it is only used to grade.
  * Trades happen the day AFTER the signal (you see the signal after the close
    and place the order next day). An optimistic same-day version is reported
    separately so you can see how much that assumption matters.
  * Every buy and every sell pays costs (brokerage + VAT + exchange fees +
    spread/slippage). Money not invested earns a money-market rate.
  * Because ~60 variants are tried, the best in-sample result is corrected for
    luck with the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
  * A walk-forward test re-picks the "best" strategy every year using only the
    past five years, exactly as a real person following backtests would.

Run:  python3 bt.py            (writes results/*.csv and results/summary.md)
Only needs pandas and numpy.
"""
from __future__ import annotations

import math
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")

# --------------------------------------------------------------------------
# Costs (per side, i.e. charged on every buy AND every sell)
#   EasyEquities brokerage 0.25% + 15% VAT = 0.2875%, plus STRATE/levies
#   (~0.01%) and half the bid-ask spread / slippage on STX40 (~0.10%).
# --------------------------------------------------------------------------
COST_LOW, COST_BASE, COST_HIGH = 0.002, 0.004, 0.006
SATRIX500_TER = 0.0035  # yearly fee of the S&P 500 feeder ETF (not in index data)

IS_START, IS_END = "2010-01-01", "2019-12-31"
OOS_START, OOS_END = "2020-01-01", "2026-12-31"

# Approximate SA money-market yield (net of fees) by calendar year, used for
# money that is not invested. Based on SARB repo-rate history; approximate.
SA_CASH = {2010: .065, 2011: .055, 2012: .053, 2013: .050, 2014: .057,
           2015: .061, 2016: .072, 2017: .072, 2018: .069, 2019: .069,
           2020: .045, 2021: .038, 2022: .055, 2023: .080, 2024: .081,
           2025: .072, 2026: .068}

TRADING_DAYS = 252


# ==========================================================================
# Data
# ==========================================================================
def load_jse_daily() -> pd.DataFrame:
    r = pd.read_csv(os.path.join(DATA, "stx40_daily_total_return.csv"),
                    index_col=0, parse_dates=True)["STX40_Return"]
    r = r.dropna()
    r = r[r.index.dayofweek < 5]
    df = pd.DataFrame({"ret": r})
    df["price"] = (1 + df.ret).cumprod() * 100  # total-return index
    df["cash"] = [(1 + SA_CASH[d.year]) ** (1 / TRADING_DAYS) - 1 for d in df.index]
    return df


def load_monthly_multi() -> pd.DataFrame:
    """Monthly returns in RAND: Top 40 (STX40), S&P 500 (Satrix 500 proxy), cash."""
    d = load_jse_daily()
    top40 = (1 + d.ret).resample("ME").prod() - 1

    sp = pd.read_csv(os.path.join(DATA, "sp500_monthly_shiller.csv"),
                     index_col=0, parse_dates=True)
    sp = sp[sp.index >= "2008-01-01"]
    div = sp["Dividend"].replace(0, np.nan).ffill()
    sp_tr_usd = sp["SP500"].pct_change() + (div / 12) / sp["SP500"].shift()
    sp_tr_usd.index = sp_tr_usd.index + pd.offsets.MonthEnd(0)

    fx1 = pd.read_csv(os.path.join(DATA, "usdzar_daily_1990.csv"), index_col=0, parse_dates=True)["value"]
    fx2 = pd.read_csv(os.path.join(DATA, "usdzar_daily_2016.csv"), index_col=0, parse_dates=True)["rate"]
    fx = pd.concat([fx1, fx2[fx2.index > fx1.index.max()]]).sort_index()
    fx_m = fx.resample("ME").mean()  # Shiller prices are monthly averages, so match
    sp_zar = (1 + sp_tr_usd) * (1 + fx_m.pct_change()) - 1 - SATRIX500_TER / 12

    cash = pd.Series({m: (1 + SA_CASH.get(m.year, .068)) ** (1 / 12) - 1 for m in top40.index})
    df = pd.concat([top40.rename("top40"), sp_zar.rename("sp500"), cash.rename("cash")], axis=1, sort=True)
    return df.dropna()


def load_us_long() -> pd.DataFrame:
    sp = pd.read_csv(os.path.join(DATA, "sp500_monthly_shiller.csv"), index_col=0, parse_dates=True)
    div = sp["Dividend"].replace(0, np.nan).ffill()
    rate = sp["Long Interest Rate"].replace(0, np.nan).ffill() / 100
    df = pd.DataFrame(index=sp.index)
    df["price"] = sp["SP500"]
    df["ret"] = sp["SP500"].pct_change() + (div / 12) / sp["SP500"].shift()
    # Shiller has no T-bill series: approximate cash as 70% of the 10y yield.
    df["cash"] = (1 + 0.7 * rate) ** (1 / 12) - 1
    return df.dropna()


# ==========================================================================
# Indicators
# ==========================================================================
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    ch = s.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def hold_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """1 from an entry signal until an exit signal (stateful, no look-ahead)."""
    out = np.zeros(len(entry))
    pos = 0
    e, x = entry.values, exit_.values
    for i in range(len(e)):
        if pos == 0 and e[i]:
            pos = 1
        elif pos == 1 and x[i]:
            pos = 0
        out[i] = pos
    return pd.Series(out, index=entry.index)


# ==========================================================================
# Strategies: each returns target exposure (0..1) using data up to that close
# ==========================================================================
def s_buy_hold(df):
    return pd.Series(1.0, index=df.index)


def s_sma(df, n):
    return (df.price > sma(df.price, n)).astype(float)


def s_golden_cross(df, fast, slow):
    return (sma(df.price, fast) > sma(df.price, slow)).astype(float)


def s_macd(df, f, s, sig):
    m = ema(df.price, f) - ema(df.price, s)
    return (m > ema(m, sig)).astype(float)


def s_rsi_reversion(df, n, lo):
    # Connors-style: buy oversold dips in an uptrend, sell on bounce above 5-day avg.
    p = df.price
    entry = (rsi(p, n) < lo) & (p > sma(p, 200))
    exit_ = p > sma(p, 5)
    return hold_state(entry.fillna(False), exit_.fillna(False))


def s_bollinger(df, n, k):
    p = df.price
    mid, sd = sma(p, n), p.rolling(n).std()
    return hold_state((p < mid - k * sd).fillna(False), (p > mid).fillna(False))


def s_donchian(df, n_in, n_out):
    p = df.price
    entry = p >= p.rolling(n_in).max()
    exit_ = p <= p.rolling(n_out).min()
    return hold_state(entry.fillna(False), exit_.fillna(False))


def s_tsmom(df, look):
    """Time-series momentum: in if the past `look`-day return beats cash; checked monthly."""
    past = df.price / df.price.shift(look) - 1
    cash_past = (1 + df.cash).rolling(look).apply(np.prod, raw=True) - 1
    sig = (past > cash_past).astype(float)
    month_end = df.index.to_series().dt.month != df.index.to_series().shift(-1).dt.month
    return sig.where(month_end).ffill().fillna(0)


def s_sell_in_may(df):
    return pd.Series(np.where(df.index.month.isin([5, 6, 7, 8, 9, 10]), 0.0, 1.0), index=df.index).shift(-1).ffill()


def s_buy_dip(df, drop, days):
    big_drop = df.ret < -drop
    out = big_drop.astype(float).rolling(days, min_periods=1).max()
    return out


def strategy_grid():
    g = [("Buy and hold", s_buy_hold, {})]
    for n in [50, 100, 150, 200, 250]:
        g.append((f"Moving average {n}d", s_sma, dict(n=n)))
    for f, s in [(20, 100), (50, 200), (20, 200), (50, 150)]:
        g.append((f"Golden cross {f}/{s}", s_golden_cross, dict(fast=f, slow=s)))
    for f, s, sig in [(8, 17, 9), (12, 26, 9), (19, 39, 9), (24, 52, 18)]:
        g.append((f"MACD {f}/{s}/{sig}", s_macd, dict(f=f, s=s, sig=sig)))
    for n in [2, 3, 5]:
        for lo in [5, 10, 20, 30]:
            g.append((f"RSI({n}) dip <{lo}", s_rsi_reversion, dict(n=n, lo=lo)))
    for n in [10, 20, 50]:
        for k in [1.5, 2.0, 2.5]:
            g.append((f"Bollinger {n}d {k}sd", s_bollinger, dict(n=n, k=k)))
    for a, b in [(20, 10), (55, 20), (100, 50), (200, 100)]:
        g.append((f"Breakout {a}/{b}", s_donchian, dict(n_in=a, n_out=b)))
    for look in [21, 63, 126, 189, 252]:
        g.append((f"Momentum {look}d", s_tsmom, dict(look=look)))
    g.append(("Sell in May", s_sell_in_may, {}))
    for drop in [0.015, 0.02, 0.03]:
        for days in [3, 5, 10, 20]:
            g.append((f"Buy dip >{drop:.1%} hold {days}d", s_buy_dip, dict(drop=drop, days=days)))
    return g


def family(name):
    return name.split(" ")[0] if not name.startswith("Buy and") else "Buy and hold"


# ==========================================================================
# Simulation and statistics
# ==========================================================================
def simulate(df, target, cost, lag=1):
    """Daily (or monthly) strategy returns.
    lag=1: signal at close t, trade at close t+1, earn returns from t+2 (realistic).
    lag=0: trade at the same close as the signal (optimistic)."""
    pos = target.shift(1 + lag).fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    r = pos * df.ret + (1 - pos) * df.cash - turnover * cost
    return r, pos, turnover


def stats(r, cash, pos, turnover, periods=TRADING_DAYS):
    n = len(r)
    if n < 20:
        return {}
    years = n / periods
    growth = (1 + r).prod()
    cagr = growth ** (1 / years) - 1
    ex = r - cash
    sharpe = ex.mean() / ex.std() * math.sqrt(periods) if ex.std() > 0 else 0.0
    curve = (1 + r).cumprod()
    mdd = (curve / curve.cummax() - 1).min()
    return dict(cagr=cagr, vol=r.std() * math.sqrt(periods), sharpe=sharpe, max_dd=mdd,
                r10k=10000 * growth, trades_per_yr=(turnover > 0).sum() / years,
                time_in=pos.mean(), years=years, skew=float(ex.skew()), kurt=float(ex.kurt() + 3),
                n=n)


def deflated_sharpe(sr_annual, n_obs, n_trials, sr_trials_annual, skew, kurt, periods=TRADING_DAYS):
    """Probability the best Sharpe is real after trying n_trials variants."""
    nd = NormalDist()
    sr = sr_annual / math.sqrt(periods)
    var_trials = np.var(np.array(sr_trials_annual) / math.sqrt(periods), ddof=1)
    emc = 0.5772156649
    sr0 = math.sqrt(var_trials) * ((1 - emc) * nd.inv_cdf(1 - 1 / n_trials)
                                   + emc * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return nd.cdf(z), sr0 * math.sqrt(periods)


def excess_tstat(r_a, r_b, periods=TRADING_DAYS):
    d = (r_a - r_b).dropna()
    ann = d.mean() * periods
    t = d.mean() / d.std() * math.sqrt(len(d)) if d.std() > 0 else 0
    return ann, t


def window(x, a, b):
    return x[(x.index >= a) & (x.index <= b)]


# ==========================================================================
# Part A: JSE Top 40 daily strategies
# ==========================================================================
def run_jse():
    df = load_jse_daily()
    rows, rets = [], {}
    for name, fn, kw in strategy_grid():
        tgt = fn(df, **kw)
        for cost_label, cost in [("low", COST_LOW), ("base", COST_BASE), ("high", COST_HIGH)]:
            for lag in [1, 0]:
                if lag == 0 and cost_label != "base":
                    continue
                r, pos, to = simulate(df, tgt, cost, lag)
                rets[(name, cost_label, lag)] = r
                for period, (a, b) in {"in_sample": (IS_START, IS_END),
                                       "out_of_sample": (OOS_START, OOS_END),
                                       "full": (IS_START, OOS_END)}.items():
                    st = stats(window(r, a, b), window(df.cash, a, b), window(pos, a, b), window(to, a, b))
                    rows.append(dict(strategy=name, family=family(name), cost=cost_label,
                                     execution="next day" if lag else "same close",
                                     period=period, **st))
    res = pd.DataFrame(rows)
    return df, res, rets


def walk_forward(df, rets, cost="base", lag=1, train_years=5, first_year=2015):
    names = [n for n, _, _ in strategy_grid()]
    chained, picks = [], []
    last_year = df.index.max().year
    for y in range(first_year, last_year + 1):
        a, b = f"{y - train_years}-01-01", f"{y - 1}-12-31"
        best, best_sr = None, -9
        for n in names:
            r = window(rets[(n, cost, lag)], a, b)
            c = window(df.cash, a, b)
            ex = r - c
            sr = ex.mean() / ex.std() * math.sqrt(TRADING_DAYS) if ex.std() > 0 else 0
            if sr > best_sr:
                best, best_sr = n, sr
        picks.append(dict(year=y, picked=best, trailing_sharpe=best_sr))
        chained.append(window(rets[(best, cost, lag)], f"{y}-01-01", f"{y}-12-31"))
    wf = pd.concat(chained)
    return wf, pd.DataFrame(picks)


# ==========================================================================
# Part B: monthly rand strategies (Top 40 vs S&P 500 vs cash)
# ==========================================================================
def run_monthly_multi():
    m = load_monthly_multi()
    rows, rets = [], {}

    def sim_weights(w, cost):
        w = w.shift(1).fillna(0)  # decided at month end, earns next month
        r = (w * m[["top40", "sp500", "cash"]]).sum(axis=1)
        to = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
        # the move to/from cash costs one side; equity to equity costs two
        return r - to * cost, w, to

    configs = {}
    configs["Top 40 buy and hold"] = pd.DataFrame({"top40": 1, "sp500": 0, "cash": 0}, index=m.index, dtype=float)
    configs["S&P 500 (rand) buy and hold"] = pd.DataFrame({"top40": 0, "sp500": 1, "cash": 0}, index=m.index, dtype=float)
    half = pd.DataFrame({"top40": .5, "sp500": .5, "cash": 0}, index=m.index, dtype=float)
    configs["50/50 Top 40 + S&P 500"] = half  # approximation: monthly rebalanced

    for L in [3, 6, 9, 12]:
        past = (1 + m).rolling(L).apply(np.prod, raw=True) - 1
        w = pd.DataFrame(0.0, index=m.index, columns=["top40", "sp500", "cash"])
        best_eq = np.where(past.top40 > past.sp500, "top40", "sp500")
        best_ret = np.maximum(past.top40, past.sp500).values
        for i, idx in enumerate(m.index):
            if np.isnan(best_ret[i]):
                continue
            w.loc[idx, best_eq[i] if best_ret[i] > past.cash.iloc[i] else "cash"] = 1.0
        configs[f"Dual momentum {L}m"] = w
        # absolute-momentum only on 50/50 (each half goes to cash if below cash)
        w2 = pd.DataFrame(0.0, index=m.index, columns=["top40", "sp500", "cash"])
        for a in ["top40", "sp500"]:
            on = (past[a] > past.cash).astype(float) * 0.5
            w2[a] = on
            w2["cash"] += 0.5 - on
        w2[past.isna().any(axis=1)] = 0
        configs[f"50/50 with trend filter {L}m"] = w2

    start = m.index[12]  # common start after the longest look-back
    for name, w in configs.items():
        for cost_label, cost in [("low", COST_LOW), ("base", COST_BASE), ("high", COST_HIGH)]:
            r, pos, to = sim_weights(w, cost)
            r, pos, to = r[r.index >= start], pos[pos.index >= start], to[to.index >= start]
            rets[(name, cost_label)] = r
            invested = 1 - pos["cash"]
            for period, (a, b) in {"in_sample": (IS_START, IS_END), "out_of_sample": (OOS_START, OOS_END),
                                   "full": (IS_START, OOS_END)}.items():
                st = stats(window(r, a, b), window(m.cash, a, b), window(invested, a, b),
                           window(to, a, b), periods=12)
                rows.append(dict(strategy=name, cost=cost_label, period=period, **st))
    return m, pd.DataFrame(rows), rets


# ==========================================================================
# Part C: 150 years of US data, monthly rules (robustness check)
# ==========================================================================
def run_us_long():
    df = load_us_long()
    cost = 0.002  # cheap US ETF trading, per side
    rules = {
        "Buy and hold": pd.Series(1.0, index=df.index),
        "10-month moving average": (df.price > sma(df.price, 10)).astype(float),
        "12-month momentum": (df.price / df.price.shift(12) - 1 > 0).astype(float),
        "Sell in May": pd.Series(np.where(df.index.month.isin([5, 6, 7, 8, 9, 10]), 0.0, 1.0), index=df.index).shift(-1).ffill(),
    }
    periods = {"1881-1949": ("1881-01-01", "1949-12-31"), "1950-1989": ("1950-01-01", "1989-12-31"),
               "1990-2026": ("1990-01-01", "2026-12-31")}
    rows = []
    for name, tgt in rules.items():
        for lag, lbl in [(0, "same month-end"), (1, "one month late")]:
            r, pos, to = simulate(df, tgt, cost, lag)
            for p, (a, b) in periods.items():
                st = stats(window(r, a, b), window(df.cash, a, b), window(pos, a, b), window(to, a, b), periods=12)
                rows.append(dict(strategy=name, execution=lbl, period=p, **st))
    return pd.DataFrame(rows)


# ==========================================================================
# Report
# ==========================================================================
def pct(x):
    return f"{x * 100:.1f}%"


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    P = lines.append

    # ---------- Part A
    df, res, rets = run_jse()
    res.to_csv(os.path.join(OUT, "jse_daily_all_results.csv"), index=False)
    base = res[(res.cost == "base") & (res.execution == "next day")]
    isd = base[base.period == "in_sample"].set_index("strategy")
    oos = base[base.period == "out_of_sample"].set_index("strategy")
    n_trials = len(isd)

    bh_is, bh_oos = isd.loc["Buy and hold"], oos.loc["Buy and hold"]
    P("# Backtest summary (generated by bt.py)\n")
    P(f"Data: Satrix 40 daily total return {df.index.min().date()} to {df.index.max().date()}. "
      f"Training (in-sample) {IS_START[:4]}-{IS_END[:4]}, test (out-of-sample) {OOS_START[:4]}-{df.index.max().year}. "
      f"Costs {COST_BASE:.2%} per buy or sell. Trades placed the day after the signal.\n")
    P(f"Strategy variants tried: {n_trials}\n")

    P("## A1. Best variant of each strategy family, picked on 2010-2019, graded on 2020-2026\n")
    P("| Strategy (best setting in training) | Training yearly return | Training Sharpe | Test yearly return | Test Sharpe | Test worst fall | R10,000 became (test) | Trades a year |")
    P("|---|---|---|---|---|---|---|---|")
    picks = isd.groupby("family")["sharpe"].idxmax()
    order = isd.loc[picks.values].sort_values("sharpe", ascending=False).index
    for s in order:
        a, b = isd.loc[s], oos.loc[s]
        P(f"| {s} | {pct(a.cagr)} | {a.sharpe:.2f} | {pct(b.cagr)} | {b.sharpe:.2f} | {pct(b.max_dd)} | R{b.r10k:,.0f} | {b.trades_per_yr:.1f} |")
    P("")

    best_is = isd["sharpe"].idxmax()
    b_is = isd.loc[best_is]
    psr, sr0 = deflated_sharpe(b_is.sharpe, int(b_is.n), n_trials, isd.sharpe.values, b_is["skew"], b_is["kurt"])
    P(f"**Overall winner in training:** {best_is} (Sharpe {b_is.sharpe:.2f} vs buy-and-hold {bh_is.sharpe:.2f}). "
      f"With {n_trials} variants tried, luck alone would produce a best Sharpe of about {sr0:.2f}. "
      f"Deflated Sharpe: probability the edge is real = {psr:.0%}.\n")
    b_oos = oos.loc[best_is]
    P(f"In the test years it returned {pct(b_oos.cagr)} a year (Sharpe {b_oos.sharpe:.2f}) vs buy-and-hold "
      f"{pct(bh_oos.cagr)} (Sharpe {bh_oos.sharpe:.2f}).\n")

    # rank stability
    corr = pd.concat([isd.sharpe, oos.sharpe], axis=1, keys=["is", "oos"]).corr(method="spearman").iloc[0, 1]
    beat = (oos.sharpe > bh_oos.sharpe).sum() - 0
    P(f"Rank correlation between training and test Sharpe across all {n_trials} variants: {corr:.2f} "
      f"(1 = training predicts test perfectly, 0 = no relation). "
      f"Variants that beat buy-and-hold's Sharpe in the test years: {beat} of {n_trials - 1}. "
      f"Variants that beat its return: {(oos.cagr > bh_oos.cagr).sum()} of {n_trials - 1}.\n")

    # cost / execution sensitivity for family winners
    P("## A2. How much costs and execution timing matter (test years, yearly return)\n")
    P("| Strategy | Same-close, base cost | Next day, low cost 0.2% | Next day, base 0.4% | Next day, high 0.6% |")
    P("|---|---|---|---|---|")
    o = res[res.period == "out_of_sample"].set_index(["strategy", "cost", "execution"])
    for s in order:
        P(f"| {s} | {pct(o.loc[(s, 'base', 'same close')].cagr)} | {pct(o.loc[(s, 'low', 'next day')].cagr)} | "
          f"{pct(o.loc[(s, 'base', 'next day')].cagr)} | {pct(o.loc[(s, 'high', 'next day')].cagr)} |")
    P("")

    # walk forward
    wf, wf_picks = walk_forward(df, rets)
    wf_picks.to_csv(os.path.join(OUT, "walk_forward_picks.csv"), index=False)
    bh_r = rets[("Buy and hold", "base", 1)]
    a0 = wf.index.min()
    bh_w = bh_r[bh_r.index >= a0]
    wst = stats(wf, df.cash.loc[wf.index], pd.Series(1, index=wf.index), pd.Series(0, index=wf.index))
    bst = stats(bh_w, df.cash.loc[bh_w.index], pd.Series(1, index=bh_w.index), pd.Series(0, index=bh_w.index))
    ann, t = excess_tstat(wf, bh_w)
    P("## A3. Walk-forward: re-pick the best strategy every January using the past 5 years\n")
    P("| Year | Strategy picked |")
    P("|---|---|")
    for _, rr in wf_picks.iterrows():
        P(f"| {rr.year} | {rr.picked} |")
    P("")
    P(f"Result {a0.year}-{wf.index.max().year}: {pct(wst['cagr'])} a year, worst fall {pct(wst['max_dd'])}, "
      f"R10,000 became R{wst['r10k']:,.0f}. Buy-and-hold: {pct(bst['cagr'])} a year, worst fall {pct(bst['max_dd'])}, "
      f"R10,000 became R{bst['r10k']:,.0f}. Difference {pct(ann)} a year (t-stat {t:.2f}; above 2 would suggest it is not luck).\n")

    # ---------- Part B
    m, mres, mrets = run_monthly_multi()
    mres.to_csv(os.path.join(OUT, "monthly_rand_results.csv"), index=False)
    mb = mres[mres.cost == "base"]
    P("## B. Monthly rules in rand: Top 40 vs S&P 500 vs cash\n")
    P("| Strategy | Training yearly return | Training Sharpe | Test yearly return | Test Sharpe | Test worst fall | R10,000 became (test) | Switches a year |")
    P("|---|---|---|---|---|---|---|---|")
    mi = mb[mb.period == "in_sample"].set_index("strategy")
    mo = mb[mb.period == "out_of_sample"].set_index("strategy")
    for s in mi.sort_values("sharpe", ascending=False).index:
        a, b = mi.loc[s], mo.loc[s]
        P(f"| {s} | {pct(a.cagr)} | {a.sharpe:.2f} | {pct(b.cagr)} | {b.sharpe:.2f} | {pct(b.max_dd)} | R{b.r10k:,.0f} | {b.trades_per_yr:.1f} |")
    P("")

    # ---------- Part C
    us = run_us_long()
    us.to_csv(os.path.join(OUT, "us_150_years_results.csv"), index=False)
    P("## C. 145 years of US data (monthly averages; flatters trend rules)\n")
    P("| Strategy | Execution | Period | Yearly return | Sharpe | Worst fall | Time invested |")
    P("|---|---|---|---|---|---|---|")
    for _, rr in us.iterrows():
        P(f"| {rr.strategy} | {rr.execution} | {rr.period} | {pct(rr.cagr)} | {rr.sharpe:.2f} | {pct(rr.max_dd)} | {pct(rr.time_in)} |")
    P("")

    # ---------- Latest signal for the one candidate worth paper trading
    tgt = s_bollinger(df, 50, 1.5)
    p = df.price
    mid, sd = sma(p, 50).iloc[-1], p.rolling(50).std().iloc[-1]
    P("## D. Paper-trading signal (Bollinger 50-day, 1.5 sd) on the last day of data\n")
    P(f"Date {df.index[-1].date()}: signal says {'HOLD Satrix 40' if tgt.iloc[-1] else 'stay in cash'}. "
      f"Buy level (index points of the total-return series): {mid - 1.5 * sd:.1f}; sell level: {mid:.1f}; "
      f"last value: {p.iloc[-1]:.1f}. Distance to buy level: {(mid - 1.5 * sd) / p.iloc[-1] - 1:.1%}.\n")

    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
