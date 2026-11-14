"""
Round 2: wider search, same honest rules as bt.py.

Part E. Multi-asset monthly strategies in RAND, 1991-2026, using assets a South
         African can buy as JSE ETFs: US shares (Satrix S&P 500), gold (NewGold),
         US government bonds (global bond ETF proxy) and SA cash.
         Train 1991-2009, test 2010-2026.
Part F. Bitcoin in rand (as bought on VALR or Luno), daily trend rules.
         Train 2014-2020, test 2021-2026.
Part G. What return is realistic per R10,000, per day / month / year.

Run: python3 bt2.py   (writes results/round2_summary.md and CSVs)
"""
import math
import os

import numpy as np
import pandas as pd

import bt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = bt.DATA, bt.OUT

# Approximate SA money-market yield 1990-2009 (SARB repo/T-bill history), extends bt.SA_CASH
SA_CASH_OLD = {1990: .175, 1991: .165, 1992: .14, 1993: .115, 1994: .105, 1995: .135, 1996: .145,
               1997: .15, 1998: .16, 1999: .125, 2000: .10, 2001: .095, 2002: .11, 2003: .10,
               2004: .073, 2005: .068, 2006: .072, 2007: .09, 2008: .11, 2009: .075}
SA_CASH = {**SA_CASH_OLD, **bt.SA_CASH}
ETF_TER = {"us_shares": .0035, "gold": .004, "us_bonds": .003}
COST = bt.COST_BASE
TRAIN = ("1991-01-01", "2009-12-31")
TEST = ("2010-01-01", "2026-12-31")


def usdzar():
    fx1 = pd.read_csv(os.path.join(DATA, "usdzar_daily_1990.csv"), index_col=0, parse_dates=True)["value"]
    fx2 = pd.read_csv(os.path.join(DATA, "usdzar_daily_2016.csv"), index_col=0, parse_dates=True)["rate"]
    return pd.concat([fx1, fx2[fx2.index > fx1.index.max()]]).sort_index().dropna()


def monthly_assets():
    fx_m = usdzar().resample("ME").mean()
    fx_r = fx_m.pct_change()

    sp = pd.read_csv(os.path.join(DATA, "sp500_monthly_shiller.csv"), index_col=0, parse_dates=True)
    div = sp["Dividend"].replace(0, np.nan).ffill()
    us = sp["SP500"].pct_change() + (div / 12) / sp["SP500"].shift()
    us.index = us.index + pd.offsets.MonthEnd(0)

    g = pd.read_csv(os.path.join(DATA, "gold_monthly_usd.csv"))
    g.index = pd.to_datetime(g.Date) + pd.offsets.MonthEnd(0)
    gold = g.Price.pct_change()

    y = pd.read_csv(os.path.join(DATA, "us10y_yield_monthly.csv"), index_col=0, parse_dates=True)["Rate"] / 100
    y.index = y.index + pd.offsets.MonthEnd(0)
    dur = 7.5  # approx. modified duration of a 10-year bond
    bonds = y.shift() / 12 - dur * y.diff() + 0.5 * dur ** 2 * y.diff() ** 2

    df = pd.concat({"us_shares": us, "gold": gold, "us_bonds": bonds}, axis=1, sort=True)
    df = df.apply(lambda c: (1 + c) * (1 + fx_r) - 1 - ETF_TER[c.name] / 12)
    df["cash"] = [(1 + SA_CASH.get(d.year, .068)) ** (1 / 12) - 1 for d in df.index]
    return df[(df.index >= "1990-02-01")].dropna()


RISKY = ["us_shares", "gold", "us_bonds"]


def strategies(m):
    """Return dict name -> monthly target weights (decided at month end, using data to that month end)."""
    idx = m.index
    W = {}

    def const(w):
        return pd.DataFrame([w] * len(idx), index=idx)[RISKY + ["cash"]].astype(float)

    W["US shares only (buy and hold)"] = const(dict(us_shares=1, gold=0, us_bonds=0, cash=0))
    W["Gold only (buy and hold)"] = const(dict(us_shares=0, gold=1, us_bonds=0, cash=0))
    W["60/40 shares and bonds"] = const(dict(us_shares=.6, gold=0, us_bonds=.4, cash=0))
    W["Equal thirds: shares, gold, bonds"] = const(dict(us_shares=1 / 3, gold=1 / 3, us_bonds=1 / 3, cash=0))
    W["Permanent portfolio (quarters incl. cash)"] = const(dict(us_shares=.25, gold=.25, us_bonds=.25, cash=.25))

    level = (1 + m[RISKY]).cumprod()
    for n in [6, 10, 12]:
        on = (level > level.rolling(n).mean()).astype(float) / 3
        w = on.copy()
        w["cash"] = 1 - on.sum(axis=1)
        w[level.rolling(n).mean().isna().any(axis=1)] = np.nan
        W[f"Trend-filtered thirds ({n}-month average)"] = w

    for L in [3, 6, 12]:
        past = (1 + m).rolling(L).apply(np.prod, raw=True) - 1
        for top in [1, 2]:
            w = pd.DataFrame(0.0, index=idx, columns=RISKY + ["cash"])
            for d in idx:
                if past.loc[d].isna().any():
                    w.loc[d] = np.nan
                    continue
                ranked = past.loc[d, RISKY].sort_values(ascending=False).index[:top]
                for a in ranked:
                    if past.loc[d, a] > past.loc[d, "cash"]:
                        w.loc[d, a] += 1 / top
                    else:
                        w.loc[d, "cash"] += 1 / top
            W[f"Momentum: best {top} of 3 ({L}-month)"] = w

    vol = m[RISKY].rolling(12).std()
    iv = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
    iv["cash"] = 0.0
    iv[vol.isna().any(axis=1)] = np.nan
    W["Risk parity (inverse volatility)"] = iv
    return W


def simulate_weights(m, w, cost):
    w = w.shift(1)  # decided at month end, earns next month
    start = w.dropna().index.min()
    w, mm = w.loc[start:], m.loc[start:]
    drift_r = (w * mm[RISKY + ["cash"]]).sum(axis=1)
    to = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    # constant-weight portfolios also pay a rebalancing cost (small, monthly)
    return drift_r - to * cost, w, to


def run_multi():
    m = monthly_assets()
    rows, rets = [], {}
    for name, w in strategies(m).items():
        r, w2, to = simulate_weights(m, w, COST)
        r = r[r.index >= "1991-01-01"]
        rets[name] = r
        inv = 1 - w2["cash"].reindex(r.index)
        for per, (a, b) in {"train": TRAIN, "test": TEST}.items():
            st = bt.stats(bt.window(r, a, b), bt.window(m.cash, a, b), bt.window(inv, a, b),
                          bt.window(to.reindex(r.index), a, b), periods=12)
            rows.append(dict(strategy=name, period=per, **st))
    return m, pd.DataFrame(rows), rets


# ---------------------------------------------------------------- Bitcoin
def btc_zar():
    b = pd.read_csv(os.path.join(DATA, "btc_daily_usd.csv"), parse_dates=["date"]).set_index("date").price_usd
    b = b[b.index >= "2013-06-01"]
    fx = usdzar().reindex(b.index).ffill()
    p = b * fx
    df = pd.DataFrame({"price": p})
    df["ret"] = p.pct_change()
    df["cash"] = [(1 + SA_CASH.get(d.year, .068)) ** (1 / 365) - 1 for d in df.index]
    return df.dropna()


def run_btc():
    df = btc_zar()
    rules = {"Bitcoin buy and hold": bt.s_buy_hold}
    grid = [("Bitcoin buy and hold", bt.s_buy_hold, {})]
    for n in [20, 50, 100, 200]:
        grid.append((f"Bitcoin above {n}-day average", bt.s_sma, dict(n=n)))
    for f, s in [(20, 100), (50, 200)]:
        grid.append((f"Bitcoin golden cross {f}/{s}", bt.s_golden_cross, dict(fast=f, slow=s)))
    for look in [30, 90, 180]:
        grid.append((f"Bitcoin momentum {look}-day", bt.s_tsmom, dict(look=look)))
    grid.append(("Bitcoin Bollinger dip-buy 50d", bt.s_bollinger, dict(n=50, k=1.5)))
    rows = []
    for name, fn, kw in grid:
        tgt = fn(df, **kw)
        for cost in [0.0025, 0.0075]:
            r, pos, to = bt.simulate(df, tgt, cost, lag=1)
            for per, (a, b) in {"train": ("2014-01-01", "2020-12-31"), "test": ("2021-01-01", "2026-12-31")}.items():
                st = bt.stats(bt.window(r, a, b), bt.window(df.cash, a, b), bt.window(pos, a, b),
                              bt.window(to, a, b), periods=365)
                rows.append(dict(strategy=name, cost=cost, period=per, **st))
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------- realistic returns
def realistic(jse):
    r = jse.ret
    up = (r > 0).mean()
    month = (1 + r).resample("ME").prod() - 1
    year = (1 + r).resample("YE").prod() - 1
    worst_12m = ((1 + r).rolling(252).apply(np.prod, raw=True) - 1).min()
    return dict(up_days=up, day_sd=r.std(), day_mean=r.mean(), up_months=(month > 0).mean(),
                up_years=(year > 0).mean(), worst_month=month.min(), worst_12m=worst_12m,
                cagr=(1 + r).prod() ** (252 / len(r)) - 1)


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    L = []
    P = L.append
    m, res, rets = run_multi()
    res.to_csv(os.path.join(OUT, "round2_multi_asset.csv"), index=False)
    tr = res[res.period == "train"].set_index("strategy")
    te = res[res.period == "test"].set_index("strategy")
    P("# Round 2 summary (generated by bt2.py)\n")
    P(f"## E. Multi-asset in rand, train {TRAIN[0][:4]}-{TRAIN[1][:4]}, test {TEST[0][:4]}-{m.index.max().year}\n")
    P("| Strategy | Train return a year | Train Sharpe | Test return a year | Test Sharpe | Test worst fall | R10,000 became (test) | Trades a year |")
    P("|---|---|---|---|---|---|---|---|")
    for s in tr.sort_values("sharpe", ascending=False).index:
        a, b = tr.loc[s], te.loc[s]
        P(f"| {s} | {pct(a.cagr)} | {a.sharpe:.2f} | {pct(b.cagr)} | {b.sharpe:.2f} | {pct(b.max_dd)} | R{b.r10k:,.0f} | {b.trades_per_yr:.1f} |")
    n = len(tr)
    best = tr.sharpe.idxmax()
    psr, sr0 = bt.deflated_sharpe(tr.loc[best].sharpe, int(tr.loc[best].n), n, tr.sharpe.values,
                                  tr.loc[best]["skew"], tr.loc[best]["kurt"], periods=12)
    corr = pd.concat([tr.sharpe, te.sharpe], axis=1).corr(method="spearman").iloc[0, 1]
    P(f"\nBest in training: {best}. Deflated Sharpe probability real: {psr:.0%} (luck benchmark Sharpe {sr0:.2f}). "
      f"Rank correlation train vs test: {corr:.2f}.\n")

    btc, bres = run_btc()
    bres.to_csv(os.path.join(OUT, "round2_bitcoin.csv"), index=False)
    P("## F. Bitcoin in rand, train 2014-2020, test 2021-2026, cost 0.25% per side (0.75% in brackets)\n")
    P("| Strategy | Train return a year | Test return a year | Test Sharpe | Test worst fall | Time invested (test) |")
    P("|---|---|---|---|---|---|")
    lo = bres[bres.cost == 0.0025]
    hi = bres[bres.cost == 0.0075]
    for s in lo.strategy.unique():
        a = lo[(lo.strategy == s) & (lo.period == "train")].iloc[0]
        b = lo[(lo.strategy == s) & (lo.period == "test")].iloc[0]
        bh = hi[(hi.strategy == s) & (hi.period == "test")].iloc[0]
        P(f"| {s} | {pct(a.cagr)} | {pct(b.cagr)} ({pct(bh.cagr)}) | {b.sharpe:.2f} | {pct(b.max_dd)} | {pct(b.time_in)} |")
    P(f"\nBitcoin data ends {btc.index.max().date()}.\n")

    jse = bt.load_jse_daily()
    rz = realistic(jse)
    P("## G. Top 40 facts, 2010-2026 (for what is realistic)\n")
    for k, v in rz.items():
        P(f"- {k}: {pct(v) if abs(v) < 5 else v}")
    with open(os.path.join(OUT, "round2_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
