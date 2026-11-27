"""
Practice (paper) trading of "trend-filtered thirds", checked once a month. No real money.

Funds (JSE ETFs on EasyEquities): Satrix S&P 500 (STX500), NewGold (GLD),
1nvest Global Government Bond (ETFGGB). Money not in a fund sits in a money-market fund.

Rule: at each month end, each fund whose price is above its 10-month average gets
one third of the account; a fund below its average has its third in cash.
Trades are made at the close on the first trading day after the month end check.

History before Oct 2026 comes from the index data bt2.py uses (S&P 500, gold and
US 10-year bonds in rand); from then on it is chained with the funds' own month-end
closes in three_fund_prices.csv (dividend-adjusted where the source gives it).
A blank price means "not found": the gold index proxy is used for GLD then.

Monthly routine: add the last trading day's closes (cents) to three_fund_prices.csv,
then run  python3 paper/three_fund.py  -> three_fund_status.md
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bt  # noqa: E402
import bt2  # noqa: E402

FUNDS = {"us_shares": "STX500", "gold": "GLD", "us_bonds": "ETFGGB"}
START = pd.Timestamp("2026-11-13")
START_CASH = 10000.0
COST = bt.COST_BASE


def gold_proxy_returns():
    g = pd.read_csv(os.path.join(bt.DATA, "gold_monthly_usd.csv"))
    g.index = pd.to_datetime(g.Date) + pd.offsets.MonthEnd(0)
    fx = bt2.usdzar().resample("ME").mean()
    return (1 + g.Price.pct_change()) * (1 + fx.pct_change()) - 1


def month_end_levels():
    m = bt2.monthly_assets()
    lvl = (1 + m[bt2.RISKY]).cumprod()
    px = pd.read_csv(os.path.join(HERE, "three_fund_prices.csv"), parse_dates=["date"]).set_index("date")
    me = px[px.note.fillna("").str.contains("month end")]
    gp = gold_proxy_returns()
    rows = {}
    last = lvl.iloc[-1].copy()
    prev_px = None
    for d, r in me.iterrows():
        key = d + pd.offsets.MonthEnd(0)
        if key <= lvl.index[-1]:
            prev_px = r
            continue
        new = last.copy()
        for a, t in FUNDS.items():
            if prev_px is not None and pd.notna(r[t]) and pd.notna(prev_px[t]):
                new[a] = last[a] * r[t] / prev_px[t]
            elif a == "gold" and key in gp.index:
                new[a] = last[a] * (1 + gp[key])
            else:
                raise SystemExit(f"Missing {t} price for {d.date()}")
        rows[key] = new
        last, prev_px = new, r
    return pd.concat([lvl, pd.DataFrame(rows).T]).sort_index(), px


def main():
    lvl, px = month_end_levels()
    avg = lvl.rolling(10).mean()
    on = (lvl > avg)
    sig_date = on.index[-1]
    signal = on.iloc[-1]

    # Trades: the October check is acted on at the practice start (13 Nov); every later
    # month-end check is acted on at that month end's close, rebalancing to thirds.
    events = [(START, on.loc[START - pd.offsets.MonthEnd(1)])]
    for d in px.index[px.index > START]:
        key = d + pd.offsets.MonthEnd(0)
        if "month end" in str(px.loc[d, "note"]) and key in on.index:
            events.append((d, on.loc[key]))
    units = {t: 0.0 for t in FUNDS.values()}
    cash, cash_date, log = START_CASH, START, []
    for d, sig in events:
        p = px.loc[d]
        cash *= (1 + bt2.SA_CASH.get(d.year, .068)) ** ((d - cash_date).days / 365)
        cash_date = d
        value = cash + sum(units[t] * p[t] / 100 for t in units)
        target = {t: value / 3 if sig[a] else 0.0 for a, t in FUNDS.items()}
        turnover = sum(abs(target[t] - units[t] * p[t] / 100) for t in units)
        value -= turnover * COST
        target = {t: value / 3 if sig[a] else 0.0 for a, t in FUNDS.items()}
        units = {t: target[t] / (p[t] / 100) for t in units}
        cash = value - sum(target.values())
        log.append(dict(date=d.date(), **{t: ("fund" if sig[a] else "cash") for a, t in FUNDS.items()},
                        account=round(value, 2), costs=round(turnover * COST, 2)))
    pd.DataFrame(log).to_csv(os.path.join(HERE, "three_fund_trades.csv"), index=False)
    holdings = units
    last_d = px.index[-1]
    last = px.loc[last_d]
    cash_now = cash * (1 + bt2.SA_CASH.get(last_d.year, .068)) ** ((last_d - cash_date).days / 365)
    value = cash_now + sum(holdings[t] * last[t] / 100 for t in holdings)
    signal = events[-1][1]

    lines = ["# Three-fund practice portfolio (no real money)", "",
             f"Rule check at month end {sig_date.date()}; next check at the end of the following month.",
             f"Practice account: R{value:,.2f} on {last_d.date()} (started R{START_CASH:,.0f} on {START.date()})", "",
             "| Fund | Above 10-month average? | Holding | Units | Distance from average |",
             "|---|---|---|---|---|"]
    for a, t in FUNDS.items():
        dist = lvl[a].iloc[-1] / avg[a].iloc[-1] - 1
        lines.append(f"| {t} | {'yes' if signal[a] else 'no'} | {'fund' if signal[a] else 'money market'} | "
                     f"{holdings[t]:.2f} | {dist:+.1%} |")
    lines.append(f"\nMoney market: R{cash_now:,.2f}")
    with open(os.path.join(HERE, "three_fund_status.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
