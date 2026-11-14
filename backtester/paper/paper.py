"""
Practice (paper) trading of the one candidate rule from bt.py. No real money.

Rule (Bollinger 50-day, 1.5 sd) on Satrix 40:
  * buy level  = 50-day average - 1.5 x standard deviation
  * sell level = 50-day average
  * out and close < buy level  -> buy at the NEXT day's close
  * in and close > sell level   -> sell at the NEXT day's close

Daily routine: add the day's STX40 close (in cents) to prices.csv, then run
    python3 paper/paper.py
It rewrites status.md and trades.csv. A virtual R10,000 starts on 13 Nov 2026.
Note on dividends: prices.csv holds plain closes. On an ex-dividend day enter
the dividend in cents in the dividend_cents column so the drop is not mistaken
for a fall.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bt  # noqa: E402

START = pd.Timestamp("2026-11-13")
START_CASH = 10000.0
COST = bt.COST_BASE
N, K = 50, 1.5


def build_index():
    df = bt.load_jse_daily()
    p = df.price
    pr = pd.read_csv(os.path.join(HERE, "prices.csv"), parse_dates=["date"]).set_index("date")
    if "dividend_cents" not in pr:
        pr["dividend_cents"] = 0
    pr["dividend_cents"] = pr["dividend_cents"].fillna(0)
    # total return from plain closes: (close + dividend) / previous close
    tr = (pr.close_cents + pr.dividend_cents) / pr.close_cents.shift()
    ext = p.loc[pr.index[0]] * tr.iloc[1:].cumprod()
    idx = pd.concat([p[p.index <= pr.index[0]], ext])
    return idx, pr


def main():
    idx, pr = build_index()
    mid = idx.rolling(N).mean()
    lo = mid - K * idx.rolling(N).std()
    state = bt.hold_state((idx < lo).fillna(False), (idx > mid).fillna(False))

    # virtual account from START: trade the day after the signal changes
    days = pr.index[pr.index >= START]
    cash, units, held = START_CASH, 0.0, 0
    trades, rows = [], []
    for i, d in enumerate(days):
        close = pr.close_cents[d] / 100
        if i > 0:
            prev = days[i - 1]
            gap = (d - prev).days
            cash *= (1 + bt.SA_CASH.get(d.year, .068)) ** (gap / 365)
            units_div = units * pr.dividend_cents[d] / 100
            cash += units_div
            want = int(state[prev])
            if want != held:
                if want == 1:
                    units = cash * (1 - COST) / close
                    trades.append(dict(date=d.date(), action="BUY", price_rand=close, amount=round(cash, 2)))
                    cash = 0.0
                else:
                    cash = units * close * (1 - COST)
                    trades.append(dict(date=d.date(), action="SELL", price_rand=close, amount=round(cash, 2)))
                    units = 0.0
                held = want
        value = cash + units * close
        rows.append(dict(date=d.date(), close_rand=close, buy_level_rand=round(lo[d] / idx[d] * close, 2),
                         sell_level_rand=round(mid[d] / idx[d] * close, 2), rule="IN" if state[d] else "OUT",
                         holding="Satrix 40" if held else "cash", value=round(value, 2)))

    hist = pd.DataFrame(rows)
    hist.to_csv(os.path.join(HERE, "daily_log.csv"), index=False)
    pd.DataFrame(trades, columns=["date", "action", "price_rand", "amount"]).to_csv(
        os.path.join(HERE, "trades.csv"), index=False)

    last = hist.iloc[-1]
    bh = START_CASH * pr.close_cents.iloc[-1] / pr.close_cents[START]
    if last.rule == "OUT" and last.holding == "cash":
        todo = f"Stay in cash. Buy only if a close falls below R{last.buy_level_rand:.2f}."
    elif last.rule == "IN" and last.holding == "cash":
        todo = "BUY Satrix 40 at the next close."
    elif last.rule == "IN":
        todo = f"Keep holding. Sell once a close is above R{last.sell_level_rand:.2f}."
    else:
        todo = "SELL Satrix 40 at the next close."
    md = [
        "# Practice trading status (no real money)",
        "",
        f"Last close: {last.date} at R{last.close_rand:.2f}",
        f"Next step: {todo}",
        f"Practice account: R{last.value:,.2f} (started with R{START_CASH:,.0f} on {START.date()})",
        f"Buy-and-hold over the same days: R{bh:,.2f}",
        f"Trades so far: {len(trades)}",
        "",
        "| Date | Close | Buy level | Sell level | Rule | Holding | Account |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in hist.iloc[::-1].iterrows():
        md.append(f"| {r.date} | R{r.close_rand:.2f} | R{r.buy_level_rand:.2f} | R{r.sell_level_rand:.2f} | "
                  f"{r.rule} | {r.holding} | R{r.value:,.2f} |")
    with open(os.path.join(HERE, "status.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:7]))


if __name__ == "__main__":
    main()
