# Trading strategy backtester (JSE, rand, real costs)

Tests well-known trading strategies the way a South African retail investor would
actually trade them: Satrix 40 / Satrix S&P 500 ETFs on EasyEquities, inside a
tax-free savings account.

```
python3 bt.py
```

Needs only `pandas` and `numpy`. Writes:

- `results/summary.md` – the readable summary
- `results/jse_daily_all_results.csv` – every strategy, cost level and period
- `results/walk_forward_picks.csv` – which strategy the walk-forward picked each year
- `results/monthly_rand_results.csv` – Top 40 / S&P 500 / cash switching rules
- `results/us_150_years_results.csv` – long-history check of trend rules

## Method

- Tune on 2010-2019 only, grade on 2020-2026 (never used for choosing).
- Trades happen the day after the signal; a same-close version is shown too.
- 0.40% cost per buy or sell (brokerage 0.25% + VAT + levies + spread), also 0.20% and 0.60%.
- Uninvested money earns an approximate SA money-market rate.
- Deflated Sharpe ratio corrects the best result for the number of variants tried.
- Walk-forward re-picks the best strategy every January from the previous five years.

## Data (`data/`)

| File | What | Source |
|---|---|---|
| `stx40_daily_total_return.csv` | Satrix 40 daily total return, 2010-2026 | github.com/Tjay-cmd/jse-top40-portfolio-optimiser (cross-checked with github.com/andrewboyley/portfolio-optimiser) |
| `sp500_monthly_shiller.csv` | S&P 500 monthly price, dividends, rates since 1871 | github.com/datasets/s-and-p-500 (Robert Shiller) |
| `usdzar_daily_1990.csv`, `usdzar_daily_2016.csv` | Rand/dollar daily | github.com/Aakashanil67/usdzar-forecasting, github.com/stSoftwareAU/GRQ-FX-validation |

To update for paper trading, append new daily returns to `stx40_daily_total_return.csv`
and rerun; section D of the summary prints the current signal.

This is research, not financial advice. Nothing here places trades.
