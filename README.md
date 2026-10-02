# service-desk-forecaster

Real data, real schedule: NYC 311 requests from [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) (22.7M rows as of 2026-10-02), refreshed daily by a GitHub Action.

**Live UI:** https://rishikeshn-eng.github.io/service-desk-forecaster/

Three jobs:
1. **Volume**: forecast each of the next 7 days, with a rolling-origin backtest and an 80% band.
2. **Routing**: predict the responsible agency.
3. **Resolution time**: predict hours to close.

## Results (real data, as of the last pipeline run; the live site shows the latest)

**Volume (daily totals, all 22.7M rows via server-side aggregation).** 52 weekly origins, 364 forecasts, each model trained only on rows whose target date precedes its origin.

| | WAPE | MAE (requests/day) |
|---|---|---|
| Gradient boosting (7 horizons, calendar + holiday + lag-ratio features) | **8.7%** | 946 |
| Same weekday last week | 11.4% | 1,235 |

The 80% band is built from backtest relative errors. Calibrated on the older half of the backtest, it covered 96% of actuals in the newer half, so it is conservative (recent errors are smaller than older ones). Weather and one-off events are not modelled.

**Routing (160,857-ticket sample, test = last 30 days of the sample).** The operator's `complaint_type` practically fixes the agency, so the main model deliberately does not see it. It sees only descriptor, location type, address type, borough and intake channel.

| | accuracy | macro-F1 |
|---|---|---|
| Majority class | 46.1% | n/a |
| **Without complaint type** | **98.5%** | 0.977 |
| With complaint type (ceiling) | 99.2% | 0.986 |

The remaining errors are mostly DEP tickets sent to NYPD/HPD/DOT, i.e. generic descriptors that several agencies share.

**Resolution time (29,816 closed test tickets, median 6.3 h).**

| | MAE (h) | median AE (h) | within 2x of truth |
|---|---|---|---|
| Per-complaint-type median | 98.0 | 5.1 | 57.6% |
| Gradient boosting | 94.7 | 5.1 | 59.5% |

This is a small win, not a good model: 3% lower MAE. Resolution time here is mostly determined by complaint type; borough, zip, descriptor and time of day add little. Two caveats: I picked this variant after comparing 16 feature/loss combinations on the same test month, so even 3% is optimistic; and only tickets closed within 90 days are scored (763 slower ones excluded), on a sample that ends 45 days before the run, so the long tail is understated.

### Sampling and what is not covered
- Ticket sample: `unique_key` ending in `7` (a deterministic ~10% sample), created 2026-03-21 to 2026-08-18. Not the full 22.7M rows.
- The volume model uses 2022-01-01 onward to avoid the 2020 COVID regime.
- The newest API day is partial and is dropped.

## The scheduled pipeline

`.github/workflows/refresh.yml` runs daily at 03:30 UTC (and on demand):
1. `pytest`
2. `python -m sd311.pipeline refresh`: pull daily counts, re-run the 52-week backtest, forecast the next 7 days, write `docs/forecast.json`
3. On Mondays or manual runs: `python -m sd311.pipeline tickets` re-pulls the ticket sample and retrains routing and resolution, writing `docs/tickets.json` (this step is slow: the API takes ~10 minutes for 160k rows)
4. Commit the JSON if it changed; the site reads those files, so Pages updates itself.

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q
python -m sd311.pipeline refresh     # ~2 min
python -m sd311.pipeline tickets     # ~10 min first time, cached in data/ after
python -m http.server -d docs        # open http://localhost:8000
```
Set `SOCRATA_APP_TOKEN` to avoid API throttling.
