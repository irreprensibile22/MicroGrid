# Microgrid energy management — Project roadmap

A "predict-then-optimise" pipeline: an ML model forecasts next-day load and PV output, and a MILP model decides hour by hour how to use the battery and the grid to minimise cost and emissions.

Every phase ends with something publishable: make one commit (or more) at the end of each.

---

## Phase 0 — Problem definition

1. Choose the scenario: a building or small campus with PV, a battery and a grid connection.
2. Fix the plant parameters (reasonable starting values, to be varied later):
   - PV peak power (e.g. 100 kWp)
   - battery capacity (e.g. 200 kWh) and maximum charge/discharge power (e.g. 50 kW)
   - charge and discharge efficiencies (e.g. 0.95 each)
   - minimum and maximum state of charge (e.g. 10%–90%)
3. Fix resolution and horizon: hourly step, planning over the next 24 hours.
4. Write in one sentence the question the project answers, for example: "How much do we save in cost and CO₂ by running the battery with forecasting + optimisation rather than with a simple rule?"

**Output:** the "Problem" section of the README.

---

## Phase 1 — Repository setup

1. Create the GitHub repository with a licence (MIT) and a Python `.gitignore`.
2. Create the layout:

   ```
   microgrid-ems/
   ├── README.md
   ├── pyproject.toml          # dependencies and configuration
   ├── config/
   │   └── default.yaml        # plant parameters, paths, seed
   ├── data/
   │   ├── raw/                # downloaded data (not versioned)
   │   └── processed/          # clean dataset (not versioned)
   ├── notebooks/              # exploratory analysis and charts only
   ├── src/microgrid/
   │   ├── data.py             # download and cleaning
   │   ├── features.py         # feature construction
   │   ├── forecast.py         # forecasting models
   │   ├── optimize.py         # MILP model
   │   ├── simulate.py         # rolling-horizon simulation
   │   └── evaluate.py         # metrics and comparisons
   ├── scripts/
   │   └── run_all.py          # runs the whole pipeline
   ├── tests/
   └── docs/
       └── formulation.md      # mathematical formulation
   ```

3. Create a virtual environment and install: `pandas`, `numpy`, `lightgbm`, `pyomo` (or `pulp`), `highspy`, `matplotlib`, `pyyaml`, `pytest`.
4. Add `data/` to `.gitignore`: data is re-downloaded with a script, not uploaded to the repo.

**Output:** an empty but tidy repo, first commit.

---

## Phase 2 — Data collection

Five hourly series are needed, over the same period (at least 1 year, better 2) and the same area:

| Series | Possible sources |
|---|---|
| Electrical load | UCI ML Repository, Open Power System Data |
| PV output | PVGIS (simulated for a location), Open Power System Data |
| Weather (temperature, irradiance, cloud cover) | Open-Meteo (historical archive) |
| Energy price | ENTSO-E Transparency Platform (day-ahead prices) |
| Grid carbon intensity (gCO₂/kWh) | Electricity Maps, or computed from the ENTSO-E generation mix |

1. Check the licence and time coverage of each source before choosing it.
2. Write in `data.py` one download function per series, saving to `data/raw/`.
3. If the load is at a different scale from your scenario (e.g. national), rescale it to the desired peak and say so in the README.
4. Record source, period and licence of every dataset in the README.

**Output:** `python -m microgrid.data` downloads everything from scratch.

---

## Phase 3 — Cleaning and exploration

1. Put all series on a single hourly index with an explicit time zone (watch out for daylight-saving changes).
2. Handle missing values: interpolation for short gaps, exclusion of days with long gaps.
3. Check for anomalous values (negative load, PV at night).
4. Merge everything into one file, `data/processed/dataset.parquet`.
5. In a notebook, explore:
   - mean daily and weekly load profile
   - seasonality of load and PV
   - load–temperature correlation
   - price and carbon intensity over the 24 hours
6. Split the data **chronologically**: train (e.g. first 70%), validation (15%), test (last 15%). Never shuffle.

**Output:** clean dataset + notebook `01_exploration.ipynb` with 4–5 charts.

---

## Phase 4 — Forecasting (machine learning)

The goal is to forecast, at a given hour of day D, the 24 hours of load and PV of day D+1.

1. **Baseline:** naive forecast (same hour 7 days earlier for load, same hour of the previous day for PV). Every model is compared against it.
2. **Features:**
   - calendar: hour, day of week, month, holiday (also sine/cosine encoded)
   - lags: values 24, 48 and 168 hours earlier
   - rolling statistics: mean of the last 24 hours and 7 days
   - weather forecast for day D+1
3. **Avoid leakage:** use only information available when the forecast is made. No lags shorter than 24 hours.
4. **Note on weather:** using observed weather as if it were a forecast is an optimistic simplification. State it in the README, or use archived weather forecasts if available.
5. **Main model:** gradient boosting (LightGBM), one for load and one for PV.
6. **Optional model:** an LSTM or an MLP, for comparison only.
7. **Validation:** time-series cross-validation (expanding windows) to choose the hyperparameters.
8. **Metrics** on the test set: MAE, RMSE and normalised MAE; report them in a baseline-vs-models table.
9. Charts: forecast vs actual over a week, error by hour of day.
10. Save the trained models and the test-set forecasts.

**Output:** `forecast.py`, metrics table, notebook `02_forecasting.ipynb`.

---

## Phase 5 — Formulation of the optimisation model

Write it in `docs/formulation.md` before coding it.

**Sets and parameters** (for each hour t = 1…24, with Δt = 1 h):

- `L_t` forecast load, `PV_t` forecast PV [kW]
- `c_t` buy price, `s_t` sell price [€/kWh]
- `e_t` grid carbon intensity [kgCO₂/kWh]
- `E_max` battery capacity, `P_max` maximum power, `η_c`, `η_d` efficiencies
- `SOC_min`, `SOC_max`, `SOC_0` initial state of charge
- `G_max` maximum power exchanged with the grid
- `λ` emission weight [€/kgCO₂]

**Variables:**

- `p_ch_t`, `p_dis_t` ≥ 0 — charging and discharging power
- `g_imp_t`, `g_exp_t` ≥ 0 — power bought and sold
- `curt_t` ≥ 0 — curtailed PV
- `soc_t` — energy in the battery
- `u_t` ∈ {0,1} — 1 if the battery charges, 0 if it discharges
- `v_t` ∈ {0,1} — 1 if we buy, 0 if we sell

**Objective function:**

```
min  Σ_t [ c_t · g_imp_t − s_t · g_exp_t + λ · e_t · g_imp_t ] · Δt
```

**Constraints:**

```
Balance:       PV_t − curt_t + p_dis_t + g_imp_t = L_t + p_ch_t + g_exp_t
SOC dynamics:  soc_t = soc_{t−1} + η_c · p_ch_t · Δt − p_dis_t · Δt / η_d
SOC limits:    SOC_min ≤ soc_t ≤ SOC_max
Charge:        p_ch_t  ≤ P_max · u_t
Discharge:     p_dis_t ≤ P_max · (1 − u_t)
Import:        g_imp_t ≤ G_max · v_t
Export:        g_exp_t ≤ G_max · (1 − v_t)
End of day:    soc_24 ≥ SOC_0
PV curtailment: curt_t ≤ PV_t
```

Notes:

1. Without `u_t` and `v_t` the problem is a pure LP. The binaries make it a MILP, solved by the solver with branch and cut.
2. The end-of-day constraint stops the model from emptying the battery in the last hour.
3. Optional extensions: battery degradation cost (a term proportional to the energy cycled), peak-power charge, diesel generator with binary start-up.

**Output:** complete `docs/formulation.md`.

---

## Phase 6 — Implementation of the optimiser

1. Implement in `optimize.py` a function that takes forecasts, prices, carbon intensity, parameters and initial SOC, and returns the hourly plan.
2. Use Pyomo (or PuLP) with the HiGHS solver.
3. Try it first on a single day with made-up data whose expected solution you know (e.g. low price at night and high in the evening: it must charge at night and discharge in the evening).
4. Check the solver status (optimal, infeasible) and handle the errors.
5. Write the tests in `tests/`:
   - the energy balance holds in every hour
   - the SOC stays within its limits
   - never charging and discharging in the same hour
   - with a zero-capacity battery, the result matches the no-battery case
6. Plot the plan of one day: load, PV, grid exchange and SOC.

**Output:** tested `optimize.py`, chart of a typical day.

---

## Phase 7 — Rolling-horizon simulation

This is where the two blocks come together. For each day of the test set:

1. Generate the load and PV forecasts for the next day.
2. Solve the MILP with the forecasts and get the battery plan.
3. Apply the battery plan to the **actual data**: the difference between forecast and reality is absorbed by the grid (you buy more or sell less).
4. If the plan cannot be executed on the actual data (e.g. battery full earlier than expected), apply a simple correction rule and document it.
5. Compute the day's actual cost and emissions.
6. Pass the final SOC on as the initial SOC of the next day.

**Output:** `simulate.py`, daily series of costs and emissions on the test set.

---

## Phase 8 — Comparisons and analysis

Run the same simulation with four strategies:

| Strategy | Description |
|---|---|
| No battery | PV covers the load, the rest goes to or comes from the grid |
| Naive rule | Charge when there is a solar surplus, discharge when there is a deficit |
| Forecast + MILP | Your system |
| Perfect forecast + MILP | Upper bound: optimisation on the actual data |

1. Summary table: total cost, total emissions, percentage saving relative to "no battery".
2. Compute how much of the theoretical saving (perfect forecast) your system captures: it measures the value of the forecast.
3. **Pareto front:** repeat the simulation varying `λ` and plot cost against emissions.
4. **Sensitivity analysis:** vary battery capacity and PV power and see how the saving changes.
5. Look at the worst days: when the forecast is most wrong, how much does it cost?

**Output:** `evaluate.py`, notebook `03_results.ipynb`, final charts.

---

## Phase 9 — Quality and reproducibility

1. `scripts/run_all.py` runs everything: download, cleaning, training, simulation, charts.
2. All parameters live in `config/default.yaml`, no numbers scattered through the code.
3. Fix the random seeds.
4. Add `ruff` for formatting and linting.
5. Set up GitHub Actions: lint and tests on every push.
6. Add docstrings to the public functions, and type hints.
7. Check reproducibility: clone the repo into a new folder and follow the README from scratch.

**Output:** green CI badge in the README.

---

## Phase 10 — README and presentation

README structure:

1. Title and a one-sentence description.
2. **Main result at the top:** one chart and 2–3 numbers (e.g. "−X% cost, −Y% CO₂ compared with no battery").
3. Pipeline diagram (data → forecast → optimisation → simulation).
4. How to run: 3–4 commands.
5. Data: sources, period, licences.
6. Method: summary of forecasting and formulation, with a link to `docs/formulation.md`.
7. Results: comparison table, Pareto front, sensitivity.
8. Stated limits and simplifications.
9. Possible developments.

Then:

1. Add a description and topics to the repository (`energy`, `optimization`, `milp`, `forecasting`, `machine-learning`).
2. Pin the repository on your GitHub profile.

**Output:** a presentable repository.

---

## Phase 11 — Optional extensions

1. Streamlit dashboard: pick the battery size and the weight `λ`, see the resulting plan.
2. Probabilistic forecasting (quantiles) and stochastic or robust optimisation over several scenarios.
3. Battery degradation cost in the objective.
4. Electric-vehicle charging as a shiftable load.
5. Intraday re-optimisation every hour (true model predictive control).

---

## Summary checklist

- [x] Phase 0 — Problem and parameters defined
- [x] Phase 1 — Repository set up
- [x] Phase 2 — Data downloadable with a script
- [x] Phase 3 — Dataset cleaned and explored
- [x] Phase 4 — Forecast better than the baseline (PV yes, MAE −39%; load no on the test set: MAE +4%, RMSE −10%)
- [x] Phase 5 — Formulation written
- [x] Phase 6 — Optimiser implemented and tested
- [x] Phase 7 — Rolling-horizon simulation
- [x] Phase 8 — Comparisons, Pareto, sensitivity
- [x] Phase 9 — CI, tests, reproducibility
- [x] Phase 10 — Complete README
- [ ] Phase 11 — Extensions (optional)
