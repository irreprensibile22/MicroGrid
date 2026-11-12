# Mathematical formulation

Optimisation model that decides, for each hour of day D+1, how much to charge or discharge the battery and how much to exchange with the grid. It is a mixed-integer linear program (MILP), solved once a day.

Parameter values are in [`config/default.yaml`](../config/default.yaml); the matching key is given next to each symbol.

## 1. Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T} = \{1, \dots, T\}$ | Hours of local day D+1. $T = 24$, except on the days of the clock change ($T = 23$ or $T = 25$). |

Hour $t$ is the interval of length $\Delta t = 1\ \text{h}$ (`time.step_hours`) that ends at instant $t$. Powers are means over the interval; the state of charge $soc_t$ refers to the end of the interval.

## 2. Parameters

### Hourly series

| Symbol | Unit | Meaning | Source |
|---|---|---|---|
| $L_t$ | kW | Forecast load | `load_forecast` |
| $PV_t$ | kW | Forecast PV output | `pv_forecast` |
| $\pi_t$ | €/kWh | Day-ahead price | `price_eur_kwh` |
| $e_t$ | kgCO₂/kWh | Grid carbon intensity | `carbon_kg_kwh` |

Price and carbon intensity of day D+1 are taken as known when the decision is made (see [Assumptions](#7-assumptions-and-limits)).

### Exchange prices

| Symbol | Unit | Definition | Key |
|---|---|---|---|
| $\tau$ | €/kWh | Levies, taxes and network charges on imported energy | `grid.import_surcharge_eur_kwh` |
| $c_t$ | €/kWh | Buy price: $c_t = \pi_t + \tau$ | — |
| $s_t$ | €/kWh | Sell price: $s_t = \pi_t$ | — |

Since $\tau \ge 0$, we always have $s_t \le c_t$: buying and selling back in the same hour never pays. The sell price can be negative: in those hours selling costs money.

### Plant

| Symbol | Unit | Meaning | Key |
|---|---|---|---|
| $E^{max}$ | kWh | Battery capacity | `battery.capacity_kwh` |
| $P^{max}$ | kW | Maximum charge and discharge power | `battery.max_power_kw` |
| $\eta_c,\ \eta_d$ | — | Charge and discharge efficiencies | `battery.eta_charge`, `battery.eta_discharge` |
| $SOC^{min},\ SOC^{max}$ | kWh | State-of-charge limits: a fraction of $E^{max}$ | `battery.soc_min_frac`, `battery.soc_max_frac` |
| $SOC_0$ | kWh | Energy in the battery at the start of the day | see below |
| $G^{max}$ | kW | Maximum power exchanged with the grid | `grid.max_power_kw` |
| $\lambda$ | €/kgCO₂ | Weight of emissions in the objective | `objective.carbon_weight_eur_kg` |

$SOC_0$ is `battery.soc_initial_frac` $\cdot E^{max}$ on the first day of the simulation; on the following days it is the energy actually left in the battery at the end of the previous day.

## 3. Variables

| Symbol | Domain | Unit | Meaning |
|---|---|---|---|
| $p^{ch}_t$ | $\ge 0$ | kW | Battery charging power |
| $p^{dis}_t$ | $\ge 0$ | kW | Battery discharging power |
| $g^{imp}_t$ | $\ge 0$ | kW | Power bought from the grid |
| $g^{exp}_t$ | $\ge 0$ | kW | Power sold to the grid |
| $curt_t$ | $\ge 0$ | kW | Curtailed PV output |
| $soc_t$ | $\in \mathbb{R}$ | kWh | Energy in the battery at the end of hour $t$ |
| $u_t$ | $\{0, 1\}$ | — | 1 if the battery may charge, 0 if it may discharge |
| $v_t$ | $\{0, 1\}$ | — | 1 if we may buy, 0 if we may sell |

Charge and discharge powers are measured on the plant side: $p^{ch}_t$ is the power the battery draws, $p^{dis}_t$ the power it gives back. Losses are in the state-of-charge dynamics.

With $T = 24$ the model has 144 continuous variables and 48 binaries.

## 4. Objective function

$$
\min \ \sum_{t \in \mathcal{T}} \Big[ \underbrace{c_t \, g^{imp}_t - s_t \, g^{exp}_t}_{\text{net cost}} + \lambda \, \underbrace{e_t \, g^{imp}_t}_{\text{emissions}} \Big] \, \Delta t
$$

The first term is the net cost of the energy exchanged with the grid [€]. The second is the emissions of the imported energy [kgCO₂], converted into euros by the weight $\lambda$. Exported energy earns no emission credit.

With $\lambda = 0$ only cost is minimised. As $\lambda$ grows the plan moves purchases towards the hours when the grid is cleaner: the sequence of solutions as $\lambda$ changes is the cost–emissions Pareto front.

## 5. Constraints

All hold for every $t \in \mathcal{T}$, unless stated otherwise. In (2), for $t = 1$ the term $soc_{t-1}$ is the parameter $SOC_0$.

**(1) Power balance.** What enters the plant node equals what leaves it.

$$
PV_t - curt_t + p^{dis}_t + g^{imp}_t = L_t + p^{ch}_t + g^{exp}_t
$$

**(2) State-of-charge dynamics.**

$$
soc_t = soc_{t-1} + \eta_c \, p^{ch}_t \, \Delta t - \frac{p^{dis}_t \, \Delta t}{\eta_d}
$$

**(3) State-of-charge limits.**

$$
SOC^{min} \le soc_t \le SOC^{max}
$$

**(4) No simultaneous charging and discharging.**

$$
p^{ch}_t \le P^{max} \, u_t \qquad\qquad p^{dis}_t \le P^{max} \, (1 - u_t)
$$

**(5) No simultaneous buying and selling.**

$$
g^{imp}_t \le G^{max} \, v_t \qquad\qquad g^{exp}_t \le G^{max} \, (1 - v_t)
$$

**(6) PV curtailment.** We cannot curtail more than is produced.

$$
curt_t \le PV_t
$$

**(7) End of day.** The battery may not end the day emptier than it started.

$$
soc_T \ge SOC_0
$$

## 6. Properties

**The model always has a solution** if $SOC^{min} \le SOC_0 \le SOC^{max}$ and $L_t \le G^{max}$ for every $t$. Simply leave the battery idle: if $PV_t < L_t$ buy the difference, otherwise sell the surplus up to $G^{max}$ and curtail the rest. In the data the load never exceeds 50 kW and $G^{max} = 100$ kW.

**The coefficients in constraints (4) and (5) are as tight as they can be.** $P^{max}$ and $G^{max}$ are already the physical limits of the variables: no arbitrary "big-M" constants are needed, and the linear relaxation is strong.

**What the binary variables are for.** Without $u_t$ and $v_t$ the problem is a linear program (LP). The binaries change the solution only when wasting energy pays:

- *Simultaneous charging and discharging* dissipates energy through the efficiencies. Removing $\delta$ from the charge and $\eta_c \eta_d \, \delta$ from the discharge leaves the state of charge unchanged and frees the power $(1 - \eta_c \eta_d)\,\delta > 0$, which reduces the import or increases the curtailment. If $c_t + \lambda e_t \ge 0$ the cost does not get worse: waste never pays. If instead $c_t + \lambda e_t < 0$, we are paid to import and the linear program would charge and discharge at once to consume more: constraint (4) prevents it.
- *Simultaneous buying and selling* of a power $\delta$ costs $(c_t + \lambda e_t - s_t)\,\delta = (\tau + \lambda e_t)\,\delta \ge 0$. It never pays; with $\tau = 0$ and $\lambda = 0$ it makes no difference, and constraint (5) rules out physically meaningless solutions.

With the default parameters ($\tau = 0.20$ €/kWh) the buy price is always positive: the lowest day-ahead price in the period is −0.13 €/kWh. So the optimal value of the MILP equals that of its linear relaxation. The binaries stay in the model because they guarantee an executable plan for any choice of $\tau$ and $\lambda$, including in the sensitivity analysis.

**Constraint (7) avoids the end-of-horizon effect.** Without it the model would empty the battery in the last hour, because leftover energy has no value in the objective. The constraint uses the $SOC_0$ of the same day: it can always be met, whatever state the battery arrives in.

## 7. Assumptions and limits

- **Known prices.** Day-ahead prices for D+1 are published around 12:45 on day D: at midnight, when the decision is made, they are known. The carbon intensity of D+1 is not: the actual value is used, as if it were a perfect forecast. Emission results are therefore optimistic.
- **Simplified tariff.** The buy price is the day-ahead price plus a fixed surcharge $\tau$; the sell price is the day-ahead price. There are no peak-power charges and no feed-in incentives. The value of $\tau$ is an assumption: it sets how much self-consumption is worth compared with price arbitrage, and it is varied in the sensitivity analysis.
- **Ideal battery.** Constant efficiencies, no self-discharge, no degradation cost, maximum power independent of the state of charge.
- **One-day horizon.** The plan does not look past midnight. Constraint (7) forces every day to close at least at its starting level: stored energy cannot be carried into the next day below $SOC_0$, even when that would pay.
- **Deterministic model.** Forecasts are treated as certain. Deviations between forecast and reality are handled during the simulation, not by the model.
- **Active power only**, on a single node: no internal network, no line losses.

## 8. Possible extensions

- Degradation cost: a term $\gamma \, (p^{ch}_t + p^{dis}_t) \, \Delta t$ in the objective.
- Peak-power charge: a variable $g^{peak} \ge g^{imp}_t$ with a cost in the objective.
- Value of leftover energy instead of constraint (7): a term $-\,\omega \, soc_T$ in the objective.
- 48-hour horizon, applying only the first 24 hours.
