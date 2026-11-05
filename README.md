# HESS Energy Management Studio — Version 3

A Streamlit research GUI for generating one-second EV HESS drive-cycle datasets, running a four-input-compatible battery/ultracapacitor/fuel-cell teacher allocator, sweeping initial SOC conditions, manually labeling scenario intervals, training a scaled ANN, and exporting datasets/results.

## Run on Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run on Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## New in Version 3

- The teacher allocator no longer uses hidden previous-power/filter states. Its source references are deterministic from the four current ANN inputs and current SOC values.
- Fuel-cell and battery actuator response/ramp dynamics are explicitly separated from the four-input teacher labels. Apply them in a downstream actuator/controller layer if needed.
- Validation uses a complete SOC run from the training cycle families; testing holds out a complete cycle family when multiple families are supplied.
- ANN test plot is downsampled for readability while exported predictions retain every test sample.
- Training fails clearly when there are not enough independent groups to make a meaningful held-out evaluation.

## New in Version 2

- Multiple initial battery SOC × UC energy-SOC combinations per cycle family. Each run starts from a reset energy state.
- Whole cycle families are held out for ANN testing, so SOC repetitions of the held-out family do not leak into training.
- Optional manual scenario intervals. Intervals use `[Start_s, End_s)`: start is included and end is excluded. Uncovered samples retain automatic labels.
- Automatic labels include Stopped/Idle, Acceleration, Deceleration, Cruising, Uphill, Downhill, and Regenerative Braking.
- Explicit power-balance error, unmet traction power, unabsorbed/curtailed surplus, source power-limit violations, and SOC-bound violations.
- Per-run diagnostic table for each cycle/initial-state combination.

## CSV upload format

Upload a CSV with `time_s` (seconds) and `speed_kmh` (km/h). The app interpolates it onto a one-second time grid. A standard drive-cycle name does not make a trace official: built-in profiles are synthetic. Use official trace files for WLTP/UDDS/FTP-75 claims.

## Dataset columns

- `Time_s`, `Cycle_Name`, `Cycle_Family`, `Scenario_Label`
- `Speed_kmh`, `Acceleration_ms2`, `Road_Grade_Percent`
- `P_dem_kW`, `Delta_P_dem_kW`
- `SOC_bat`, `SOC_uc`, `Initial_SOC_bat`, `Initial_SOC_uc`
- `P_bat_ref_kW`, `P_uc_ref_kW`, `P_fc_ref_kW`
- `Unmet_Power_kW`, `Curtailed_Power_kW`, `Source_Sum_kW`, `Power_Balance_Error_kW`

`SOC_uc` is the UC energy fraction relative to its rated maximum energy, i.e. `E_uc / E_uc,max`; initial UC voltage is set using `V0 = Vmax * sqrt(SOC_uc,initial)` and must not fall below the minimum-voltage energy fraction.

## ANN interface (unchanged)

Inputs: `[P_dem_kW, SOC_bat, SOC_uc, Delta_P_dem_kW]`

Targets: `[P_bat_ref_kW, P_uc_ref_kW, P_fc_ref_kW]`

`Scenario_Label` remains descriptive metadata, not an ANN input. The ANN is trained to imitate the rule-based labels; this does not by itself prove optimality or improved fuel economy.

## Modeling limitations

- Default vehicle and component values are illustrative, editable assumptions, not verified hardware ratings.
- Built-in drive profiles are synthetic demonstrations, not official certification cycles.
- The vehicle model uses a constant road grade per generated/uploaded trace. Uploaded speed-time data do not inherently contain road grade.
- The rule-based allocator is a simplified instantaneous supervisory teacher, not a detailed battery ECM, UC ESR model, fuel-cell electrochemical model, converter model, actuator dynamics model, or switching-level DC bus simulation. FC ramp and source response-time values are configuration notes for downstream implementation; they are not used to generate the stateless teacher labels.
- Power shortfalls and surplus that cannot be absorbed are surfaced as diagnostics; inspect them before treating generated labels as feasible reference targets.
- A held-out cycle-family test is more informative than a random row split, but a synthetic Highway family may have a substantially different distribution from the synthetic training families. Include diverse training profiles and test with official cycle traces before making generalization claims.
