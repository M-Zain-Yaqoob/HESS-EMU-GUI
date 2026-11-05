import numpy as np
import pandas as pd
from backend.core import (DEFAULT_VEH, DEFAULT_HESS, DEFAULT_EMU, synthetic_cycle,
    calculate_demand, simulate_rule_emu, apply_manual_scenario_intervals, diagnostics_summary)

veh=DEFAULT_VEH.copy(); hess=DEFAULT_HESS.copy(); emu=DEFAULT_EMU.copy()
cycle=synthetic_cycle('Urban stop-go', duration_s=180, seed=7)
demand=calculate_demand(cycle, veh, 0.0)
hess['bat_soc0']=0.7
labeled=simulate_rule_emu(demand, hess, emu)
assert len(labeled)==181
assert labeled['SOC_bat'].between(hess['bat_soc_min']-1e-9,hess['bat_soc_max']+1e-9).all()
assert labeled['SOC_uc'].between(0-1e-9,1+1e-9).all()
assert (labeled['P_bat_ref_kW'] <= hess['bat_dis_kW']+1e-9).all()
assert (labeled['P_bat_ref_kW'] >= -hess['bat_chg_kW']-1e-9).all()
assert (labeled['P_uc_ref_kW'] <= hess['uc_dis_kW']+1e-9).all()
assert (labeled['P_uc_ref_kW'] >= -hess['uc_chg_kW']-1e-9).all()
assert (labeled['P_fc_ref_kW'] <= hess['fc_max_kW']+1e-9).all()
assert np.allclose(labeled['Power_Balance_Error_kW'], labeled['Unmet_Power_kW']-labeled['Curtailed_Power_kW'], atol=1e-7)
manual=apply_manual_scenario_intervals(labeled, __import__('pandas').DataFrame([
    {'Start_s':0,'End_s':20,'Scenario_Label':'Test label'}]))
assert (manual.loc[manual.Time_s<20,'Scenario_Label']=='Test label').all()
diag=diagnostics_summary(labeled,hess)
assert diag['battery_power_violations']==diag['uc_power_violations']==diag['fc_power_violations']==0
print('Smoke tests passed:', len(labeled), 'samples; diagnostics:', diag)

# Four-input-compatible teacher: identical current operating points give identical references.
probe = demand.iloc[[40]].copy()
probe_a = simulate_rule_emu(probe, hess, emu)
probe_b = simulate_rule_emu(probe.copy(), hess, emu)
assert np.allclose(probe_a[['P_bat_ref_kW','P_uc_ref_kW','P_fc_ref_kW']].to_numpy(),
                   probe_b[['P_bat_ref_kW','P_uc_ref_kW','P_fc_ref_kW']].to_numpy())
print('Four-input teacher determinism test passed.')
