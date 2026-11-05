from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["P_dem_kW", "SOC_bat", "SOC_uc", "Delta_P_dem_kW"]
TARGETS = ["P_bat_ref_kW", "P_uc_ref_kW", "P_fc_ref_kW"]

DEFAULT_VEH = dict(mass_kg=1500.0, area_m2=2.2, rho=1.225, Cd=0.29, Crr=0.012,
                   eta_drivetrain=0.92, eta_regen=0.75, aux_W=500.0, max_motor_W=80000.0)
DEFAULT_HESS = dict(bat_cap_kWh=1.54, bat_soc0=0.70, bat_soc_min=0.20, bat_soc_max=0.90,
                    bat_dis_kW=35.0, bat_chg_kW=20.0, bat_eta=0.95,
                    uc_cap_F=165.0, uc_v_min=30.0, uc_v_max=60.0, uc_v0=48.0,
                    uc_dis_kW=15.0, uc_chg_kW=15.0, uc_eta=0.95,
                    fc_max_kW=15.0, fc_min_kW=0.0, fc_ramp_kWps=2.0, fc_tau_s=8.0,
                    bat_tau_s=2.0)
DEFAULT_EMU = dict(uc_transient_factor=0.8, fc_base_fraction=0.35, fc_soc_reference=0.65)

def synthetic_cycle(kind: str, duration_s: int = 1800, max_speed_kmh: float = 90, seed: int = 42) -> pd.DataFrame:
    """Synthetic illustrative speed traces. Not official certification drive cycles."""
    rng=np.random.default_rng(seed); t=np.arange(int(duration_s)+1,dtype=float); n=len(t)
    if kind == "Urban stop-go":
        v=np.zeros(n); cur=0.0
        for i in range(1,n):
            if i%85<10: target=0
            elif i%85<38: target=max_speed_kmh*rng.uniform(.35,.75)
            else: target=max_speed_kmh*rng.uniform(.15,.55)
            cur += np.clip(target-cur,-2.5,1.2); v[i]=max(0,cur)
    elif kind == "Highway":
        v=np.clip(max_speed_kmh*.78 + 4*np.sin(t/75)+rng.normal(0,0.7,n),0,max_speed_kmh)
        v[:20]=np.linspace(0,v[20],20)
    elif kind == "Aggressive acceleration":
        v=np.clip(max_speed_kmh*.35+max_speed_kmh*.25*np.sin(t/13)+max_speed_kmh*.2*np.sin(t/4),0,max_speed_kmh)
        v[:10]=np.linspace(0,v[10],10)
    else: # mixed
        v=np.clip(max_speed_kmh*(.42+.22*np.sin(t/37)+.13*np.sin(t/9)),0,max_speed_kmh)
        v[:30]=np.linspace(0,v[30],30)
        v[(t.astype(int)%180)<18]=0
    return pd.DataFrame({"Time_s":t,"Speed_kmh":v})

def load_uploaded_cycle(file) -> pd.DataFrame:
    df=pd.read_csv(file)
    lower={c.lower().strip():c for c in df.columns}
    tc=next((lower[k] for k in ["time_s","time","time (s)"] if k in lower),None)
    sc=next((lower[k] for k in ["speed_kmh","speed (km/h)","speed_kph","speed"] if k in lower),None)
    if tc is None or sc is None: raise ValueError("CSV must contain time_s/time and speed_kmh/speed columns.")
    out=pd.DataFrame({"Time_s":pd.to_numeric(df[tc],errors="coerce"),"Speed_kmh":pd.to_numeric(df[sc],errors="coerce")}).dropna()
    out=out[(out.Time_s>=0)&(out.Speed_kmh>=0)].sort_values("Time_s").drop_duplicates("Time_s")
    if len(out)<3: raise ValueError("The uploaded cycle needs at least 3 valid samples.")
    # Resample to one-second grid for consistent interval semantics.
    grid=np.arange(int(np.ceil(out.Time_s.min())),int(np.floor(out.Time_s.max()))+1)
    if len(grid)<3: raise ValueError("Cycle duration is too short.")
    speed=np.interp(grid,out.Time_s,out.Speed_kmh)
    return pd.DataFrame({"Time_s":grid.astype(float),"Speed_kmh":speed})

def calculate_demand(cycle: pd.DataFrame, veh: dict, grade_pct: float=0.0) -> pd.DataFrame:
    d=cycle.copy().sort_values("Time_s").reset_index(drop=True)
    t=d.Time_s.to_numpy(float); v=d.Speed_kmh.to_numpy(float)/3.6
    a=np.gradient(v,t)
    grade=np.full(len(d),float(grade_pct))
    theta=np.arctan(grade/100.0)
    force=(veh['mass_kg']*a + veh['mass_kg']*9.81*np.sin(theta)
           + veh['mass_kg']*9.81*veh['Crr']*np.cos(theta)
           + .5*veh['rho']*veh['Cd']*veh['area_m2']*v**2)
    p_wheel=force*v
    p_wheel=np.clip(p_wheel,-veh['max_motor_W'],veh['max_motor_W'])
    p_dc=np.where(p_wheel>=0,p_wheel/max(veh['eta_drivetrain'],.01),p_wheel*veh['eta_regen'])+veh['aux_W']
    p_dc[v<1e-8]=0.0
    d['Acceleration_ms2']=a; d['Road_Grade_Percent']=grade
    d['P_dem_kW']=p_dc/1000
    d['Delta_P_dem_kW']=np.r_[0,np.diff(d.P_dem_kW)]
    return d

def classify_scenarios(df: pd.DataFrame) -> pd.Series:
    v=df.Speed_kmh.to_numpy(); a=df.Acceleration_ms2.to_numpy(); g=df.Road_Grade_Percent.to_numpy(); p=df.P_dem_kW.to_numpy()
    labels=np.full(len(df),'Cruising',dtype=object)
    labels[v<1.0]='Stopped/Idle'
    labels[(v>=1)&(a>0.35)]='Acceleration'
    labels[(v>=1)&(a<-.35)]='Deceleration'
    labels[(v>=1)&(np.abs(a)<=.35)]='Cruising'
    labels[g>1.0]='Uphill'; labels[g< -1.0]='Downhill'
    labels[p<-.05]='Regenerative Braking'
    return pd.Series(labels,index=df.index)

def simulate_rule_emu(df: pd.DataFrame, hess: dict, emu: dict) -> pd.DataFrame:
    """Stateless-instantaneous supervisory allocator plus SOC state integration.

    For the four-input ANN interface, source references are deterministic
    functions of [P_dem, SOC_bat, SOC_uc, Delta_P_dem]. No hidden filter or
    previous-power state is used in the teacher labels. Positive source power
    discharges into the DC bus; negative battery/UC power charges them.

    FC ramp dynamics are intentionally not embedded in these labels because
    previous FC power is not an ANN input. Enforce FC ramp limits in a downstream
    actuator/controller layer if the physical implementation requires them.
    """
    out=df.copy().sort_values('Time_s').reset_index(drop=True)
    n=len(out); t=out.Time_s.to_numpy(float)
    demand=out.P_dem_kW.to_numpy(float)
    delta=out.Delta_P_dem_kW.to_numpy(float)

    bat_cap=float(hess['bat_cap_kWh'])
    emin=bat_cap*float(hess['bat_soc_min']); emax=bat_cap*float(hess['bat_soc_max'])
    ebat=bat_cap*np.clip(float(hess['bat_soc0']),hess['bat_soc_min'],hess['bat_soc_max'])
    c=float(hess['uc_cap_F']); vmin=float(hess['uc_v_min']); vmax=float(hess['uc_v_max'])
    vu=np.clip(float(hess['uc_v0']),vmin,vmax)
    euc=.5*c*vu**2/3.6e6; eucmin=.5*c*vmin**2/3.6e6; eucmax=.5*c*vmax**2/3.6e6

    pfc=np.zeros(n); pbat=np.zeros(n); puc=np.zeros(n)
    unmet=np.zeros(n); curtailed=np.zeros(n); soc_bat=np.zeros(n); soc_uc=np.zeros(n)
    for k in range(n):
        dt=max(1e-6,t[k]-t[k-1]) if k else 1.0
        dem=float(demand[k]); traction=max(dem,0.0)
        sb=ebat/bat_cap; su=euc/max(eucmax,1e-12)

        # Fuel cell supplies a smooth base share; SOC support is instantaneous
        # and therefore fully observable from the ANN's current inputs.
        fc_request=(emu['fc_base_fraction']*traction
                    + max(0.0,emu['fc_soc_reference']-sb)*hess['fc_max_kW'])
        pfc[k]=float(np.clip(fc_request,0.0,hess['fc_max_kW'])) if traction>0 else 0.0
        residual=dem-pfc[k]

        if residual>=0:
            # UC provides a bounded share of positive demand transients.
            uc_request=emu['uc_transient_factor']*max(0.0,float(delta[k]))
            uc_energy_limit=max(0.0,euc-eucmin)*3600.0*hess['uc_eta']/dt
            puc[k]=min(uc_request,residual,hess['uc_dis_kW'],uc_energy_limit)
            remaining=residual-puc[k]
            bat_energy_limit=max(0.0,ebat-emin)*3600.0*hess['bat_eta']/dt
            pbat[k]=min(remaining,hess['bat_dis_kW'],bat_energy_limit)
            unmet[k]=max(0.0,remaining-pbat[k])
        else:
            # Regenerative surplus charges UC first, then battery.
            surplus=-residual
            uc_headroom=max(0.0,eucmax-euc)*3600.0/(hess['uc_eta']*dt)
            puc[k]=-min(surplus,hess['uc_chg_kW'],uc_headroom)
            remaining=surplus+puc[k]
            bat_headroom=max(0.0,emax-ebat)*3600.0/(hess['bat_eta']*dt)
            pbat[k]=-min(remaining,hess['bat_chg_kW'],bat_headroom)
            curtailed[k]=max(0.0,remaining+pbat[k])

        # SOC update for the next sample.
        if pbat[k]>=0: ebat-=pbat[k]*dt/3600.0/hess['bat_eta']
        else: ebat+=(-pbat[k])*dt/3600.0*hess['bat_eta']
        if puc[k]>=0: euc-=puc[k]*dt/3600.0/hess['uc_eta']
        else: euc+=(-puc[k])*dt/3600.0*hess['uc_eta']
        ebat=float(np.clip(ebat,emin,emax)); euc=float(np.clip(euc,eucmin,eucmax))
        soc_bat[k]=ebat/bat_cap; soc_uc[k]=euc/max(eucmax,1e-12)

    out['SOC_bat']=soc_bat; out['SOC_uc']=soc_uc
    out['P_bat_ref_kW']=pbat; out['P_uc_ref_kW']=puc; out['P_fc_ref_kW']=pfc
    out['Unmet_Power_kW']=unmet; out['Curtailed_Power_kW']=curtailed
    out['Source_Sum_kW']=pbat+puc+pfc
    out['Power_Balance_Error_kW']=demand-out['Source_Sum_kW']
    out['Scenario_Label']=classify_scenarios(out)
    return out


def apply_manual_scenario_intervals(df: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Apply custom labels to half-open intervals [Start_s, End_s)."""
    out=df.copy()
    if intervals is None or intervals.empty: return out
    for _, row in intervals.iterrows():
        if pd.isna(row.get('Start_s')) and pd.isna(row.get('End_s')) and pd.isna(row.get('Scenario_Label')):
            continue
        if pd.isna(row.get('Start_s')) or pd.isna(row.get('End_s')) or pd.isna(row.get('Scenario_Label')):
            raise ValueError('Complete all three fields for each manual scenario interval.')
        start=float(row['Start_s']); end=float(row['End_s']); label=str(row['Scenario_Label']).strip()
        if end<=start or not label: raise ValueError('Scenario intervals need a label and End_s > Start_s.')
        out.loc[(out.Time_s>=start)&(out.Time_s<end),'Scenario_Label']=label
    return out


def diagnostics_summary(df: pd.DataFrame, hess: dict) -> dict:
    tol=1e-6; bat=df.P_bat_ref_kW; uc=df.P_uc_ref_kW; fc=df.P_fc_ref_kW
    return {
      'max_abs_balance_error_kW':float(df.Power_Balance_Error_kW.abs().max()),
      'mean_abs_balance_error_kW':float(df.Power_Balance_Error_kW.abs().mean()),
      'unmet_samples':int((df.Unmet_Power_kW>tol).sum()),
      'unmet_energy_kWh':float(df.Unmet_Power_kW.clip(lower=0).sum()/3600),
      'curtailed_samples':int((df.Curtailed_Power_kW>tol).sum()),
      'curtailed_energy_kWh':float(df.Curtailed_Power_kW.clip(lower=0).sum()/3600),
      'battery_power_violations':int(((bat>hess['bat_dis_kW']+tol)|(bat< -hess['bat_chg_kW']-tol)).sum()),
      'uc_power_violations':int(((uc>hess['uc_dis_kW']+tol)|(uc< -hess['uc_chg_kW']-tol)).sum()),
      'fc_power_violations':int(((fc<hess['fc_min_kW']-tol)|(fc>hess['fc_max_kW']+tol)).sum()),
      'battery_soc_violations':int(((df.SOC_bat<hess['bat_soc_min']-tol)|(df.SOC_bat>hess['bat_soc_max']+tol)).sum()),
      'uc_soc_violations':int(((df.SOC_uc< -tol)|(df.SOC_uc>1+tol)).sum())}

class ScaledANN:
    def __init__(self, hidden=(32,16), epochs=500, learning_rate=.001, batch_size=64, seed=42, activation='relu'):
        self.xs=StandardScaler(); self.ys=StandardScaler()
        self.net=MLPRegressor(hidden_layer_sizes=hidden,activation=activation,solver='adam',
            learning_rate_init=learning_rate,max_iter=epochs,early_stopping=True,
            n_iter_no_change=25,random_state=seed,batch_size=batch_size)
    def fit(self,x,y):
        self.net.fit(self.xs.fit_transform(x),self.ys.fit_transform(y)); return self
    def predict(self,x): return self.ys.inverse_transform(self.net.predict(self.xs.transform(x)))

def train_and_evaluate(df: pd.DataFrame, hidden=(32,16), epochs=500, learning_rate=.001, batch_size=64, seed=42, activation='relu', test_fraction=.2):
    clean=df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+TARGETS).copy()
    group_col='Cycle_Family' if 'Cycle_Family' in clean else 'Cycle_Name'
    groups=clean[group_col].drop_duplicates().tolist() if group_col in clean else []
    if len(groups)>=2:
        test_group=groups[-1]
        test=clean[clean[group_col]==test_group].copy()
        trainval=clean[clean[group_col]!=test_group].copy()
        split_kind='held-out cycle family'
    else:
        # With only one cycle family, hold out a complete initial-state run when possible.
        run_groups=clean['Cycle_Name'].drop_duplicates().tolist() if 'Cycle_Name' in clean else []
        if len(run_groups)>=2:
            test_group=run_groups[-1]; test=clean[clean.Cycle_Name==test_group].copy()
            trainval=clean[clean.Cycle_Name!=test_group].copy(); split_kind='held-out SOC run (only one cycle family supplied)'
        else:
            raise ValueError('Generate at least two distinct cycle families (recommended) or two separate initial-SOC runs before training. A random row split would risk temporal leakage.')
    if len(test)<5 or len(trainval)<30: raise ValueError('Not enough samples for training and testing. Generate longer cycles or more cycle families.')

    # Validation is a complete run from the training families, not random rows.
    if 'Cycle_Name' in trainval and trainval.Cycle_Name.nunique()>=2:
        val_name=trainval.Cycle_Name.drop_duplicates().iloc[-1]
        val=trainval[trainval.Cycle_Name==val_name].copy()
        train=trainval[trainval.Cycle_Name!=val_name].copy()
    else:
        # A single run is split chronologically; training remains earlier than validation.
        cut=max(1,int(len(trainval)*.85)); train=trainval.iloc[:cut].copy(); val=trainval.iloc[cut:].copy()
    if len(train)<20 or len(val)<2: raise ValueError('Insufficient training/validation samples after group splitting. Add more SOC runs or cycle families.')

    model=ScaledANN(hidden,epochs,learning_rate,batch_size,seed,activation).fit(train[FEATURES],train[TARGETS])
    pred=model.predict(test[FEATURES]); true=test[TARGETS].to_numpy()
    metrics=[]
    for j,name in enumerate(TARGETS):
        metrics.append({'Output':name,'MAE_kW':mean_absolute_error(true[:,j],pred[:,j]),
                        'RMSE_kW':mean_squared_error(true[:,j],pred[:,j])**.5,'R2':r2_score(true[:,j],pred[:,j])})
    pred_df=pd.DataFrame(pred,columns=[x.replace('_ref','_pred') for x in TARGETS],index=test.index)
    result=pd.concat([test.reset_index(drop=True),pred_df.reset_index(drop=True)],axis=1)
    result['Predicted_Sum_kW']=pred.sum(axis=1)
    result['ANN_Power_Balance_Error_kW']=result.P_dem_kW-result.Predicted_Sum_kW
    result['Teacher_Power_Balance_Error_kW']=result.P_dem_kW-result[TARGETS].sum(axis=1)
    return model,pd.DataFrame(metrics),result,{'train':len(train),'validation':len(val),'test':len(test),
        'held_out_cycle':str(test_group),'split_kind':split_kind,'validation_run':str(val.Cycle_Name.iloc[0]) if 'Cycle_Name' in val else 'chronological segment',
        'epochs_run':int(model.net.n_iter_)}

