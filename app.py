import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from backend.core import (
    DEFAULT_VEH, DEFAULT_HESS, DEFAULT_EMU, synthetic_cycle, load_uploaded_cycle,
    calculate_demand, simulate_rule_emu, apply_manual_scenario_intervals,
    diagnostics_summary, train_and_evaluate, FEATURES, TARGETS,
)

st.set_page_config(page_title='HESS EMU Studio', page_icon='', layout='wide')
st.title('HESS Energy Management Studio')
st.caption('Drive-cycle generation · Multi-SOC experiments · Four-input-compatible EMU teacher · ANN training')
st.info('Built-in cycles are synthetic demonstrations, not official WLTP/UDDS/FTP-75 traces. Upload official time–speed data for standards-based evaluation.')

for key, value in [('dataset', None), ('model', None), ('test_results', None), ('run_config', None)]:
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header('1 · Drive-cycle setup')
    cycle_type = st.selectbox('Cycle source', ['Synthetic · Urban stop-go', 'Synthetic · Highway', 'Synthetic · Aggressive acceleration', 'Synthetic · Mixed', 'Upload CSV'])
    extra_profiles = st.multiselect('Additional training cycles', ['Urban stop-go', 'Highway', 'Aggressive acceleration', 'Mixed'], help='A separate cycle family is held out for testing.')
    duration = st.slider('Synthetic duration (seconds)', 300, 7200, 1800, 300)
    max_speed = st.slider('Maximum speed (km/h)', 30, 160, 90, 5)
    seed = st.number_input('Random seed', 0, 999999, 42)
    uploaded = st.file_uploader('Upload cycle CSV (time_s, speed_kmh)', type=['csv']) if cycle_type == 'Upload CSV' else None

    st.divider(); st.header('2 · Vehicle parameters')
    veh = {}
    for key, label, mn, mx, step in [
        ('mass_kg','Vehicle mass (kg)',500,4000,50), ('area_m2','Frontal area (m²)',1,4,.05),
        ('Cd','Drag coefficient',.15,.6,.01), ('Crr','Rolling resistance',.005,.03,.001),
        ('eta_drivetrain','Drivetrain efficiency',.5,1,.01), ('eta_regen','Regenerative efficiency',.3,1,.01),
        ('aux_W','Auxiliary load (W)',0,3000,50), ('max_motor_W','Motor power limit (W)',10000,250000,1000)]:
        veh[key] = st.number_input(label, float(mn), float(mx), float(DEFAULT_VEH[key]), float(step))
    veh['rho'] = DEFAULT_VEH['rho']

    st.divider(); st.header('3 · HESS component limits')
    hess = {}
    hess['bat_cap_kWh'] = st.number_input('Battery capacity (kWh)', .2, 10., float(DEFAULT_HESS['bat_cap_kWh']), .1)
    hess['bat_soc_min'] = st.slider('Battery minimum SOC', .05, .8, float(DEFAULT_HESS['bat_soc_min']), .01)
    hess['bat_soc_max'] = st.slider('Battery maximum SOC', .3, 1., float(DEFAULT_HESS['bat_soc_max']), .01)
    hess['bat_dis_kW'] = st.number_input('Battery max discharge (kW)', 1., 100., float(DEFAULT_HESS['bat_dis_kW']))
    hess['bat_chg_kW'] = st.number_input('Battery max charge (kW)', 1., 100., float(DEFAULT_HESS['bat_chg_kW']))
    hess['bat_eta'] = st.slider('Battery efficiency', .5, 1., float(DEFAULT_HESS['bat_eta']), .01)
    hess['uc_cap_F'] = st.number_input('Ultracapacitance (F)', 10., 2000., float(DEFAULT_HESS['uc_cap_F']))
    hess['uc_v_min'] = st.number_input('UC minimum voltage (V)', 5., 100., float(DEFAULT_HESS['uc_v_min']))
    hess['uc_v_max'] = st.number_input('UC maximum voltage (V)', 10., 200., float(DEFAULT_HESS['uc_v_max']))
    hess['uc_dis_kW'] = st.number_input('UC max discharge (kW)', .5, 100., float(DEFAULT_HESS['uc_dis_kW']))
    hess['uc_chg_kW'] = st.number_input('UC max charge (kW)', .5, 100., float(DEFAULT_HESS['uc_chg_kW']))
    hess['uc_eta'] = st.slider('UC efficiency', .5, 1., float(DEFAULT_HESS['uc_eta']), .01)
    hess['fc_max_kW'] = st.number_input('Fuel-cell max power (kW)', 1., 100., float(DEFAULT_HESS['fc_max_kW']))
    hess['fc_min_kW'] = st.number_input('Fuel-cell minimum power (kW)', 0., 20., float(DEFAULT_HESS['fc_min_kW']))
    hess['fc_ramp_kWps'] = st.number_input('FC physical ramp limit (kW/s) — downstream only', .1, 20., float(DEFAULT_HESS['fc_ramp_kWps']))
    hess['fc_tau_s'] = st.number_input('FC physical response time (s) — downstream only', .5, 60., float(DEFAULT_HESS['fc_tau_s']))
    hess['bat_tau_s'] = st.number_input('Battery physical response time (s) — downstream only', .1, 20., float(DEFAULT_HESS['bat_tau_s']))

    st.divider(); st.header('4 · Initial SOC experiment')
    multi_soc = st.checkbox('Run multiple initial SOC combinations', True)
    if multi_soc:
        bat_soc_values = st.multiselect('Battery initial SOC values', [.3,.4,.5,.6,.7,.8,.9], default=[.5,.7,.9])
        uc_soc_values = st.multiselect('UC initial energy SOC values', [.25,.4,.55,.7,.85,1.0], default=[.4,.7,.9] if False else [.4,.7,.85])
    else:
        bat_soc_values = [st.slider('Battery initial SOC', .05, 1., .7, .01)]
        uc_soc_values = [st.slider('UC initial energy SOC', .1, 1., .64, .01)]

    st.divider(); st.header('5 · Rule-based EMU')
    emu = {'uc_transient_factor': st.slider('UC transient allocation factor', 0., 1., .8, .05),
           'fc_base_fraction': st.slider('FC base contribution fraction', 0., 1., .35, .05),
           'fc_soc_reference': st.slider('Battery SOC reference for FC support', .2, .9, .65, .01)}
    grade = st.slider('Constant road grade (%)', -10., 10., 0., .5)

st.header('6 · Scenario labeling')
manual_labels = st.checkbox('Enable manual scenario intervals', value=False, help='Intervals use [start, end): the end time is excluded. Uncovered samples keep automatic labels.')
if manual_labels:
    st.caption('Add one row per interval. These intervals are applied to each selected cycle; keep end times within the cycle duration.')
    scenario_table = st.data_editor(pd.DataFrame({'Start_s':pd.Series(dtype='float'), 'End_s':pd.Series(dtype='float'), 'Scenario_Label':pd.Series(dtype='str')}), num_rows='dynamic', use_container_width=True, key='scenario_editor')
else:
    scenario_table = pd.DataFrame(columns=['Start_s','End_s','Scenario_Label'])

st.header('7 · ANN configuration')
c1,c2,c3,c4,c5 = st.columns(5)
with c1: arch_name = st.selectbox('Architecture', ['16–8 baseline','32–16 recommended','64–32–16'])
with c2: epochs = st.slider('Max epochs', 50, 2000, 500, 50)
with c3: lr = st.select_slider('Learning rate', [.0001,.0005,.001,.005,.01], value=.001, format_func=lambda x:f'{x:g}')
with c4: batch = st.select_slider('Batch size', [16,32,64,128,256], value=64)
with c5: activation = st.selectbox('Activation', ['relu','tanh'])
hidden = {'16–8 baseline':(16,8), '32–16 recommended':(32,16), '64–32–16':(64,32,16)}[arch_name]

if st.button('⚙️ Generate multi-run HESS dataset', type='primary', use_container_width=True):
    try:
        if hess['bat_soc_min'] >= hess['bat_soc_max']: raise ValueError('Battery minimum SOC must be below maximum SOC.')
        if hess['uc_v_min'] >= hess['uc_v_max']: raise ValueError('UC minimum voltage must be below maximum voltage.')
        if hess['fc_min_kW'] > hess['fc_max_kW']: raise ValueError('Fuel-cell minimum power cannot exceed its maximum power.')
        if not bat_soc_values or not uc_soc_values: raise ValueError('Select at least one initial SOC value for each storage source.')
        if min(bat_soc_values) < hess['bat_soc_min'] or max(bat_soc_values) > hess['bat_soc_max']:
            raise ValueError('Selected battery initial SOC values must be within the configured battery SOC limits.')
        cycle_specs = []
        if cycle_type == 'Upload CSV':
            if uploaded is None: raise ValueError('Upload a cycle CSV first.')
            cycle_specs.append((uploaded.name, load_uploaded_cycle(uploaded)))
        else:
            selected = cycle_type.replace('Synthetic · ','')
            cycle_specs.append((selected, synthetic_cycle(selected, duration, max_speed, int(seed))))
            for j, name in enumerate(extra_profiles):
                if name != selected: cycle_specs.append((name, synthetic_cycle(name, duration, max_speed, int(seed)+j+1)))
        runs=[]
        for family, cycle in cycle_specs:
            demand = calculate_demand(cycle, veh, grade)
            for b_soc in bat_soc_values:
                for u_soc in uc_soc_values:
                    run_hess = dict(hess)
                    run_hess['bat_soc0'] = float(b_soc)
                    # UC energy SOC is defined relative to its rated maximum energy.
                    min_energy_soc = (hess['uc_v_min']/hess['uc_v_max'])**2
                    if u_soc < min_energy_soc: raise ValueError(f'UC initial SOC {u_soc:.2f} is below the minimum energy SOC implied by Vmin/Vmax ({min_energy_soc:.2f}).')
                    run_hess['uc_v0'] = hess['uc_v_max'] * np.sqrt(float(u_soc))
                    labeled = simulate_rule_emu(demand, run_hess, emu)
                    labeled = apply_manual_scenario_intervals(labeled, scenario_table)
                    labeled['Cycle_Family'] = family
                    labeled['Cycle_Name'] = f'{family}_BatSOC{b_soc:.2f}_UCSOC{u_soc:.2f}'
                    labeled['Initial_SOC_bat'] = float(b_soc)
                    labeled['Initial_SOC_uc'] = float(u_soc)
                    runs.append(labeled)
        st.session_state.dataset = pd.concat(runs, ignore_index=True)
        st.session_state.model = None; st.session_state.test_results = None
        st.session_state.run_config = {'cycles':len(cycle_specs),'soc_runs':len(bat_soc_values)*len(uc_soc_values),'hess':hess}
        st.success(f"Generated {len(st.session_state.dataset):,} samples: {len(cycle_specs)} cycle family/families × {len(bat_soc_values)*len(uc_soc_values)} initial-SOC run(s).")
    except Exception as exc:
        st.error(str(exc))

df = st.session_state.dataset
if df is not None:
    st.header('Dataset and engineering diagnostics')
    diag = diagnostics_summary(df, st.session_state.run_config['hess'])
    m=st.columns(5)
    m[0].metric('Samples',f'{len(df):,}'); m[1].metric('Cycle families',df.Cycle_Family.nunique())
    m[2].metric('Initial-state runs',df[['Initial_SOC_bat','Initial_SOC_uc','Cycle_Family']].drop_duplicates().shape[0])
    m[3].metric('Unmet power samples',f"{diag['unmet_samples']:,}"); m[4].metric('Max |balance error|',f"{diag['max_abs_balance_error_kW']:.3f} kW")
    st.subheader('Constraint and power-balance audit')
    audit = pd.DataFrame([{'Diagnostic':k.replace('_',' '),'Value':v} for k,v in diag.items()])
    st.dataframe(audit, use_container_width=True, hide_index=True)
    run_audit = df.groupby(['Cycle_Family','Cycle_Name']).agg(
        Samples=('Time_s','size'), Max_Abs_Balance_Error_kW=('Power_Balance_Error_kW',lambda x:x.abs().max()),
        Unmet_Samples=('Unmet_Power_kW',lambda x:int((x>1e-6).sum())),
        Unmet_Energy_kWh=('Unmet_Power_kW',lambda x:x.clip(lower=0).sum()/3600),
        Min_Battery_SOC=('SOC_bat','min'), Min_UC_SOC=('SOC_uc','min'), Max_UC_SOC=('SOC_uc','max')).reset_index()
    with st.expander('Per-run diagnostics', expanded=False): st.dataframe(run_audit,use_container_width=True)
    if diag['unmet_samples'] or diag['curtailed_samples']:
        st.warning('Some samples have unmet traction demand or unabsorbed surplus. These are explicitly retained in the dataset; inspect them before using the labels for ANN training.')
    tabs=st.tabs(['Drive cycle & demand','Source allocation & SOC','Scenario analysis','Dataset preview'])
    with tabs[0]:
        fam=st.selectbox('Cycle family to plot',df.Cycle_Family.unique(),key='plot_family')
        view=df[df.Cycle_Family==fam].copy()
        run=view.Cycle_Name.unique()[0]
        view=view[view.Cycle_Name==run]
        fig,ax=plt.subplots(figsize=(10,3)); ax.plot(view.Time_s,view.Speed_kmh); ax.set(xlabel='Time (s)',ylabel='Speed (km/h)'); ax.grid(True); st.pyplot(fig)
        fig,ax=plt.subplots(figsize=(10,3)); ax.plot(view.Time_s,view.P_dem_kW); ax.set(xlabel='Time (s)',ylabel='Demand (kW)'); ax.grid(True); st.pyplot(fig)
    with tabs[1]:
        run_name=st.selectbox('Initial-state run',df.Cycle_Name.unique(),key='allocation_run')
        view=df[df.Cycle_Name==run_name]
        fig,ax=plt.subplots(figsize=(10,3))
        for col in TARGETS: ax.plot(view.Time_s,view[col],label=col.replace('_ref_kW',''))
        ax.set(xlabel='Time (s)',ylabel='Power (kW)'); ax.legend(); ax.grid(True); st.pyplot(fig)
        fig,ax=plt.subplots(figsize=(10,3)); ax.plot(view.Time_s,view.SOC_bat,label='Battery SOC'); ax.plot(view.Time_s,view.SOC_uc,label='UC energy SOC'); ax.set(xlabel='Time (s)',ylabel='SOC'); ax.legend(); ax.grid(True); st.pyplot(fig)
    with tabs[2]:
        st.dataframe(df.groupby(['Cycle_Family','Scenario_Label']).size().rename('Samples').reset_index(),use_container_width=True)
        st.caption('Scenario labels are descriptive metadata. They are not ANN inputs in the current CEP feature definition.')
    with tabs[3]: st.dataframe(df.head(1500),use_container_width=True)
    st.download_button('⬇️ Download labeled HESS dataset CSV',df.to_csv(index=False).encode(),file_name='HESS_labeled_dataset.csv',mime='text/csv')
    st.subheader('Train ANN')
    st.caption('Inputs: [P_dem_kW, SOC_bat, SOC_uc, Delta_P_dem_kW] · Targets: [P_bat_ref_kW, P_uc_ref_kW, P_fc_ref_kW]. Teacher labels are deterministic from these four current inputs; FC/battery actuator dynamics are not included in the teacher labels.')
    if st.button('🧠 Train and evaluate ANN',type='primary'):
        try:
            model,metrics,test,counts=train_and_evaluate(df,hidden,epochs,lr,batch,int(seed),activation)
            st.session_state.model=model; st.session_state.test_results=(metrics,test,counts)
            st.success('ANN training and held-out evaluation complete.')
            st.info('Interpret ANN metrics against the held-out cycle. A good fit to teacher labels indicates imitation, not proof of globally optimal energy management.')
        except Exception as exc: st.error(str(exc))

if st.session_state.test_results is not None:
    metrics,test,counts=st.session_state.test_results
    st.header('ANN held-out evaluation')
    st.write(f"Split: {counts['train']:,} train · {counts['validation']:,} validation · {counts['test']:,} test. {counts['split_kind']}: **{counts['held_out_cycle']}** · Validation run: {counts['validation_run']} · Epochs: {counts['epochs_run']}")
    st.dataframe(metrics,use_container_width=True)
    plot_test=test.iloc[::max(1,len(test)//4000)].copy()
    fig,ax=plt.subplots(figsize=(10,3)); ax.plot(plot_test.P_dem_kW.to_numpy(),label='Demand'); ax.plot(plot_test.Predicted_Sum_kW.to_numpy(),label='ANN predicted source sum'); ax.set(xlabel='Held-out sample',ylabel='Power (kW)'); ax.legend(); ax.grid(True); st.pyplot(fig)
    fig,ax=plt.subplots(figsize=(10,3)); ax.plot(plot_test.ANN_Power_Balance_Error_kW.to_numpy()); ax.axhline(0,linewidth=1); ax.set(xlabel='Held-out sample',ylabel='ANN power balance error (kW)'); ax.grid(True); st.pyplot(fig)
    st.download_button('⬇️ Download ANN test predictions',test.to_csv(index=False).encode(),file_name='ANN_test_predictions.csv',mime='text/csv')
    st.download_button('⬇️ Download ANN metrics',metrics.to_csv(index=False).encode(),file_name='ANN_metrics.csv',mime='text/csv')
    if st.session_state.model is not None:
        st.download_button('⬇️ Download trained model + scalers',pickle.dumps(st.session_state.model),file_name='hess_ann_model.pkl',mime='application/octet-stream')

st.divider(); st.caption('Research prototype. Default component values are illustrative assumptions. Verify ratings, source dynamics, and controller constraints against the intended hardware before claiming physical validation.')
