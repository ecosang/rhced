import pymc3 as pm
import pandas as pd
import numpy as np
from theano import tensor as tt
import pickle
import feather
import re

__all__=['posterior_prediction','posterior_outputs','unit_prediction']

from rhced.training import *
from rhced.data_utils import *

def unit_prediction(unitcode,bldg,start_date,end_date,output_path,new_training=False,force_update=False,n_samples=10000,n_training=60000,n_inference=3,max_values={},time_interval=15):
    
    #time_interval=

    n_days=int(((pd.Timestamp(end_date)-pd.Timestamp(start_date)).total_seconds())/(3600*24)+1)
    input_values_path=output_path.joinpath(f"input_values_{start_date}_{end_date}.pickle")
    prior_values_path=output_path.joinpath(f"prior_values_{start_date}_{end_date}.pickle")
    if not(input_values_path.parents[0].exists()):
        if not(input_values_path.parents[1].exists()):
            if not(input_values_path.parents[2].exists()):
                input_values_path.parents[2].mkdir()
            input_values_path.parents[1].mkdir()
        input_values_path.parents[0].mkdir()


    # read data
    meter_data=pd.read_csv(f"data/{bldg}/{unitcode}/raw_data/meter_data_{unitcode}_{start_date}_{end_date}.csv",parse_dates=['timestamp']) # 
    thermostat_data=pd.read_csv(f"data/{bldg}/{unitcode}/raw_data/thermostat_data_{unitcode}_{start_date}_{end_date}.csv",parse_dates=['timestamp']) # 
    
    if new_training:
        scale_const=get_scale_const_template() # scaling constant
        scale_const['net_max']=max_values['net_max'] #maximum values of 
        scale_const['heat_max']=max_values['heat_max']
        scale_const['cool_max']=max_values['cool_max']
        scale_const['aux_max']=max_values['aux_max']
        scale_const['df_max']=max_values['df_max']
        unit_input=create_unit_input(meter_data=meter_data,thermostat_data=thermostat_data,scale_const=scale_const,time_interval=time_interval)
        input_values=create_input_values(unit_input=unit_input,scale_const=scale_const,training=True)
        last_prior_values=None
    else:
        #update
        last_input_values, last_prior_values,last_input_values_path,last_prior_values_path=get_last_outputs(output_path)
        time_interval=last_input_values['time_delta']
        print(f'Model time_interval from previous training output is {time_interval} minutes. If not please double check.')
        unit_input=create_unit_input(meter_data=meter_data,thermostat_data=thermostat_data,scale_const=last_input_values['scale_const'],time_interval=time_interval,training=False)
        input_values=create_input_values(unit_input=unit_input,scale_const=last_input_values['scale_const'],training=True)

    # model training or prediction
    if new_training:
        print("New training, so new training and export the trained results.")
        input_values,prior_values=training(input_values=input_values,prior_values=None,n_training=n_training,n_inference=n_inference)
        with open(input_values_path.__str__(),"wb") as handle:
            pickle.dump(input_values,handle)
        with open(prior_values_path.__str__(),"wb") as handle:
            pickle.dump(prior_values,handle)

        outputs=posterior_prediction(input_values=input_values,prior_values=prior_values,n_samples=n_samples)

    elif force_update:
        # force update if there is no more recent training results
        if end_date<re.findall('\d\d\d\d-\d\d-\d\d',last_prior_values_path.__str__())[0]:
            # last_prior_value is created with more recent data than current training period.
            # So skip training
            print("Skip update because there is prior_values trained with more recent data.")
            outputs=posterior_prediction(input_values=input_values,prior_values=last_prior_values,n_samples=n_samples)
        else:
            # Force update
            print("Model update and then posterior prediction.")
            input_values,prior_values=training(input_values=input_values,prior_values=last_prior_values,n_training=n_training,n_inference=n_inference)
            outputs=posterior_prediction(input_values=input_values,prior_values=prior_values,n_samples=n_samples)
            
            with open(input_values_path.__str__(),"wb") as handle:
                pickle.dump(input_values,handle)
            with open(prior_values_path.__str__(),"wb") as handle:
                pickle.dump(prior_values,handle)
    else:
        # check if prediction is available.
        try:
            outputs=posterior_prediction(input_values=input_values,prior_values=last_prior_values,n_samples=n_samples)
            print("Posterior prediction only")
        except:
            # fail to prediction only because new opeartion is detected.
            print("Model update and then posterior prediction.")
            input_values,prior_values=training(input_values=input_values,prior_values=last_prior_values,n_training=n_training,n_inference=n_inference)
            outputs=posterior_prediction(input_values=input_values,prior_values=prior_values,n_samples=n_samples)
            with open(input_values_path.__str__(),"wb") as handle:
                pickle.dump(input_values,handle)
            with open(prior_values_path.__str__(),"wb") as handle:
                pickle.dump(prior_values,handle)

    
    df=posterior_outputs(input_values,outputs)
    feather.write_dataframe(df,output_path.joinpath(f'df_{start_date}_{end_date}.feather').__str__())
    print(f"feather file is created at {output_path.joinpath(f'df_{start_date}_{end_date}.feather').__str__()}")
    print(f"lower: {np.nanmean(df['P_hc_lower'])/1000*24*n_days}")
    print(f"median: {np.nanmean(df['P_hc_mid'])/1000*24*n_days}")
    print(f"upper: {np.nanmean(df['P_hc_upper'])/1000*24*n_days}")
    print(f"measurement: {np.nanmean(df['hc']*df['i_hc'])/1000*24*n_days}") 
    # it is true to multiply (i_hc) because the mean of hc already take accounts for time fraction. 
    return df,outputs



def posterior_outputs(input_values,outputs):
    
    nan_index=input_values['nan_index']
    base_=np.zeros([nan_index.shape[0]])
    base_[nan_index]=np.nan
    
    mu_phi_df=(outputs['mu_phi_df'])
    s_T_out=input_values['s_T_out']
    i_hc=base_.copy()
    i_hc[~nan_index]=calculate_i_fraction(input_values['i_hc'],input_values['i_hc'])
    
    i_heat1_all=base_.copy()
    i_heat1=base_.copy()
    i_df1=base_.copy()
    
    i_heat1_all_=calculate_i_fraction(input_values['i_heat1_all'],input_values['i_hc'])
    i_df1_=i_heat1_all_.copy()
    i_heat1_=i_heat1_all_.copy()

    i_df1_[s_T_out>=mu_phi_df]=0
    i_heat1_[s_T_out<mu_phi_df]=0
    
    i_heat1[~nan_index]=i_heat1_
    i_df1[~nan_index]=i_df1_
    i_heat1[~nan_index]=i_heat1_

    # i_heat2=base_.copy()
    # i_heat2[~nan_index]=calculate_i_fraction(input_values['i_heat2'],input_values['i_hc'])
    i_cool1=base_.copy()
    i_cool1[~nan_index]=calculate_i_fraction(input_values['i_cool1'],input_values['i_hc'])
    # i_cool2=base_.copy()
    # i_cool2[~nan_index]=calculate_i_fraction(input_values['i_cool2'],input_values['i_hc'])
    # i_df1=base_.copy()
    # i_df1[~nan_index]=calculate_i_fraction(input_values['i_df1'],input_values['i_hc'])
    # i_df2=base_.copy()
    # i_df2[~nan_index]=calculate_i_fraction(input_values['i_df2'],input_values['i_hc'])
    i_aux1=base_.copy()
    i_aux1[~nan_index]=calculate_i_fraction(input_values['i_aux1'],input_values['i_hc'])
    i_fan1=base_.copy()
    i_fan1[~nan_index]=calculate_i_fraction(input_values['i_fan1'],input_values['i_hc'])

    P_hc=outputs['P_hc']
    P_hc_mid=base_.copy()
    P_hc_lower=base_.copy()
    P_hc_upper=base_.copy()
    P_hc_lq=base_.copy()
    P_hc_uq=base_.copy()
    P_hc_mid[~nan_index]=np.median(P_hc,axis=0)
    P_hc_lower[~nan_index]=np.quantile(P_hc,0.025,axis=0)
    P_hc_lq[~nan_index]=np.quantile(P_hc,0.25,axis=0)
    P_hc_uq[~nan_index]=np.quantile(P_hc,0.75,axis=0)
    P_hc_upper[~nan_index]=np.quantile(P_hc,0.975,axis=0)

    P_heat1=outputs['P_heat1']
    P_heat1_mid=base_.copy()
    P_heat1_lower=base_.copy()
    P_heat1_upper=base_.copy()
    P_heat1_lq=base_.copy()
    P_heat1_uq=base_.copy()
    P_heat1_mid[~nan_index]=np.median(P_heat1,axis=0)
    P_heat1_lower[~nan_index]=np.quantile(P_heat1,0.025,axis=0)
    P_heat1_lq[~nan_index]=np.quantile(P_heat1,0.25,axis=0)
    P_heat1_uq[~nan_index]=np.quantile(P_heat1,0.75,axis=0)
    P_heat1_upper[~nan_index]=np.quantile(P_heat1,0.975,axis=0)

    # P_heat2=outputs['P_heat2']
    # P_heat2_mid=base_.copy()
    # P_heat2_lower=base_.copy()
    # P_heat2_upper=base_.copy()
    # P_heat2_lq=base_.copy()
    # P_heat2_uq=base_.copy()
    # P_heat2_mid[~nan_index]=np.median(P_heat2,axis=0)
    # P_heat2_lower[~nan_index]=np.quantile(P_heat2,0.025,axis=0)
    # P_heat2_lq[~nan_index]=np.quantile(P_heat2,0.25,axis=0)
    # P_heat2_uq[~nan_index]=np.quantile(P_heat2,0.75,axis=0)
    # P_heat2_upper[~nan_index]=np.quantile(P_heat2,0.975,axis=0)

    P_cool1=outputs['P_cool1']
    P_cool1_mid=base_.copy()
    P_cool1_lower=base_.copy()
    P_cool1_upper=base_.copy()
    P_cool1_lq=base_.copy()
    P_cool1_uq=base_.copy()
    P_cool1_mid[~nan_index]=np.median(P_cool1,axis=0)
    P_cool1_lower[~nan_index]=np.quantile(P_cool1,0.025,axis=0)
    P_cool1_lq[~nan_index]=np.quantile(P_cool1,0.25,axis=0)
    P_cool1_uq[~nan_index]=np.quantile(P_cool1,0.75,axis=0)
    P_cool1_upper[~nan_index]=np.quantile(P_cool1,0.975,axis=0)

    # P_cool2=outputs['P_cool2']
    # P_cool2_mid=base_.copy()
    # P_cool2_lower=base_.copy()
    # P_cool2_upper=base_.copy()
    # P_cool2_lq=base_.copy()
    # P_cool2_uq=base_.copy()
    # P_cool2_mid[~nan_index]=np.median(P_cool2,axis=0)
    # P_cool2_lower[~nan_index]=np.quantile(P_cool2,0.025,axis=0)
    # P_cool2_lq[~nan_index]=np.quantile(P_cool2,0.25,axis=0)
    # P_cool2_uq[~nan_index]=np.quantile(P_cool2,0.75,axis=0)
    # P_cool2_upper[~nan_index]=np.quantile(P_cool2,0.975,axis=0)

    P_df1=outputs['P_df1']
    P_df1_mid=base_.copy()
    P_df1_lower=base_.copy()
    P_df1_upper=base_.copy()
    P_df1_lq=base_.copy()
    P_df1_uq=base_.copy()
    P_df1_mid[~nan_index]=np.median(P_df1,axis=0)
    P_df1_lower[~nan_index]=np.quantile(P_df1,0.025,axis=0)
    P_df1_lq[~nan_index]=np.quantile(P_df1,0.25,axis=0)
    P_df1_uq[~nan_index]=np.quantile(P_df1,0.75,axis=0)
    P_df1_upper[~nan_index]=np.quantile(P_df1,0.975,axis=0)

    # P_df2=outputs['P_df2']
    # P_df2_mid=base_.copy()
    # P_df2_lower=base_.copy()
    # P_df2_upper=base_.copy()
    # P_df2_lq=base_.copy()
    # P_df2_uq=base_.copy()
    # P_df2_mid[~nan_index]=np.median(P_df2,axis=0)
    # P_df2_lower[~nan_index]=np.quantile(P_df2,0.025,axis=0)
    # P_df2_lq[~nan_index]=np.quantile(P_df2,0.25,axis=0)
    # P_df2_uq[~nan_index]=np.quantile(P_df2,0.75,axis=0)
    # P_df2_upper[~nan_index]=np.quantile(P_df2,0.975,axis=0)

    P_aux1=outputs['P_aux1']
    P_aux1_mid=base_.copy()
    P_aux1_lower=base_.copy()
    P_aux1_upper=base_.copy()
    P_aux1_lq=base_.copy()
    P_aux1_uq=base_.copy()
    P_aux1_mid[~nan_index]=np.median(P_aux1,axis=0)
    P_aux1_lower[~nan_index]=np.quantile(P_aux1,0.025,axis=0)
    P_aux1_lq[~nan_index]=np.quantile(P_aux1,0.25,axis=0)
    P_aux1_uq[~nan_index]=np.quantile(P_aux1,0.75,axis=0)

    P_aux1_upper[~nan_index]=np.quantile(P_aux1,0.975,axis=0)

    P_fan1=outputs['P_fan1']
    P_fan1_mid=base_.copy()
    P_fan1_lower=base_.copy()
    P_fan1_upper=base_.copy()
    P_fan1_lq=base_.copy()
    P_fan1_uq=base_.copy()
    P_fan1_mid[~nan_index]=np.median(P_fan1,axis=0)
    P_fan1_lower[~nan_index]=np.quantile(P_fan1,0.025,axis=0)
    P_fan1_lq[~nan_index]=np.quantile(P_fan1,0.25,axis=0)
    P_fan1_uq[~nan_index]=np.quantile(P_fan1,0.75,axis=0)
    P_fan1_upper[~nan_index]=np.quantile(P_fan1,0.975,axis=0)

    df=pd.DataFrame(data={"timestamp":input_values['timestamp']})
    df['timestamp']=df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df['i_hc']=i_hc
    df['i_heat1_all']=i_heat1_all
    #df['i_heat2']=i_heat2
    df['i_cool1']=i_cool1
    #df['i_cool2']=i_cool2
    df['i_aux1']=i_aux1
    df['i_df1']=i_df1
    df['i_heat1']=i_heat1
    df['i_fan1']=i_fan1
    df['i_hc']=i_hc

    df['P_hc_lower']=P_hc_lower
    df['P_hc_mid']=P_hc_mid
    df['P_hc_lq']=P_hc_lq
    df['P_hc_uq']=P_hc_uq
    df['P_hc_upper']=P_hc_upper

    df['P_heat1_lower']=P_heat1_lower
    df['P_heat1_mid']=P_heat1_mid
    df['P_heat1_upper']=P_heat1_upper
    df['P_heat1_lq']=P_heat1_lq
    df['P_heat1_uq']=P_heat1_uq

    # df['P_heat2_lower']=P_heat2_lower
    # df['P_heat2_mid']=P_heat2_mid
    # df['P_heat2_upper']=P_heat2_upper
    # df['P_heat2_lq']=P_heat2_lq
    # df['P_heat2_uq']=P_heat2_uq

    df['P_df1_lower']=P_df1_lower
    df['P_df1_mid']=P_df1_mid
    df['P_df1_upper']=P_df1_upper
    df['P_df1_lq']=P_df1_lq
    df['P_df1_uq']=P_df1_uq

    # df['P_df2_lower']=P_df2_lower
    # df['P_df2_mid']=P_df2_mid
    # df['P_df2_upper']=P_df2_upper
    # df['P_df2_lq']=P_df2_lq
    # df['P_df2_uq']=P_df2_uq

    df['P_cool1_lower']=P_cool1_lower
    df['P_cool1_mid']=P_cool1_mid
    df['P_cool1_upper']=P_cool1_upper
    df['P_cool1_uq']=P_cool1_uq
    df['P_cool1_lq']=P_cool1_lq

    # df['P_cool2_lower']=P_cool2_lower
    # df['P_cool2_mid']=P_cool2_mid
    # df['P_cool2_upper']=P_cool2_upper
    # df['P_cool2_uq']=P_cool2_uq
    # df['P_cool2_lq']=P_cool2_lq

    df['P_aux1_lower']=P_aux1_lower
    df['P_aux1_mid']=P_aux1_mid
    df['P_aux1_upper']=P_aux1_upper
    df['P_aux1_uq']=P_aux1_uq
    df['P_aux1_lq']=P_aux1_lq

    df['P_fan1_lower']=P_fan1_lower
    df['P_fan1_mid']=P_fan1_mid
    df['P_fan1_upper']=P_fan1_upper
    df['P_fan1_uq']=P_fan1_uq
    df['P_fan1_lq']=P_fan1_lq
    
    if 'hc' in input_values.keys():
        hc=base_.copy()
        hc[~nan_index]=input_values['hc']
        df['hc']=hc

    return df



def posterior_prediction(input_values,prior_values=None,n_samples=10000):
    # from the parameters, predict results
    
    i_heat1_all=input_values['i_heat1_all']
    i_heat2_all=input_values['i_heat2_all']
    
    i_cool1=input_values['i_cool1']
    i_cool2=input_values['i_cool2']
    
    i_aux1=input_values['i_aux1']
    i_fan1=input_values['i_fan1']

    n_i_heat1_all=np.nansum(i_heat1_all>0).astype("int")
    n_i_heat2_all=np.nansum(i_heat1_all>0).astype("int")
    
    n_i_cool1=np.nansum(i_cool1>0).astype("int")
    n_i_cool2=np.nansum(i_cool2>0).astype("int")
    n_i_aux1=np.nansum(i_aux1>0).astype("int")
    n_i_fan1=np.nansum(i_fan1>0).astype("int")

    
    heat_max=input_values['scale_const']['heat_max']
    cool_max=input_values['scale_const']['cool_max']
    df_max=input_values['scale_const']['df_max']
    aux_max=input_values['scale_const']['aux_max']
    net_max=input_values['scale_const']['net_max']

    scale_const=input_values['scale_const']
    s_net=input_values['s_net']
    s_T_out=input_values['s_T_out']
    s_T_in=input_values['s_T_in']
    s_wb_in=input_values['s_wb_in']
    mu_phi_df=prior_values['mu_phi_df']
    n_N=s_T_out.shape[0]
    
    if ('mu_beta0_heat1' in prior_values.keys()) and (n_i_heat1_all>0):
        # mu_beta0_heat1, sigma_beta0_heat1
        # mu_beta1_heat1_, sigma_beta1_heat1_
        # beta0_heat1=np.random.normal(loc=prior_values['mu_beta0_heat1'],scale=prior_values['sigma_beta0_heat1'],size=n_samples)
        # beta1_heat1=npsoftplus(np.random.normal(loc=prior_values['mu_beta1_heat1_'],scale=prior_values['sigma_beta1_heat1_'],size=n_samples))
        beta0_heat1=np.random.normal(loc=prior_values['mu_beta0_heat1'],scale=prior_values['sigma_beta0_heat1'],size=n_samples)
        beta1_heat1=np.random.normal(loc=prior_values['mu_beta1_heat1'],scale=prior_values['sigma_beta1_heat1'],size=n_samples)
        beta2_heat1=np.random.normal(loc=prior_values['mu_beta2_heat1'],scale=prior_values['sigma_beta2_heat1'],size=n_samples)
        beta3_heat1=np.random.normal(loc=prior_values['mu_beta3_heat1'],scale=prior_values['sigma_beta3_heat1'],size=n_samples)
        beta4_heat1=np.random.normal(loc=prior_values['mu_beta4_heat1'],scale=prior_values['sigma_beta4_heat1'],size=n_samples)
        beta5_heat1=np.random.normal(loc=prior_values['mu_beta5_heat1'],scale=prior_values['sigma_beta5_heat1'],size=n_samples)

        beta6_heat1=np.random.normal(loc=prior_values['mu_beta6_heat1'],scale=prior_values['sigma_beta6_heat1'],size=n_samples)
        beta7_heat1=np.random.normal(loc=prior_values['mu_beta7_heat1'],scale=prior_values['sigma_beta7_heat1'],size=n_samples)
        beta8_heat1=np.random.normal(loc=prior_values['mu_beta8_heat1'],scale=prior_values['sigma_beta8_heat1'],size=n_samples)
        beta9_heat1=np.random.normal(loc=prior_values['mu_beta9_heat1'],scale=prior_values['sigma_beta9_heat1'],size=n_samples)
        beta10_heat1=np.random.normal(loc=prior_values['mu_beta10_heat1'],scale=prior_values['sigma_beta10_heat1'],size=n_samples)

        #mu_sigma_P_heat1=prior_values['mu_sigma_P_heat1_']
        sigma_sigma_P_heat1=prior_values['sigma_sigma_P_heat1_']
        sigma_P_heat1=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_heat1_'],scale=prior_values['sigma_sigma_P_heat1_'],size=n_N)) #.reshape([n_samples,n_N])
        epsilon_heat1=np.random.normal(loc=0,scale=sigma_P_heat1,size=(n_samples,n_N)) # sigma_P_heat1 n_N size, output is n_samples x n_N
        
        # epsilon_heat1=np.random.normal(loc=0,scale=prior_values['sigma_P_heat1_'],size=n_samples*n_N).reshape([n_samples,n_N])
        # P_heat1=npsigmoid(beta0_heat1[:,None]+np.dot(beta1_heat1[:,None],(s_T_out[None,:]))+epsilon_heat1)*heat_max/net_max # n_samples,n_N

        P_heat1=npsigmoid(beta0_heat1[:,None]+\
                            np.dot(beta1_heat1[:,None],(s_T_in[None,:]))+\
                            np.dot(beta2_heat1[:,None],(s_T_in[None,:]**2))+\
                            np.dot(beta3_heat1[:,None],(s_T_in[None,:]**3))+\
                            np.dot(beta4_heat1[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_heat1[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_heat1[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_heat1[:,None],(s_T_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_heat1[:,None],(s_T_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_heat1[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_heat1[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_heat1)*heat_max/net_max # n_samples,n_N

        # mu_P_heat1=beta0_heat1[:,None]+np.dot(beta1_heat1[:,None],s_T_out[None,:]) # n_samples,n_N
        # sigma_P_heat1=prior_values['sigma_P_heat1_']
        # concentration_alpha_heat1,concentration_beta_heat1=calculate_concentration(mu_P_heat1,sigma_P_heat1)
        # P_heat1=np.random.beta(a=concentration_alpha_heat1,b=concentration_beta_heat1)
        
        #'mu_phi_df': array([0.02628323]),
        #'sigma_phi_df': array([0.06657257]),
        phi_df=np.random.normal(loc=prior_values['mu_phi_df'],scale=prior_values['sigma_phi_df'],size=n_samples)

        beta0_df1=np.random.normal(loc=prior_values['mu_beta0_df1'],scale=prior_values['sigma_beta0_df1'],size=n_samples)
        beta1_df1=np.random.normal(loc=prior_values['mu_beta1_df1'],scale=prior_values['sigma_beta1_df1'],size=n_samples)
        beta2_df1=np.random.normal(loc=prior_values['mu_beta2_df1'],scale=prior_values['sigma_beta2_df1'],size=n_samples)
        beta3_df1=np.random.normal(loc=prior_values['mu_beta3_df1'],scale=prior_values['sigma_beta3_df1'],size=n_samples)
        beta4_df1=np.random.normal(loc=prior_values['mu_beta4_df1'],scale=prior_values['sigma_beta4_df1'],size=n_samples)
        beta5_df1=np.random.normal(loc=prior_values['mu_beta5_df1'],scale=prior_values['sigma_beta5_df1'],size=n_samples)

        beta6_df1=np.random.normal(loc=prior_values['mu_beta6_df1'],scale=prior_values['sigma_beta6_df1'],size=n_samples)
        beta7_df1=np.random.normal(loc=prior_values['mu_beta7_df1'],scale=prior_values['sigma_beta7_df1'],size=n_samples)
        beta8_df1=np.random.normal(loc=prior_values['mu_beta8_df1'],scale=prior_values['sigma_beta8_df1'],size=n_samples)
        beta9_df1=np.random.normal(loc=prior_values['mu_beta9_df1'],scale=prior_values['sigma_beta9_df1'],size=n_samples)
        beta10_df1=np.random.normal(loc=prior_values['mu_beta10_df1'],scale=prior_values['sigma_beta10_df1'],size=n_samples)
        
        i_df1=np.zeros([n_samples,n_N])
        i_heat1=np.zeros([n_samples,n_N])
        for ix in np.arange(n_N):
            i_df1[:,ix]=np.round(npreverse_logistic(x=s_T_out[ix],x0=phi_df),0)*input_values['i_heat1_all'][ix]
            i_heat1[:,ix]=np.abs(np.round(npreverse_logistic(x=s_T_out[ix],x0=phi_df),0)-1)*input_values['i_heat1_all'][ix]
            #i_heat1_ndf[:,ix]=nplogistic(x=s_T_out[ix],x0=phi_df)*input_values['i_heat1'][ix]
        

        sigma_P_df1=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_df1_'],scale=prior_values['sigma_sigma_P_df1_'],size=n_N)) 
        epsilon_df1=np.random.normal(loc=0,scale=sigma_P_df1,size=(n_samples,n_N))
        #epsilon_df1=np.random.normal(loc=0,scale=prior_values['sigma_P_df1_'],size=n_samples*n_N).reshape([n_samples,n_N])
        #P_df1=npsigmoid(beta0_df1[:,None]+np.dot(-1.*beta1_df1[:,None],s_T_out[None,:])+epsilon_df1)*df_max/net_max
        P_df1=npsigmoid(beta0_df1[:,None]+\
                            np.dot(beta1_df1[:,None],(s_T_in[None,:]))+\
                            np.dot(beta2_df1[:,None],(s_T_in[None,:]**2))+\
                            np.dot(beta3_df1[:,None],(s_T_in[None,:]**3))+\
                            np.dot(beta4_df1[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_df1[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_df1[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_df1[:,None],(s_T_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_df1[:,None],(s_T_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_df1[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_df1[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_df1)*df_max/net_max # n_samples,n_N

    elif (np.nansum(i_heat1_all)>0):
        raise ValueError("heat1 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_heat1=np.zeros([n_samples,n_N])
        P_df1=np.zeros([n_samples,n_N])
        i_df1=np.zeros([n_samples,n_N])
        i_heat1=np.zeros([n_samples,n_N])

    if ('mu_beta0_heat2' in prior_values.keys()) and (n_i_heat2_all>0):
        # mu_beta0_heat2, sigma_beta0_heat2
        # mu_beta1_heat2_, sigma_beta1_heat2_
        # beta0_heat2=np.random.normal(loc=prior_values['mu_beta0_heat2'],scale=prior_values['sigma_beta0_heat2'],size=n_samples)
        # beta1_heat2=npsoftplus(np.random.normal(loc=prior_values['mu_beta1_heat2_'],scale=prior_values['sigma_beta1_heat2_'],size=n_samples))
        beta0_heat2=np.random.normal(loc=prior_values['mu_beta0_heat2'],scale=prior_values['sigma_beta0_heat2'],size=n_samples)
        beta1_heat2=np.random.normal(loc=prior_values['mu_beta1_heat2'],scale=prior_values['sigma_beta1_heat2'],size=n_samples)
        beta2_heat2=np.random.normal(loc=prior_values['mu_beta2_heat2'],scale=prior_values['sigma_beta2_heat2'],size=n_samples)
        beta3_heat2=np.random.normal(loc=prior_values['mu_beta3_heat2'],scale=prior_values['sigma_beta3_heat2'],size=n_samples)
        beta4_heat2=np.random.normal(loc=prior_values['mu_beta4_heat2'],scale=prior_values['sigma_beta4_heat2'],size=n_samples)
        beta5_heat2=np.random.normal(loc=prior_values['mu_beta5_heat2'],scale=prior_values['sigma_beta5_heat2'],size=n_samples)

        beta6_heat2=np.random.normal(loc=prior_values['mu_beta6_heat2'],scale=prior_values['sigma_beta6_heat2'],size=n_samples)
        beta7_heat2=np.random.normal(loc=prior_values['mu_beta7_heat2'],scale=prior_values['sigma_beta7_heat2'],size=n_samples)
        beta8_heat2=np.random.normal(loc=prior_values['mu_beta8_heat2'],scale=prior_values['sigma_beta8_heat2'],size=n_samples)
        beta9_heat2=np.random.normal(loc=prior_values['mu_beta9_heat2'],scale=prior_values['sigma_beta9_heat2'],size=n_samples)
        beta10_heat2=np.random.normal(loc=prior_values['mu_beta10_heat2'],scale=prior_values['sigma_beta10_heat2'],size=n_samples)

        #mu_sigma_P_heat2=prior_values['mu_sigma_P_heat2_']
        sigma_sigma_P_heat2=prior_values['sigma_sigma_P_heat2_']
        sigma_P_heat2=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_heat2_'],scale=prior_values['sigma_sigma_P_heat2_'],size=n_N)) #.reshape([n_samples,n_N])
        epsilon_heat2=np.random.normal(loc=0,scale=sigma_P_heat2,size=(n_samples,n_N)) # sigma_P_heat2 n_N size, output is n_samples x n_N
        
        # epsilon_heat2=np.random.normal(loc=0,scale=prior_values['sigma_P_heat2_'],size=n_samples*n_N).reshape([n_samples,n_N])
        # P_heat2=npsigmoid(beta0_heat2[:,None]+np.dot(beta1_heat2[:,None],(s_T_out[None,:]))+epsilon_heat2)*heat_max/net_max # n_samples,n_N

        P_heat2=npsigmoid(beta0_heat2[:,None]+\
                            np.dot(beta1_heat2[:,None],(s_T_in[None,:]))+\
                            np.dot(beta2_heat2[:,None],(s_T_in[None,:]**2))+\
                            np.dot(beta3_heat2[:,None],(s_T_in[None,:]**3))+\
                            np.dot(beta4_heat2[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_heat2[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_heat2[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_heat2[:,None],(s_T_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_heat2[:,None],(s_T_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_heat2[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_heat2[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_heat2)*heat_max/net_max # n_samples,n_N

        
        #'mu_phi_df': array([0.02628323]),
        #'sigma_phi_df': array([0.06657257]),
        phi_df=np.random.normal(loc=prior_values['mu_phi_df'],scale=prior_values['sigma_phi_df'],size=n_samples)

        #beta0_df2=np.random.normal(loc=prior_values['mu_beta0_df2'],scale=prior_values['sigma_beta0_df2'],size=n_samples)
        #beta1_df2=npsoftplus(np.random.normal(loc=prior_values['mu_beta1_df2_'],scale=prior_values['sigma_beta1_df2_'],size=n_samples))
        
        beta0_df2=np.random.normal(loc=prior_values['mu_beta0_df2'],scale=prior_values['sigma_beta0_df2'],size=n_samples)
        beta1_df2=np.random.normal(loc=prior_values['mu_beta1_df2'],scale=prior_values['sigma_beta1_df2'],size=n_samples)
        beta2_df2=np.random.normal(loc=prior_values['mu_beta2_df2'],scale=prior_values['sigma_beta2_df2'],size=n_samples)
        beta3_df2=np.random.normal(loc=prior_values['mu_beta3_df2'],scale=prior_values['sigma_beta3_df2'],size=n_samples)
        beta4_df2=np.random.normal(loc=prior_values['mu_beta4_df2'],scale=prior_values['sigma_beta4_df2'],size=n_samples)
        beta5_df2=np.random.normal(loc=prior_values['mu_beta5_df2'],scale=prior_values['sigma_beta5_df2'],size=n_samples)

        beta6_df2=np.random.normal(loc=prior_values['mu_beta6_df2'],scale=prior_values['sigma_beta6_df2'],size=n_samples)
        beta7_df2=np.random.normal(loc=prior_values['mu_beta7_df2'],scale=prior_values['sigma_beta7_df2'],size=n_samples)
        beta8_df2=np.random.normal(loc=prior_values['mu_beta8_df2'],scale=prior_values['sigma_beta8_df2'],size=n_samples)
        beta9_df2=np.random.normal(loc=prior_values['mu_beta9_df2'],scale=prior_values['sigma_beta9_df2'],size=n_samples)
        beta10_df2=np.random.normal(loc=prior_values['mu_beta10_df2'],scale=prior_values['sigma_beta10_df2'],size=n_samples)
        
        i_df2=np.zeros([n_samples,n_N])
        i_heat2=np.zeros([n_samples,n_N])
        for ix in np.arange(n_N):
            i_df2[:,ix]=np.round(npreverse_logistic(x=s_T_out[ix],x0=phi_df),0)*input_values['i_heat2_all'][ix]
            i_heat2[:,ix]=np.abs(np.round(npreverse_logistic(x=s_T_out[ix],x0=phi_df),0)-1)*input_values['i_heat2_all'][ix]

        

        sigma_P_df2=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_df2_'],scale=prior_values['sigma_sigma_P_df2_'],size=n_N)) 
        epsilon_df2=np.random.normal(loc=0,scale=sigma_P_df2,size=(n_samples,n_N))
        #epsilon_df2=np.random.normal(loc=0,scale=prior_values['sigma_P_df2_'],size=n_samples*n_N).reshape([n_samples,n_N])
        #P_df2=npsigmoid(beta0_df2[:,None]+np.dot(-1.*beta1_df2[:,None],s_T_out[None,:])+epsilon_df2)*df_max/net_max
        P_df2=npsigmoid(beta0_df2[:,None]+\
                            np.dot(beta1_df2[:,None],(s_T_in[None,:]))+\
                            np.dot(beta2_df2[:,None],(s_T_in[None,:]**2))+\
                            np.dot(beta3_df2[:,None],(s_T_in[None,:]**3))+\
                            np.dot(beta4_df2[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_df2[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_df2[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_df2[:,None],(s_T_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_df2[:,None],(s_T_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_df2[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_df2[:,None],(s_T_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_df2)*df_max/net_max # n_samples,n_N

    elif (np.nansum(i_heat2_all)>0):
        raise ValueError("heat2 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_heat2=np.zeros([n_samples,n_N])
        P_df2=np.zeros([n_samples,n_N])
        i_df2=np.zeros([n_samples,n_N])
        i_heat2=np.zeros([n_samples,n_N])


    if ('mu_beta0_cool1' in prior_values.keys()) and (n_i_cool1>0):
        
        beta0_cool1=np.random.normal(loc=prior_values['mu_beta0_cool1'],scale=prior_values['sigma_beta0_cool1'],size=n_samples)
        beta1_cool1=np.random.normal(loc=prior_values['mu_beta1_cool1'],scale=prior_values['sigma_beta1_cool1'],size=n_samples)
        beta2_cool1=np.random.normal(loc=prior_values['mu_beta2_cool1'],scale=prior_values['sigma_beta2_cool1'],size=n_samples)
        beta3_cool1=np.random.normal(loc=prior_values['mu_beta3_cool1'],scale=prior_values['sigma_beta3_cool1'],size=n_samples)
        beta4_cool1=np.random.normal(loc=prior_values['mu_beta4_cool1'],scale=prior_values['sigma_beta4_cool1'],size=n_samples)
        beta5_cool1=np.random.normal(loc=prior_values['mu_beta5_cool1'],scale=prior_values['sigma_beta5_cool1'],size=n_samples)

        beta6_cool1=np.random.normal(loc=prior_values['mu_beta6_cool1'],scale=prior_values['sigma_beta6_cool1'],size=n_samples)
        beta7_cool1=np.random.normal(loc=prior_values['mu_beta7_cool1'],scale=prior_values['sigma_beta7_cool1'],size=n_samples)
        beta8_cool1=np.random.normal(loc=prior_values['mu_beta8_cool1'],scale=prior_values['sigma_beta8_cool1'],size=n_samples)
        beta9_cool1=np.random.normal(loc=prior_values['mu_beta9_cool1'],scale=prior_values['sigma_beta9_cool1'],size=n_samples)
        beta10_cool1=np.random.normal(loc=prior_values['mu_beta10_cool1'],scale=prior_values['sigma_beta10_cool1'],size=n_samples)


        sigma_P_cool1=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_cool1_'],scale=prior_values['sigma_sigma_P_cool1_'],size=n_N)) 
        epsilon_cool1=np.random.normal(loc=0,scale=sigma_P_cool1,size=(n_samples,n_N))
        #epsilon_cool1=np.random.normal(loc=0,scale=prior_values['sigma_P_cool1_'],size=n_samples*n_N).reshape([n_samples,n_N])
        #P_cool1=npsigmoid(beta0_cool1[:,None]+np.dot(beta1_cool1[:,None],s_T_out[None,:])+epsilon_cool1)*cool_max/net_max
        
        P_cool1=npsigmoid(beta0_cool1[:,None]+\
                            np.dot(beta1_cool1[:,None],(s_wb_in[None,:]))+\
                            np.dot(beta2_cool1[:,None],(s_wb_in[None,:]**2))+\
                            np.dot(beta3_cool1[:,None],(s_wb_in[None,:]**3))+\
                            np.dot(beta4_cool1[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_cool1[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_cool1[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_cool1[:,None],(s_wb_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_cool1[:,None],(s_wb_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_cool1[:,None],(s_wb_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_cool1[:,None],(s_wb_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_cool1)*cool_max/net_max # n_samples,n_N

        #mu_P_cool1=beta0_cool1[:,None]+np.dot(beta1_cool1[:,None],s_T_out[None,:])
        #sigma_P_cool1=prior_values['sigma_P_cool1_']
        #concentration_alpha_cool1,concentration_beta_cool1=calculate_concentration(mu_P_cool1,sigma_P_cool1)
        #P_cool1=np.random.beta(a=concentration_alpha_cool1,b=concentration_beta_cool1)

    elif (np.nansum(i_cool1)>0):
        raise ValueError("cool1 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_cool1=np.zeros([n_samples,n_N])

    if ('mu_beta0_cool2' in prior_values.keys()) and (n_i_cool2>0):
        # mu_beta0_cool2, sigma_beta0_cool2
        # mu_beta1_cool2_, sigma_beta1_cool2_
        #beta0_cool2=np.random.normal(loc=prior_values['mu_beta0_cool2'],scale=prior_values['sigma_beta0_cool2'],size=n_samples)
        #beta1_cool2=npsoftplus(np.random.normal(loc=prior_values['mu_beta1_cool2_'],scale=prior_values['sigma_beta1_cool2_'],size=n_samples))
        
        beta0_cool2=np.random.normal(loc=prior_values['mu_beta0_cool2'],scale=prior_values['sigma_beta0_cool2'],size=n_samples)
        beta1_cool2=np.random.normal(loc=prior_values['mu_beta1_cool2'],scale=prior_values['sigma_beta1_cool2'],size=n_samples)
        beta2_cool2=np.random.normal(loc=prior_values['mu_beta2_cool2'],scale=prior_values['sigma_beta2_cool2'],size=n_samples)
        beta3_cool2=np.random.normal(loc=prior_values['mu_beta3_cool2'],scale=prior_values['sigma_beta3_cool2'],size=n_samples)
        beta4_cool2=np.random.normal(loc=prior_values['mu_beta4_cool2'],scale=prior_values['sigma_beta4_cool2'],size=n_samples)
        beta5_cool2=np.random.normal(loc=prior_values['mu_beta5_cool2'],scale=prior_values['sigma_beta5_cool2'],size=n_samples)

        beta6_cool2=np.random.normal(loc=prior_values['mu_beta6_cool2'],scale=prior_values['sigma_beta6_cool2'],size=n_samples)
        beta7_cool2=np.random.normal(loc=prior_values['mu_beta7_cool2'],scale=prior_values['sigma_beta7_cool2'],size=n_samples)
        beta8_cool2=np.random.normal(loc=prior_values['mu_beta8_cool2'],scale=prior_values['sigma_beta8_cool2'],size=n_samples)
        beta9_cool2=np.random.normal(loc=prior_values['mu_beta9_cool2'],scale=prior_values['sigma_beta9_cool2'],size=n_samples)
        beta10_cool2=np.random.normal(loc=prior_values['mu_beta10_cool2'],scale=prior_values['sigma_beta10_cool2'],size=n_samples)


        sigma_P_cool2=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_cool2_'],scale=prior_values['sigma_sigma_P_cool2_'],size=n_N)) 
        epsilon_cool2=np.random.normal(loc=0,scale=sigma_P_cool2,size=(n_samples,n_N))
        #epsilon_cool2=np.random.normal(loc=0,scale=prior_values['sigma_P_cool2_'],size=n_samples*n_N).reshape([n_samples,n_N])
        #P_cool2=npsigmoid(beta0_cool2[:,None]+np.dot(beta1_cool2[:,None],s_T_out[None,:])+epsilon_cool2)*cool_max/net_max
        
        P_cool2=npsigmoid(beta0_cool2[:,None]+\
                            np.dot(beta1_cool2[:,None],(s_wb_in[None,:]))+\
                            np.dot(beta2_cool2[:,None],(s_wb_in[None,:]**2))+\
                            np.dot(beta3_cool2[:,None],(s_wb_in[None,:]**3))+\
                            np.dot(beta4_cool2[:,None],(s_T_out[None,:]))+\
                            np.dot(beta5_cool2[:,None],(s_T_out[None,:]**2))+\
                            np.dot(beta6_cool2[:,None],(s_T_out[None,:]**3))+\
                            np.dot(beta7_cool2[:,None],(s_wb_in[None,:])*(s_T_out[None,:]))+\
                            np.dot(beta8_cool2[:,None],(s_wb_in[None,:])*(s_T_out[None,:]**2))+\
                            np.dot(beta9_cool2[:,None],(s_wb_in[None,:]**2)*(s_T_out[None,:]))+\
                            np.dot(beta10_cool2[:,None],(s_wb_in[None,:]**2)*(s_T_out[None,:]**2))+\
                            epsilon_cool2)*cool_max/net_max # n_samples,n_N

        

    elif (np.nansum(i_cool2)>0):
        raise ValueError("cool2 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_cool2=np.zeros([n_samples,n_N])


    if ('mu_mu_P_aux1' in prior_values.keys()) and (n_i_aux1>0):
        # mu_P_aux1_ sigma_mu_P_aux1_
        
        mu_P_aux1=np.random.normal(loc=prior_values['mu_mu_P_aux1'],scale=prior_values['sigma_mu_P_aux1'],size=n_N) 
        sigma_P_aux1=npsoftplus(np.random.normal(loc=prior_values['mu_sigma_P_aux1_'],scale=prior_values['sigma_sigma_P_aux1_'],size=n_N)) 
 
        P_aux1=npsigmoid(np.random.normal(loc=mu_P_aux1,scale=sigma_P_aux1,size=(n_samples,n_N)))*aux_max/net_max

    elif (np.nansum(i_aux1)>0):
        raise ValueError("aux1 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_aux1=np.zeros([n_samples,n_N])

    if ('mu_P_fan1_' in prior_values.keys()) and (n_i_fan1>0):
        P_fan1=npsoftplus(np.random.normal(loc=prior_values['mu_P_fan1_'],scale=prior_values['sigma_P_fan1_'],size=(n_samples,n_N)))
        # P_aux1=npsigmoid(np.random.normal(loc=prior_values['mu_P_aux1_'],scale=prior_values['sigma_P_aux1_'],size=n_samples))*aux_max/net_max

    elif (np.nansum(i_fan1)>0):
        raise ValueError("fan1 operation exists, but the model parameters(prior_values) doesn't include it. Please update model parameteres with the current data(input_values).")
    else:
        P_fan1=np.zeros([n_samples,n_N])
    
    # outputs by each operation + scale back by multiplying net_max
    P_heat1=P_heat1*i_heat1*net_max
    P_heat2=P_heat2*i_heat2*net_max
    P_df1=P_df1*i_df1*net_max
    P_df2=P_df2*i_df2*net_max
    
    P_cool1=P_cool1*input_values['i_cool1']*net_max
    P_cool2=P_cool2*input_values['i_cool2']*net_max
    P_aux1=P_aux1*input_values['i_aux1']*net_max
    P_fan1=P_fan1*input_values['i_fan1']*net_max
    
    P_hc=P_heat1+P_heat2+P_df1+P_df2+P_cool1+P_cool2+P_aux1+P_fan1 # no nansum

    outputs={"P_hc":P_hc,"P_heat1":P_heat1,"P_heat2":P_heat2,"P_cool1":P_cool1,"P_cool2":P_cool2,"P_df1":P_df1,"P_df2":P_df2,"P_aux1":P_aux1,"P_fan1":P_fan1,"mu_phi_df":mu_phi_df}

    
    return outputs