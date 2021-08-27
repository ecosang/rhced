import pymc3 as pm
import pandas as pd
import numpy as np
from theano import tensor as tt


__all__=['resnihcm','training','npsoftplus','sqrt_mean_square','npsigmoid','calculate_i_fraction','npreverse_logistic','nplogistic']

def npreverse_logistic(x,x0):
    return 1./(1+np.exp(-500*(-x+x0)))

def reverse_logistic(x,x0):
    return 1./(1+tt.exp(-500*(-x+x0)))

def nplogistic(x,x0):
    return 1./(1+np.exp(-500*(x-x0)))

def logistic(x,x0):
    return 1./(1+tt.exp(-500*(x-x0)))

def npsoftplus(x):
    return(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))
def sqrt_mean_square(x):
    return(np.array([np.sqrt(np.mean(x**2))]))

def calculate_i_fraction(i_target,i_hc):
    i_hc_base=i_hc.copy()
    i_hc_base[i_hc==0]=1e-3
    return(i_target/i_hc_base)

def npsigmoid(x):
    return(1/(1+np.exp(-x)))

def training(input_values,prior_values=None,n_training=60000,n_inference=5):
    # here prior values need to be updated based on input_values flags.
    
    prev_hist=1e+10
    if not(prior_values is None):
        if input_values['time_delta']!=prior_values['time_delta']:
            raise ValueError(f"input_values' time_delta is {input_values['time_delta']} but prior_values' time_delta is {prior_values['time_delta']}")
    for i in np.arange(n_inference):
        advi_=resnihcm(input_values=input_values,prior_values=prior_values)
        advi_.fit(n = n_training,obj_optimizer = pm.adam(learning_rate=0.001))
        curr_hist=advi_.hist[-1]
        if curr_hist<prev_hist:
            prev_hist=curr_hist
            advi=advi_
        else:
            del advi_

    mus=(advi.approx.bij.rmap(advi.approx.params[0].eval()))
    stds=(advi.approx.bij.rmap(npsoftplus(advi.approx.params[1].eval())))
    par_keys=mus.keys()

    curr_hist<prev_hist

    if prior_values is None:
        prior_values={}
        prev_prior_values=None
        prior_values['time_delta']=input_values['time_delta']
        prior_values['n_i_heat1']=0
        prior_values['n_i_df1']=0
        prior_values['n_i_cool1']=0
        prior_values['n_i_heat2']=0
        prior_values['n_i_df2']=0
        prior_values['n_i_cool2']=0
        prior_values['n_i_aux1']=0
        prior_values['n_i_fan1']=0

    else:
        prev_prior_values=prior_values.copy()
    
    #update or overwrite values
    for pk in (par_keys):
        if pk in ['P_heat1_','P_heat2_','P_cool1_','P_cool2_','P_df1_','P_df2_','P_aux1_','P_nonhc_']:
            pass # do nothing no parameter update
            #prior_values.update({'sigma_'+pk:sqrt_mean_square(stds[pk])})
        else:
            prior_values.update({'mu_'+pk:mus[pk]})
            prior_values.update({'sigma_'+pk:stds[pk]})
    
    mu_phi_df=prior_values['mu_phi_df']
    s_T_out=input_values['s_T_out']
    
    i_heat1_all=input_values['i_heat1_all']
    i_heat2_all=input_values['i_heat2_all']
    i_df1=i_heat1_all.copy()
    i_heat1=i_heat1_all.copy()
    
    i_df2=i_heat2_all.copy()
    i_heat2=i_heat2_all.copy()

    i_df1[(s_T_out>=mu_phi_df)]=0
    i_heat1[(s_T_out<mu_phi_df)]=0
    i_cool1=input_values['i_cool1']
    
    i_df2[(s_T_out>=mu_phi_df)]=0
    i_heat2[(s_T_out<mu_phi_df)]=0
    i_cool2=input_values['i_cool2']

    i_aux1=input_values['i_aux1']
    i_fan1=input_values['i_fan1']

    n_i_heat1_all=np.nansum(i_heat1_all>0).astype("int")
    n_i_heat2_all=np.nansum(i_heat2_all>0).astype("int")

    n_i_df1=np.nansum(i_df1>0).astype("int")
    n_i_heat1=np.nansum(i_heat1>0).astype("int")

    n_i_df2=np.nansum(i_df2>0).astype("int")
    n_i_heat2=np.nansum(i_heat2>0).astype("int")
    
    n_i_cool1=np.nansum(i_cool1>0).astype("int")
    n_i_cool2=np.nansum(i_cool2>0).astype("int")

    n_i_aux1=np.nansum(i_aux1>0).astype("int")
    n_i_fan1=np.nansum(i_fan1>0).astype("int")

    # store training data period. 
    prior_values['n_i_heat1']=prior_values['n_i_heat1']+n_i_heat1/(60/prior_values['time_delta']) #hours
    prior_values['n_i_df1']=prior_values['n_i_df1']+n_i_df1/(60/prior_values['time_delta'])
    prior_values['n_i_cool1']=prior_values['n_i_cool1']+n_i_cool1/(60/prior_values['time_delta'])
    
    prior_values['n_i_heat2']=prior_values['n_i_heat2']+n_i_heat2/(60/prior_values['time_delta']) #hours
    prior_values['n_i_df2']=prior_values['n_i_df2']+n_i_df2/(60/prior_values['time_delta'])
    prior_values['n_i_cool2']=prior_values['n_i_cool2']+n_i_cool2/(60/prior_values['time_delta'])

    prior_values['n_i_aux1']=prior_values['n_i_aux1']+n_i_aux1/(60/prior_values['time_delta'])
    prior_values['n_i_fan1']=prior_values['n_i_fan1']+n_i_fan1/(60/prior_values['time_delta'])
    prior_values['time_delta']

    print(f"During this training, heat1 hours: {n_i_heat1/(60/prior_values['time_delta'])}, \
            heat2 hours: {n_i_heat2/(60/prior_values['time_delta'])}, \
            df1 hours: {n_i_df1/(60/prior_values['time_delta'])}, \
            df2 hours: {n_i_df2/(60/prior_values['time_delta'])}, \
            cool1 hours: {n_i_cool1/(60/prior_values['time_delta'])}, \
            cool2 hours: {n_i_cool2/(60/prior_values['time_delta'])}, \
            aux1 hours: {n_i_aux1/(60/prior_values['time_delta'])}, \
            fan1 hours: {n_i_fan1/(60/prior_values['time_delta'])}")

    print(f"So far, heat1 hours: {prior_values['n_i_heat1']}, \
            heat2 hours: {prior_values['n_i_heat2']}, \
            df1 hours: {prior_values['n_i_df1']}, \
            df2 hours: {prior_values['n_i_df2']}, \
            cool1 hours: {prior_values['n_i_cool1']}, \
            cool2 hours: {prior_values['n_i_cool2']}, \
            aux1 hours: {prior_values['n_i_aux1']}, \
            fan1 hours: {prior_values['n_i_fan1']}")
    
    if not(prev_prior_values is None):
        # do not update
        if (n_i_heat1_all==0)  and ('mu_phi_df' in prev_prior_values.keys()):
            prior_values['mu_phi_df']=prev_prior_values['mu_phi_df']
            prior_values['sigma_phi_df']=prev_prior_values['sigma_phi_df']

        if (n_i_heat1==0)  and ('mu_beta0_heat1' in prev_prior_values.keys()):
            prior_values['mu_sigma_P_heat1_']=prev_prior_values['mu_sigma_P_heat1_']
            prior_values['sigma_sigma_P_heat1_']=prev_prior_values['sigma_sigma_P_heat1_']

            prior_values['mu_beta0_heat1']=prev_prior_values['mu_beta0_heat1']
            prior_values['sigma_beta0_heat1']=prev_prior_values['sigma_beta0_heat1']
            prior_values['mu_beta1_heat1']=prev_prior_values['mu_beta1_heat1']
            prior_values['sigma_beta1_heat1']=prev_prior_values['sigma_beta1_heat1']
            prior_values['mu_beta2_heat1']=prev_prior_values['mu_beta2_heat1']
            prior_values['sigma_beta2_heat1']=prev_prior_values['sigma_beta2_heat1']
            prior_values['mu_beta3_heat1']=prev_prior_values['mu_beta3_heat1']
            prior_values['sigma_beta3_heat1']=prev_prior_values['sigma_beta3_heat1']
            prior_values['mu_beta4_heat1']=prev_prior_values['mu_beta4_heat1']
            prior_values['sigma_beta4_heat1']=prev_prior_values['sigma_beta4_heat1']
            prior_values['mu_beta5_heat1']=prev_prior_values['mu_beta5_heat1']
            prior_values['sigma_beta5_heat1']=prev_prior_values['sigma_beta5_heat1']
            prior_values['mu_beta6_heat1']=prev_prior_values['mu_beta6_heat1']
            prior_values['sigma_beta6_heat1']=prev_prior_values['sigma_beta6_heat1']
            prior_values['mu_beta7_heat1']=prev_prior_values['mu_beta7_heat1']
            prior_values['sigma_beta7_heat1']=prev_prior_values['sigma_beta7_heat1']
            prior_values['mu_beta8_heat1']=prev_prior_values['mu_beta8_heat1']
            prior_values['sigma_beta8_heat1']=prev_prior_values['sigma_beta8_heat1']
            prior_values['mu_beta9_heat1']=prev_prior_values['mu_beta9_heat1']
            prior_values['sigma_beta9_heat1']=prev_prior_values['sigma_beta9_heat1']
            prior_values['mu_beta10_heat1']=prev_prior_values['mu_beta10_heat1']
            prior_values['sigma_beta10_heat1']=prev_prior_values['sigma_beta10_heat1']


        if (n_i_heat2==0)  and ('mu_beta0_heat2' in prev_prior_values.keys()):
            prior_values['mu_sigma_P_heat2_']=prev_prior_values['mu_sigma_P_heat2_']
            prior_values['sigma_sigma_P_heat2_']=prev_prior_values['sigma_sigma_P_heat2_']

            prior_values['mu_beta0_heat2']=prev_prior_values['mu_beta0_heat2']
            prior_values['sigma_beta0_heat2']=prev_prior_values['sigma_beta0_heat2']
            prior_values['mu_beta1_heat2']=prev_prior_values['mu_beta1_heat2']
            prior_values['sigma_beta1_heat2']=prev_prior_values['sigma_beta1_heat2']
            prior_values['mu_beta2_heat2']=prev_prior_values['mu_beta2_heat2']
            prior_values['sigma_beta2_heat2']=prev_prior_values['sigma_beta2_heat2']
            prior_values['mu_beta3_heat2']=prev_prior_values['mu_beta3_heat2']
            prior_values['sigma_beta3_heat2']=prev_prior_values['sigma_beta3_heat2']
            prior_values['mu_beta4_heat2']=prev_prior_values['mu_beta4_heat2']
            prior_values['sigma_beta4_heat2']=prev_prior_values['sigma_beta4_heat2']
            prior_values['mu_beta5_heat2']=prev_prior_values['mu_beta5_heat2']
            prior_values['sigma_beta5_heat2']=prev_prior_values['sigma_beta5_heat2']
            prior_values['mu_beta6_heat2']=prev_prior_values['mu_beta6_heat2']
            prior_values['sigma_beta6_heat2']=prev_prior_values['sigma_beta6_heat2']
            prior_values['mu_beta7_heat2']=prev_prior_values['mu_beta7_heat2']
            prior_values['sigma_beta7_heat2']=prev_prior_values['sigma_beta7_heat2']
            prior_values['mu_beta8_heat2']=prev_prior_values['mu_beta8_heat2']
            prior_values['sigma_beta8_heat2']=prev_prior_values['sigma_beta8_heat2']
            prior_values['mu_beta9_heat2']=prev_prior_values['mu_beta9_heat2']
            prior_values['sigma_beta9_heat2']=prev_prior_values['sigma_beta9_heat2']
            prior_values['mu_beta10_heat2']=prev_prior_values['mu_beta10_heat2']
            prior_values['sigma_beta10_heat2']=prev_prior_values['sigma_beta10_heat2']


        if (n_i_cool1==0) and ('mu_beta0_cool1' in prev_prior_values.keys()):
            prior_values['mu_sigma_P_cool1_']=prev_prior_values['mu_sigma_P_cool1_']
            prior_values['sigma_sigma_P_cool1_']=prev_prior_values['sigma_sigma_P_cool1_']
            
            prior_values['mu_beta0_cool1']=prev_prior_values['sigma_beta0_cool1']
            prior_values['sigma_beta0_cool1']=prev_prior_values['sigma_beta0_cool1']
            prior_values['mu_beta1_cool1']=prev_prior_values['mu_beta1_cool1']
            prior_values['sigma_beta1_cool1']=prev_prior_values['sigma_beta1_cool1']
            prior_values['mu_beta2_cool1']=prev_prior_values['mu_beta2_cool1']
            prior_values['sigma_beta2_cool1']=prev_prior_values['sigma_beta2_cool1']
            prior_values['mu_beta3_cool1']=prev_prior_values['mu_beta3_cool1']
            prior_values['sigma_beta3_cool1']=prev_prior_values['sigma_beta3_cool1']
            prior_values['mu_beta4_cool1']=prev_prior_values['mu_beta4_cool1']
            prior_values['sigma_beta4_cool1']=prev_prior_values['sigma_beta4_cool1']
            prior_values['mu_beta5_cool1']=prev_prior_values['mu_beta5_cool1']
            prior_values['sigma_beta5_cool1']=prev_prior_values['sigma_beta5_cool1']
            prior_values['mu_beta6_cool1']=prev_prior_values['mu_beta6_cool1']
            prior_values['sigma_beta6_cool1']=prev_prior_values['sigma_beta6_cool1']
            prior_values['mu_beta7_cool1']=prev_prior_values['mu_beta7_cool1']
            prior_values['sigma_beta7_cool1']=prev_prior_values['sigma_beta7_cool1']
            prior_values['mu_beta8_cool1']=prev_prior_values['mu_beta8_cool1']
            prior_values['sigma_beta8_cool1']=prev_prior_values['sigma_beta8_cool1']
            prior_values['mu_beta9_cool1']=prev_prior_values['mu_beta9_cool1']
            prior_values['sigma_beta9_cool1']=prev_prior_values['sigma_beta9_cool1']
            prior_values['mu_beta10_cool1']=prev_prior_values['mu_beta10_cool1']
            prior_values['sigma_beta10_cool1']=prev_prior_values['sigma_beta10_cool1']

        if (n_i_cool2==0) and ('mu_beta0_cool2' in prev_prior_values.keys()):
            prior_values['mu_sigma_P_cool2_']=prev_prior_values['mu_sigma_P_cool2_']
            prior_values['sigma_sigma_P_cool2_']=prev_prior_values['sigma_sigma_P_cool2_']
            
            prior_values['mu_beta0_cool2']=prev_prior_values['sigma_beta0_cool2']
            prior_values['sigma_beta0_cool2']=prev_prior_values['sigma_beta0_cool2']
            prior_values['mu_beta1_cool2']=prev_prior_values['mu_beta1_cool2']
            prior_values['sigma_beta1_cool2']=prev_prior_values['sigma_beta1_cool2']
            prior_values['mu_beta2_cool2']=prev_prior_values['mu_beta2_cool2']
            prior_values['sigma_beta2_cool2']=prev_prior_values['sigma_beta2_cool2']
            prior_values['mu_beta3_cool2']=prev_prior_values['mu_beta3_cool2']
            prior_values['sigma_beta3_cool2']=prev_prior_values['sigma_beta3_cool2']
            prior_values['mu_beta4_cool2']=prev_prior_values['mu_beta4_cool2']
            prior_values['sigma_beta4_cool2']=prev_prior_values['sigma_beta4_cool2']
            prior_values['mu_beta5_cool2']=prev_prior_values['mu_beta5_cool2']
            prior_values['sigma_beta5_cool2']=prev_prior_values['sigma_beta5_cool2']
            prior_values['mu_beta6_cool2']=prev_prior_values['mu_beta6_cool2']
            prior_values['sigma_beta6_cool2']=prev_prior_values['sigma_beta6_cool2']
            prior_values['mu_beta7_cool2']=prev_prior_values['mu_beta7_cool2']
            prior_values['sigma_beta7_cool2']=prev_prior_values['sigma_beta7_cool2']
            prior_values['mu_beta8_cool2']=prev_prior_values['mu_beta8_cool2']
            prior_values['sigma_beta8_cool2']=prev_prior_values['sigma_beta8_cool2']
            prior_values['mu_beta9_cool2']=prev_prior_values['mu_beta9_cool2']
            prior_values['sigma_beta9_cool2']=prev_prior_values['sigma_beta9_cool2']
            prior_values['mu_beta10_cool2']=prev_prior_values['mu_beta10_cool2']
            prior_values['sigma_beta10_cool2']=prev_prior_values['sigma_beta10_cool2']



        if (n_i_df1==0)  and ('mu_beta0_df1' in prev_prior_values.keys()):
            prior_values['mu_beta0_df1']=prev_prior_values['sigma_beta0_df1']
            prior_values['sigma_beta0_df1']=prev_prior_values['sigma_beta0_df1']
            prior_values['mu_beta1_df1']=prev_prior_values['mu_beta1_df1']
            prior_values['sigma_beta1_df1']=prev_prior_values['sigma_beta1_df1']
            prior_values['mu_beta2_df1']=prev_prior_values['mu_beta2_df1']
            prior_values['sigma_beta2_df1']=prev_prior_values['sigma_beta2_df1']
            prior_values['mu_beta3_df1']=prev_prior_values['mu_beta3_df1']
            prior_values['sigma_beta3_df1']=prev_prior_values['sigma_beta3_df1']
            prior_values['mu_beta4_df1']=prev_prior_values['mu_beta4_df1']
            prior_values['sigma_beta4_df1']=prev_prior_values['sigma_beta4_df1']
            prior_values['mu_beta5_df1']=prev_prior_values['mu_beta5_df1']
            prior_values['sigma_beta5_df1']=prev_prior_values['sigma_beta5_df1']
            
            prior_values['mu_beta6_df1']=prev_prior_values['mu_beta6_df1']
            prior_values['sigma_beta6_df1']=prev_prior_values['sigma_beta6_df1']
            prior_values['mu_beta7_df1']=prev_prior_values['mu_beta7_df1']
            prior_values['sigma_beta7_df1']=prev_prior_values['sigma_beta7_df1']
            prior_values['mu_beta8_df1']=prev_prior_values['mu_beta8_df1']
            prior_values['sigma_beta8_df1']=prev_prior_values['sigma_beta8_df1']
            prior_values['mu_beta9_df1']=prev_prior_values['mu_beta9_df1']
            prior_values['sigma_beta9_df1']=prev_prior_values['sigma_beta9_df1']
            prior_values['mu_beta10_df1']=prev_prior_values['mu_beta10_df1']
            prior_values['sigma_beta10_df1']=prev_prior_values['sigma_beta10_df1']

            # prior_values['mu_sigma_P_df1_']=prev_prior_values['mu_sigma_P_df1_']
            # prior_values['sigma_sigma_P_df1_']=prev_prior_values['sigma_sigma_P_df1_']
            # prior_values['mu_beta1_df1_']=prev_prior_values['mu_beta1_df1_']
            # prior_values['sigma_beta1_df1_']=prev_prior_values['sigma_beta1_df1_']
            # prior_values['mu_beta0_df1']=prev_prior_values['mu_beta0_df1']
            # prior_values['sigma_beta0_df1']=prev_prior_values['sigma_beta0_df1']
        
        if (n_i_df2==0)  and ('mu_beta0_df2' in prev_prior_values.keys()):
            prior_values['mu_beta0_df2']=prev_prior_values['sigma_beta0_df2']
            prior_values['sigma_beta0_df2']=prev_prior_values['sigma_beta0_df2']
            prior_values['mu_beta1_df2']=prev_prior_values['mu_beta1_df2']
            prior_values['sigma_beta1_df2']=prev_prior_values['sigma_beta1_df2']
            prior_values['mu_beta2_df2']=prev_prior_values['mu_beta2_df2']
            prior_values['sigma_beta2_df2']=prev_prior_values['sigma_beta2_df2']
            prior_values['mu_beta3_df2']=prev_prior_values['mu_beta3_df2']
            prior_values['sigma_beta3_df2']=prev_prior_values['sigma_beta3_df2']
            prior_values['mu_beta4_df2']=prev_prior_values['mu_beta4_df2']
            prior_values['sigma_beta4_df2']=prev_prior_values['sigma_beta4_df2']
            prior_values['mu_beta5_df2']=prev_prior_values['mu_beta5_df2']
            prior_values['sigma_beta5_df2']=prev_prior_values['sigma_beta5_df2']
            
            prior_values['mu_beta6_df2']=prev_prior_values['mu_beta6_df2']
            prior_values['sigma_beta6_df2']=prev_prior_values['sigma_beta6_df2']
            prior_values['mu_beta7_df2']=prev_prior_values['mu_beta7_df2']
            prior_values['sigma_beta7_df2']=prev_prior_values['sigma_beta7_df2']
            prior_values['mu_beta8_df2']=prev_prior_values['mu_beta8_df2']
            prior_values['sigma_beta8_df2']=prev_prior_values['sigma_beta8_df2']
            prior_values['mu_beta9_df2']=prev_prior_values['mu_beta9_df2']
            prior_values['sigma_beta9_df2']=prev_prior_values['sigma_beta9_df2']
            prior_values['mu_beta10_df2']=prev_prior_values['mu_beta10_df2']
            prior_values['sigma_beta10_df2']=prev_prior_values['sigma_beta10_df2']



        if (n_i_aux1==0)  and ('mu_mu_P_aux1' in prev_prior_values.keys()):
            prior_values['mu_mu_P_aux1']=prev_prior_values['mu_mu_P_aux1']
            prior_values['sigma_mu_P_aux1']=prev_prior_values['sigma_mu_P_aux1']
            prior_values['mu_sigma_P_aux1_']=prev_prior_values['mu_sigma_P_aux1_']
            prior_values['sigma_sigma_P_aux1_']=prev_prior_values['sigma_sigma_P_aux1_']

        if (n_i_fan1==0)  and ('mu_P_fan1_' in prev_prior_values.keys()):
            prior_values['mu_P_fan1_']=prev_prior_values['mu_mu_P_fan1_']
            prior_values['sigma_P_fan1_']=prev_prior_values['sigma_mu_P_fan1_']
    else:
        #first training but no update for heat1_ndf
        if (n_i_heat1==0) and ('mu_beta0_heat1' in prior_values.keys()):
            
            prior_values.pop('mu_sigma_P_heat1_')
            prior_values.pop('sigma_sigma_P_heat1_')

            prior_values.pop('mu_beta0_heat1')
            prior_values.pop('sigma_beta0_heat1')
            prior_values.pop('mu_beta1_heat1')
            prior_values.pop('sigma_beta1_heat1')
            prior_values.pop('mu_beta2_heat1')
            prior_values.pop('sigma_beta2_heat1')
            prior_values.pop('mu_beta3_heat1')
            prior_values.pop('sigma_beta3_heat1')
            prior_values.pop('mu_beta4_heat1')
            prior_values.pop('sigma_beta4_heat1')
            prior_values.pop('mu_beta5_heat1')
            prior_values.pop('sigma_beta6_heat1')
            prior_values.pop('mu_beta7_heat1')
            prior_values.pop('sigma_beta7_heat1')
            prior_values.pop('mu_beta8_heat1')
            prior_values.pop('sigma_beta8_heat1')
            prior_values.pop('mu_beta9_heat1')
            prior_values.pop('sigma_beta9_heat1')
            prior_values.pop('mu_beta10_heat1')
            prior_values.pop('sigma_beta10_heat1')
        if (n_i_heat2==0) and ('mu_beta0_heat2' in prior_values.keys()):
            
            prior_values.pop('mu_sigma_P_heat2_')
            prior_values.pop('sigma_sigma_P_heat2_')

            prior_values.pop('mu_beta0_heat2')
            prior_values.pop('sigma_beta0_heat2')
            prior_values.pop('mu_beta1_heat2')
            prior_values.pop('sigma_beta1_heat2')
            prior_values.pop('mu_beta2_heat2')
            prior_values.pop('sigma_beta2_heat2')
            prior_values.pop('mu_beta3_heat2')
            prior_values.pop('sigma_beta3_heat2')
            prior_values.pop('mu_beta4_heat2')
            prior_values.pop('sigma_beta4_heat2')
            prior_values.pop('mu_beta5_heat2')
            prior_values.pop('sigma_beta6_heat2')
            prior_values.pop('mu_beta7_heat2')
            prior_values.pop('sigma_beta7_heat2')
            prior_values.pop('mu_beta8_heat2')
            prior_values.pop('sigma_beta8_heat2')
            prior_values.pop('mu_beta9_heat2')
            prior_values.pop('sigma_beta9_heat2')
            prior_values.pop('mu_beta10_heat2')
            prior_values.pop('sigma_beta10_heat2')

        elif (n_i_df1==0) and ('mu_beta0_df1' in prior_values.keys()):
            prior_values.pop('mu_beta0_df1')
            prior_values.pop('sigma_beta0_df1')
            prior_values.pop('mu_beta1_df1')
            prior_values.pop('sigma_beta1_df1')
            prior_values.pop('mu_beta2_df1')
            prior_values.pop('sigma_beta2_df1')
            prior_values.pop('mu_beta3_df1')
            prior_values.pop('sigma_beta3_df1')
            prior_values.pop('mu_beta4_df1')
            prior_values.pop('sigma_beta4_df1')
            prior_values.pop('mu_beta5_df1')
            prior_values.pop('sigma_beta5_df1')

            prior_values.pop('mu_beta6_df1')
            prior_values.pop('sigma_beta6_df1')
            prior_values.pop('mu_beta7_df1')
            prior_values.pop('sigma_beta7_df1')
            prior_values.pop('mu_beta8_df1')
            prior_values.pop('sigma_beta8_df1')
            prior_values.pop('mu_beta9_df1')
            prior_values.pop('sigma_beta9_df1')
            prior_values.pop('mu_beta10_df1')
            prior_values.pop('sigma_beta10_df1')

            # prior_values.pop('mu_sigma_P_df1_')
            # prior_values.pop('sigma_sigma_P_df1_')
            # prior_values.pop('mu_beta1_df1_')
            # prior_values.pop('sigma_beta1_df1_')
            # prior_values.pop('mu_beta0_df1')
            # prior_values.pop('sigma_beta0_df1')
        
        elif (n_i_df2==0) and ('mu_beta0_df2' in prior_values.keys()):
            prior_values.pop('mu_beta0_df2')
            prior_values.pop('sigma_beta0_df2')
            prior_values.pop('mu_beta1_df2')
            prior_values.pop('sigma_beta1_df2')
            prior_values.pop('mu_beta2_df2')
            prior_values.pop('sigma_beta2_df2')
            prior_values.pop('mu_beta3_df2')
            prior_values.pop('sigma_beta3_df2')
            prior_values.pop('mu_beta4_df2')
            prior_values.pop('sigma_beta4_df2')
            prior_values.pop('mu_beta5_df2')
            prior_values.pop('sigma_beta5_df2')

            prior_values.pop('mu_beta6_df2')
            prior_values.pop('sigma_beta6_df2')
            prior_values.pop('mu_beta7_df2')
            prior_values.pop('sigma_beta7_df2')
            prior_values.pop('mu_beta8_df2')
            prior_values.pop('sigma_beta8_df2')
            prior_values.pop('mu_beta9_df2')
            prior_values.pop('sigma_beta9_df2')
            prior_values.pop('mu_beta10_df2')
            prior_values.pop('sigma_beta10_df2')
    del advi
    return input_values,prior_values

def resnihcm(input_values,prior_values=None):
    
    if prior_values is None:
        print("New training")
    
    T_out=input_values['s_T_out'] 
    T_in=input_values['s_T_in']
    wb_in=input_values['s_wb_in']
    y_net=input_values['s_net'] 
    n_N=T_out.shape[0] # t_k=1,2,...,N

    heat_max=input_values['scale_const']['heat_max']
    cool_max=input_values['scale_const']['cool_max']
    df_max=input_values['scale_const']['df_max']
    aux_max=input_values['scale_const']['aux_max']
    net_max=input_values['scale_const']['net_max']

    if ('i_heat1_all' in input_values.keys()):
        # 1st stage heat pump heating.
        i_heat1_all=input_values['i_heat1_all']
        n_i_heat1_all=np.nansum(i_heat1_all>0).astype("int")
        
        if (prior_values is None):
            # mu_beta0_heat1=0.5
            # sigma_beta0_heat1=0.5
            # mu_beta1_heat1=-3.0
            # sigma_beta1_heat1=1.8
            # 1 x, x2, x3, y, y2, y3, xy, xy2, x2y, x2y2
            mu_beta0_heat1=0.
            sigma_beta0_heat1=0.25
            mu_beta1_heat1=0.0
            sigma_beta1_heat1=0.25
            mu_beta2_heat1=0.0
            sigma_beta2_heat1=0.25
            mu_beta3_heat1=0.0
            sigma_beta3_heat1=0.25
            mu_beta4_heat1=0.0
            sigma_beta4_heat1=0.25
            mu_beta5_heat1=0.0
            sigma_beta5_heat1=0.25
            
            mu_beta6_heat1=0.0
            sigma_beta6_heat1=0.25
            mu_beta7_heat1=0.0
            sigma_beta7_heat1=0.25
            mu_beta8_heat1=0.0
            sigma_beta8_heat1=0.25
            mu_beta9_heat1=0.0
            sigma_beta9_heat1=0.25
            mu_beta10_heat1=0.0
            sigma_beta10_heat1=0.25
            
            mu_sigma_P_heat1=-0.5
            sigma_sigma_P_heat1=0.5

            # defrost control
            # mu_beta0_df1=0.5
            # sigma_beta0_df1=0.5
            # mu_beta1_df1=-3.0
            # sigma_beta1_df1=1.8

            mu_beta0_df1=0.0
            sigma_beta0_df1=0.25
            mu_beta1_df1=0.0
            sigma_beta1_df1=0.25
            mu_beta2_df1=0.0
            sigma_beta2_df1=0.25
            mu_beta3_df1=0.0
            sigma_beta3_df1=0.25
            mu_beta4_df1=0.0
            sigma_beta4_df1=0.25
            mu_beta5_df1=0.0
            sigma_beta5_df1=0.25

            mu_beta6_df1=0.0
            sigma_beta6_df1=0.25
            mu_beta7_df1=0.0
            sigma_beta7_df1=0.25
            mu_beta8_df1=0.0
            sigma_beta8_df1=0.25
            mu_beta9_df1=0.0
            sigma_beta9_df1=0.25
            mu_beta10_df1=0.0
            sigma_beta10_df1=0.25


            mu_sigma_P_df1=-0.5
            sigma_sigma_P_df1=0.5

            mu_phi_df=-1/3
            sigma_phi_df=0.005
            
        elif not('mu_beta0_heat1' in prior_values.keys()):
            # new training of mu_beta0_heat1
            # mu_beta0_heat1=0.5
            # sigma_beta0_heat1=0.5
            # mu_beta1_heat1=-3.0
            # sigma_beta1_heat1=1.8
            
            mu_beta0_heat1=0.0
            sigma_beta0_heat1=0.25
            mu_beta1_heat1=0.0
            sigma_beta1_heat1=0.25
            mu_beta2_heat1=0.0
            sigma_beta2_heat1=0.25
            mu_beta3_heat1=0.0
            sigma_beta3_heat1=0.25
            mu_beta4_heat1=0.0
            sigma_beta4_heat1=0.25
            mu_beta5_heat1=0.0
            sigma_beta5_heat1=0.25

            mu_beta6_heat1=0.0
            sigma_beta6_heat1=0.25
            mu_beta7_heat1=0.0
            sigma_beta7_heat1=0.25
            mu_beta8_heat1=0.0
            sigma_beta8_heat1=0.25
            mu_beta9_heat1=0.0
            sigma_beta9_heat1=0.25
            mu_beta10_heat1=0.0
            sigma_beta10_heat1=0.25

            mu_sigma_P_heat1=-0.5
            sigma_sigma_P_heat1=0.5

            mu_beta0_df1=0.0
            sigma_beta0_df1=0.25
            mu_beta1_df1=0.0
            sigma_beta1_df1=0.25
            mu_beta2_df1=0.0
            sigma_beta2_df1=0.25
            mu_beta3_df1=0.0
            sigma_beta3_df1=0.25
            mu_beta4_df1=0.0
            sigma_beta4_df1=0.25
            mu_beta5_df1=0.0
            sigma_beta5_df1=0.25

            mu_beta6_df1=0.0
            sigma_beta6_df1=0.25
            mu_beta7_df1=0.0
            sigma_beta7_df1=0.25
            mu_beta8_df1=0.0
            sigma_beta8_df1=0.25
            mu_beta9_df1=0.0
            sigma_beta9_df1=0.25
            mu_beta10_df1=0.0
            sigma_beta10_df1=0.25

            mu_sigma_P_df1=-0.5
            sigma_sigma_P_df1=0.5

            mu_phi_df=-1/3
            sigma_phi_df=0.2
        else:        
            mu_beta0_heat1=prior_values['mu_beta0_heat1']
            sigma_beta0_heat1=prior_values['sigma_beta0_heat1']
            mu_beta1_heat1=prior_values['mu_beta1_heat1']
            sigma_beta1_heat1=prior_values['sigma_beta1_heat1']
            mu_beta2_heat1=prior_values['mu_beta2_heat1']
            sigma_beta2_heat1=prior_values['sigma_beta2_heat1']
            mu_beta3_heat1=prior_values['mu_beta3_heat1']
            sigma_beta3_heat1=prior_values['sigma_beta3_heat1']
            mu_beta4_heat1=prior_values['mu_beta4_heat1']
            sigma_beta4_heat1=prior_values['sigma_beta4_heat1']
            mu_beta5_heat1=prior_values['mu_beta5_heat1']
            sigma_beta5_heat1=prior_values['sigma_beta5_heat1']

            mu_beta6_heat1=prior_values['mu_beta6_heat1']
            sigma_beta6_heat1=prior_values['sigma_beta6_heat1']
            mu_beta7_heat1=prior_values['mu_beta7_heat1']
            sigma_beta7_heat1=prior_values['sigma_beta7_heat1']
            mu_beta8_heat1=prior_values['mu_beta8_heat1']
            sigma_beta8_heat1=prior_values['sigma_beta8_heat1']
            mu_beta9_heat1=prior_values['mu_beta9_heat1']
            sigma_beta9_heat1=prior_values['sigma_beta9_heat1']
            mu_beta10_heat1=prior_values['mu_beta10_heat1']
            sigma_beta10_heat1=prior_values['sigma_beta10_heat1']


            mu_sigma_P_heat1=prior_values['mu_sigma_P_heat1_']
            sigma_sigma_P_heat1=prior_values['sigma_sigma_P_heat1_']

            # defrost control
            # mu_beta0_df1=prior_values['mu_beta0_df1']
            # sigma_beta0_df1=prior_values['sigma_beta0_df1']
            # mu_beta1_df1=prior_values['mu_beta1_df1_']
            # sigma_beta1_df1=prior_values['sigma_beta1_df1_']

            mu_beta0_df1=prior_values['mu_beta0_df1']
            sigma_beta0_df1=prior_values['sigma_beta0_df1']
            mu_beta1_df1=prior_values['mu_beta1_df1']
            sigma_beta1_df1=prior_values['sigma_beta1_df1']
            mu_beta2_df1=prior_values['mu_beta2_df1']
            sigma_beta2_df1=prior_values['sigma_beta2_df1']
            mu_beta3_df1=prior_values['mu_beta3_df1']
            sigma_beta3_df1=prior_values['sigma_beta3_df1']
            mu_beta4_df1=prior_values['mu_beta4_df1']
            sigma_beta4_df1=prior_values['sigma_beta4_df1']
            mu_beta5_df1=prior_values['mu_beta5_df1']
            sigma_beta5_df1=prior_values['sigma_beta5_df1']

            mu_beta6_df1=prior_values['mu_beta6_df1']
            sigma_beta6_df1=prior_values['sigma_beta6_df1']
            mu_beta7_df1=prior_values['mu_beta7_df1']
            sigma_beta7_df1=prior_values['sigma_beta7_df1']
            mu_beta8_df1=prior_values['mu_beta8_df1']
            sigma_beta8_df1=prior_values['sigma_beta8_df1']
            mu_beta9_df1=prior_values['mu_beta9_df1']
            sigma_beta9_df1=prior_values['sigma_beta9_df1']
            mu_beta10_df1=prior_values['mu_beta10_df1']
            sigma_beta10_df1=prior_values['sigma_beta10_df1']

            mu_sigma_P_df1=prior_values['mu_sigma_P_df1_']
            sigma_sigma_P_df1=prior_values['sigma_sigma_P_df1_']

            mu_phi_df=prior_values['mu_phi_df']
            sigma_phi_df=prior_values['sigma_phi_df']
    else:
        i_heat1_all=np.zeros_like(T_out)
        n_i_heat1_all=0

    if ('i_heat2_all' in input_values.keys()):
        # 1st stage heat pump heating.
        i_heat2_all=input_values['i_heat2_all']
        n_i_heat2_all=np.nansum(i_heat2_all>0).astype("int")
        
        if (prior_values is None):
            # mu_beta0_heat2=0.5
            # sigma_beta0_heat2=0.5
            # mu_beta1_heat2=-3.0
            # sigma_beta1_heat2=1.8
            # 1 x, x2, x3, y, y2, y3, xy, xy2, x2y, x2y2
            mu_beta0_heat2=0.
            sigma_beta0_heat2=0.25
            mu_beta1_heat2=0.0
            sigma_beta1_heat2=0.25
            mu_beta2_heat2=0.0
            sigma_beta2_heat2=0.25
            mu_beta3_heat2=0.0
            sigma_beta3_heat2=0.25
            mu_beta4_heat2=0.0
            sigma_beta4_heat2=0.25
            mu_beta5_heat2=0.0
            sigma_beta5_heat2=0.25
            
            mu_beta6_heat2=0.0
            sigma_beta6_heat2=0.25
            mu_beta7_heat2=0.0
            sigma_beta7_heat2=0.25
            mu_beta8_heat2=0.0
            sigma_beta8_heat2=0.25
            mu_beta9_heat2=0.0
            sigma_beta9_heat2=0.25
            mu_beta10_heat2=0.0
            sigma_beta10_heat2=0.25
            
            mu_sigma_P_heat2=-0.5
            sigma_sigma_P_heat2=0.5

            # defrost control
            # mu_beta0_df2=0.5
            # sigma_beta0_df2=0.5
            # mu_beta1_df2=-3.0
            # sigma_beta1_df2=1.8

            mu_beta0_df2=0.0
            sigma_beta0_df2=0.25
            mu_beta1_df2=0.0
            sigma_beta1_df2=0.25
            mu_beta2_df2=0.0
            sigma_beta2_df2=0.25
            mu_beta3_df2=0.0
            sigma_beta3_df2=0.25
            mu_beta4_df2=0.0
            sigma_beta4_df2=0.25
            mu_beta5_df2=0.0
            sigma_beta5_df2=0.25

            mu_beta6_df2=0.0
            sigma_beta6_df2=0.25
            mu_beta7_df2=0.0
            sigma_beta7_df2=0.25
            mu_beta8_df2=0.0
            sigma_beta8_df2=0.25
            mu_beta9_df2=0.0
            sigma_beta9_df2=0.25
            mu_beta10_df2=0.0
            sigma_beta10_df2=0.25


            mu_sigma_P_df2=-0.5
            sigma_sigma_P_df2=0.5

            mu_phi_df=-1/3
            sigma_phi_df=0.005
            
        elif not('mu_beta0_heat2' in prior_values.keys()):
            # new training of mu_beta0_heat2
            # mu_beta0_heat2=0.5
            # sigma_beta0_heat2=0.5
            # mu_beta1_heat2=-3.0
            # sigma_beta1_heat2=1.8
            
            mu_beta0_heat2=0.0
            sigma_beta0_heat2=0.25
            mu_beta1_heat2=0.0
            sigma_beta1_heat2=0.25
            mu_beta2_heat2=0.0
            sigma_beta2_heat2=0.25
            mu_beta3_heat2=0.0
            sigma_beta3_heat2=0.25
            mu_beta4_heat2=0.0
            sigma_beta4_heat2=0.25
            mu_beta5_heat2=0.0
            sigma_beta5_heat2=0.25

            mu_beta6_heat2=0.0
            sigma_beta6_heat2=0.25
            mu_beta7_heat2=0.0
            sigma_beta7_heat2=0.25
            mu_beta8_heat2=0.0
            sigma_beta8_heat2=0.25
            mu_beta9_heat2=0.0
            sigma_beta9_heat2=0.25
            mu_beta10_heat2=0.0
            sigma_beta10_heat2=0.25

            mu_sigma_P_heat2=-0.5
            sigma_sigma_P_heat2=0.5

            mu_beta0_df2=0.0
            sigma_beta0_df2=0.25
            mu_beta1_df2=0.0
            sigma_beta1_df2=0.25
            mu_beta2_df2=0.0
            sigma_beta2_df2=0.25
            mu_beta3_df2=0.0
            sigma_beta3_df2=0.25
            mu_beta4_df2=0.0
            sigma_beta4_df2=0.25
            mu_beta5_df2=0.0
            sigma_beta5_df2=0.25

            mu_beta6_df2=0.0
            sigma_beta6_df2=0.25
            mu_beta7_df2=0.0
            sigma_beta7_df2=0.25
            mu_beta8_df2=0.0
            sigma_beta8_df2=0.25
            mu_beta9_df2=0.0
            sigma_beta9_df2=0.25
            mu_beta10_df2=0.0
            sigma_beta10_df2=0.25

            mu_sigma_P_df2=-0.5
            sigma_sigma_P_df2=0.5

            mu_phi_df=-1/3
            sigma_phi_df=0.2
        else:        
            mu_beta0_heat2=prior_values['mu_beta0_heat2']
            sigma_beta0_heat2=prior_values['sigma_beta0_heat2']
            mu_beta1_heat2=prior_values['mu_beta1_heat2']
            sigma_beta1_heat2=prior_values['sigma_beta1_heat2']
            mu_beta2_heat2=prior_values['mu_beta2_heat2']
            sigma_beta2_heat2=prior_values['sigma_beta2_heat2']
            mu_beta3_heat2=prior_values['mu_beta3_heat2']
            sigma_beta3_heat2=prior_values['sigma_beta3_heat2']
            mu_beta4_heat2=prior_values['mu_beta4_heat2']
            sigma_beta4_heat2=prior_values['sigma_beta4_heat2']
            mu_beta5_heat2=prior_values['mu_beta5_heat2']
            sigma_beta5_heat2=prior_values['sigma_beta5_heat2']

            mu_beta6_heat2=prior_values['mu_beta6_heat2']
            sigma_beta6_heat2=prior_values['sigma_beta6_heat2']
            mu_beta7_heat2=prior_values['mu_beta7_heat2']
            sigma_beta7_heat2=prior_values['sigma_beta7_heat2']
            mu_beta8_heat2=prior_values['mu_beta8_heat2']
            sigma_beta8_heat2=prior_values['sigma_beta8_heat2']
            mu_beta9_heat2=prior_values['mu_beta9_heat2']
            sigma_beta9_heat2=prior_values['sigma_beta9_heat2']
            mu_beta10_heat2=prior_values['mu_beta10_heat2']
            sigma_beta10_heat2=prior_values['sigma_beta10_heat2']


            mu_sigma_P_heat2=prior_values['mu_sigma_P_heat2_']
            sigma_sigma_P_heat2=prior_values['sigma_sigma_P_heat2_']

            # defrost control
            # mu_beta0_df2=prior_values['mu_beta0_df2']
            # sigma_beta0_df2=prior_values['sigma_beta0_df2']
            # mu_beta1_df2=prior_values['mu_beta1_df2_']
            # sigma_beta1_df2=prior_values['sigma_beta1_df2_']

            mu_beta0_df2=prior_values['mu_beta0_df2']
            sigma_beta0_df2=prior_values['sigma_beta0_df2']
            mu_beta1_df2=prior_values['mu_beta1_df2']
            sigma_beta1_df2=prior_values['sigma_beta1_df2']
            mu_beta2_df2=prior_values['mu_beta2_df2']
            sigma_beta2_df2=prior_values['sigma_beta2_df2']
            mu_beta3_df2=prior_values['mu_beta3_df2']
            sigma_beta3_df2=prior_values['sigma_beta3_df2']
            mu_beta4_df2=prior_values['mu_beta4_df2']
            sigma_beta4_df2=prior_values['sigma_beta4_df2']
            mu_beta5_df2=prior_values['mu_beta5_df2']
            sigma_beta5_df2=prior_values['sigma_beta5_df2']

            mu_beta6_df2=prior_values['mu_beta6_df2']
            sigma_beta6_df2=prior_values['sigma_beta6_df2']
            mu_beta7_df2=prior_values['mu_beta7_df2']
            sigma_beta7_df2=prior_values['sigma_beta7_df2']
            mu_beta8_df2=prior_values['mu_beta8_df2']
            sigma_beta8_df2=prior_values['sigma_beta8_df2']
            mu_beta9_df2=prior_values['mu_beta9_df2']
            sigma_beta9_df2=prior_values['sigma_beta9_df2']
            mu_beta10_df2=prior_values['mu_beta10_df2']
            sigma_beta10_df2=prior_values['sigma_beta10_df2']

            mu_sigma_P_df2=prior_values['mu_sigma_P_df2_']
            sigma_sigma_P_df2=prior_values['sigma_sigma_P_df2_']

            mu_phi_df=prior_values['mu_phi_df']
            sigma_phi_df=prior_values['sigma_phi_df']
    else:
        i_heat2_all=np.zeros_like(T_out)
        n_i_heat2_all=0


    if ('i_cool1' in input_values.keys()):
        # 1st stage heat pump cooling.
        i_cool1=input_values['i_cool1']
        n_i_cool1=np.nansum(i_cool1>0).astype("int")
        if prior_values is None:
            # mu_beta0_cool1=0.5
            # sigma_beta0_cool1=0.5
            # mu_beta1_cool1=-3.0
            # sigma_beta1_cool1=1.8
            mu_beta0_cool1=0.0
            sigma_beta0_cool1=0.25
            mu_beta1_cool1=0.0
            sigma_beta1_cool1=0.25
            mu_beta2_cool1=0.0
            sigma_beta2_cool1=0.25
            mu_beta3_cool1=0.0
            sigma_beta3_cool1=0.25
            mu_beta4_cool1=0.0
            sigma_beta4_cool1=0.25
            mu_beta5_cool1=0.0
            sigma_beta5_cool1=0.25

            mu_beta6_cool1=0.0
            sigma_beta6_cool1=0.25
            mu_beta7_cool1=0.0
            sigma_beta7_cool1=0.25
            mu_beta8_cool1=0.0
            sigma_beta8_cool1=0.25
            mu_beta9_cool1=0.0
            sigma_beta9_cool1=0.25
            mu_beta10_cool1=0.0
            sigma_beta10_cool1=0.25


            mu_sigma_P_cool1=-0.5
            sigma_sigma_P_cool1=0.5
        elif not('mu_beta0_cool1' in prior_values.keys()):
            # new training of mu_beta0_cool1
            # mu_beta0_cool1=0.5
            # sigma_beta0_cool1=.5
            # mu_beta1_cool1=-3.0
            # sigma_beta1_cool1=1.8
            mu_beta0_cool1=0.0
            sigma_beta0_cool1=0.25
            mu_beta1_cool1=0.0
            sigma_beta1_cool1=0.25
            mu_beta2_cool1=0.0
            sigma_beta2_cool1=0.25
            mu_beta3_cool1=0.0
            sigma_beta3_cool1=0.25
            mu_beta4_cool1=0.0
            sigma_beta4_cool1=0.25
            mu_beta5_cool1=0.0
            sigma_beta5_cool1=0.25

            mu_beta6_cool1=0.0
            sigma_beta6_cool1=0.25
            mu_beta7_cool1=0.0
            sigma_beta7_cool1=0.25
            mu_beta8_cool1=0.0
            sigma_beta8_cool1=0.25
            mu_beta9_cool1=0.0
            sigma_beta9_cool1=0.25
            mu_beta10_cool1=0.0
            sigma_beta10_cool1=0.25


            mu_sigma_P_cool1=-0.5
            sigma_sigma_P_cool1=0.5

            # sigma_P_cool1=0.05
        else:
            print("updating cool")
            # mu_beta0_cool1=prior_values['mu_beta0_cool1']
            # sigma_beta0_cool1=prior_values['sigma_beta0_cool1']
            # mu_beta1_cool1=prior_values['mu_beta1_cool1_']
            # sigma_beta1_cool1=prior_values['sigma_beta1_cool1_']
            mu_beta0_cool1=prior_values['mu_beta0_cool1']
            sigma_beta0_cool1=prior_values['sigma_beta0_cool1']
            mu_beta1_cool1=prior_values['mu_beta1_cool1']
            sigma_beta1_cool1=prior_values['sigma_beta1_cool1']
            mu_beta2_cool1=prior_values['mu_beta2_cool1']
            sigma_beta2_cool1=prior_values['sigma_beta2_cool1']
            mu_beta3_cool1=prior_values['mu_beta3_cool1']
            sigma_beta3_cool1=prior_values['sigma_beta3_cool1']
            mu_beta4_cool1=prior_values['mu_beta4_cool1']
            sigma_beta4_cool1=prior_values['sigma_beta4_cool1']
            mu_beta5_cool1=prior_values['mu_beta5_cool1']
            sigma_beta5_cool1=prior_values['sigma_beta5_cool1']

            mu_beta6_cool1=prior_values['mu_beta6_cool1']
            sigma_beta6_cool1=prior_values['sigma_beta6_cool1']
            mu_beta7_cool1=prior_values['mu_beta7_cool1']
            sigma_beta7_cool1=prior_values['sigma_beta7_cool1']
            mu_beta8_cool1=prior_values['mu_beta8_cool1']
            sigma_beta8_cool1=prior_values['sigma_beta8_cool1']
            mu_beta9_cool1=prior_values['mu_beta9_cool1']
            sigma_beta9_cool1=prior_values['sigma_beta9_cool1']
            mu_beta10_cool1=prior_values['mu_beta10_cool1']
            sigma_beta10_cool1=prior_values['sigma_beta10_cool1']


            mu_sigma_P_cool1=prior_values['mu_sigma_P_cool1_']
            sigma_sigma_P_cool1=prior_values['sigma_sigma_P_cool1_']
    else:
        i_cool1=np.zeros_like(T_out)
        n_i_cool1=0

    if ('i_cool2' in input_values.keys()):
        # 1st stage heat pump cooling.
        i_cool2=input_values['i_cool2']
        n_i_cool2=np.nansum(i_cool2>0).astype("int")
        if prior_values is None:
            # mu_beta0_cool2=0.5
            # sigma_beta0_cool2=0.5
            # mu_beta1_cool2=-3.0
            # sigma_beta1_cool2=1.8
            mu_beta0_cool2=0.0
            sigma_beta0_cool2=0.25
            mu_beta1_cool2=0.0
            sigma_beta1_cool2=0.25
            mu_beta2_cool2=0.0
            sigma_beta2_cool2=0.25
            mu_beta3_cool2=0.0
            sigma_beta3_cool2=0.25
            mu_beta4_cool2=0.0
            sigma_beta4_cool2=0.25
            mu_beta5_cool2=0.0
            sigma_beta5_cool2=0.25

            mu_beta6_cool2=0.0
            sigma_beta6_cool2=0.25
            mu_beta7_cool2=0.0
            sigma_beta7_cool2=0.25
            mu_beta8_cool2=0.0
            sigma_beta8_cool2=0.25
            mu_beta9_cool2=0.0
            sigma_beta9_cool2=0.25
            mu_beta10_cool2=0.0
            sigma_beta10_cool2=0.25


            mu_sigma_P_cool2=-0.5
            sigma_sigma_P_cool2=0.5
        elif not('mu_beta0_cool2' in prior_values.keys()):
            # new training of mu_beta0_cool2
            # mu_beta0_cool2=0.5
            # sigma_beta0_cool2=.5
            # mu_beta1_cool2=-3.0
            # sigma_beta1_cool2=1.8
            mu_beta0_cool2=0.0
            sigma_beta0_cool2=0.25
            mu_beta1_cool2=0.0
            sigma_beta1_cool2=0.25
            mu_beta2_cool2=0.0
            sigma_beta2_cool2=0.25
            mu_beta3_cool2=0.0
            sigma_beta3_cool2=0.25
            mu_beta4_cool2=0.0
            sigma_beta4_cool2=0.25
            mu_beta5_cool2=0.0
            sigma_beta5_cool2=0.25

            mu_beta6_cool2=0.0
            sigma_beta6_cool2=0.25
            mu_beta7_cool2=0.0
            sigma_beta7_cool2=0.25
            mu_beta8_cool2=0.0
            sigma_beta8_cool2=0.25
            mu_beta9_cool2=0.0
            sigma_beta9_cool2=0.25
            mu_beta10_cool2=0.0
            sigma_beta10_cool2=0.25


            mu_sigma_P_cool2=-0.5
            sigma_sigma_P_cool2=0.5

            # sigma_P_cool2=0.05
        else:
            print("updating cool")
            # mu_beta0_cool2=prior_values['mu_beta0_cool2']
            # sigma_beta0_cool2=prior_values['sigma_beta0_cool2']
            # mu_beta1_cool2=prior_values['mu_beta1_cool2_']
            # sigma_beta1_cool2=prior_values['sigma_beta1_cool2_']
            mu_beta0_cool2=prior_values['mu_beta0_cool2']
            sigma_beta0_cool2=prior_values['sigma_beta0_cool2']
            mu_beta1_cool2=prior_values['mu_beta1_cool2']
            sigma_beta1_cool2=prior_values['sigma_beta1_cool2']
            mu_beta2_cool2=prior_values['mu_beta2_cool2']
            sigma_beta2_cool2=prior_values['sigma_beta2_cool2']
            mu_beta3_cool2=prior_values['mu_beta3_cool2']
            sigma_beta3_cool2=prior_values['sigma_beta3_cool2']
            mu_beta4_cool2=prior_values['mu_beta4_cool2']
            sigma_beta4_cool2=prior_values['sigma_beta4_cool2']
            mu_beta5_cool2=prior_values['mu_beta5_cool2']
            sigma_beta5_cool2=prior_values['sigma_beta5_cool2']

            mu_beta6_cool2=prior_values['mu_beta6_cool2']
            sigma_beta6_cool2=prior_values['sigma_beta6_cool2']
            mu_beta7_cool2=prior_values['mu_beta7_cool2']
            sigma_beta7_cool2=prior_values['sigma_beta7_cool2']
            mu_beta8_cool2=prior_values['mu_beta8_cool2']
            sigma_beta8_cool2=prior_values['sigma_beta8_cool2']
            mu_beta9_cool2=prior_values['mu_beta9_cool2']
            sigma_beta9_cool2=prior_values['sigma_beta9_cool2']
            mu_beta10_cool2=prior_values['mu_beta10_cool2']
            sigma_beta10_cool2=prior_values['sigma_beta10_cool2']


            mu_sigma_P_cool2=prior_values['mu_sigma_P_cool2_']
            sigma_sigma_P_cool2=prior_values['sigma_sigma_P_cool2_']
    else:
        i_cool2=np.zeros_like(T_out)
        n_i_cool2=0


    if ('i_aux1' in input_values.keys()):
        # 1st stage heat pump heating.
        i_aux1=input_values['i_aux1']
        n_i_aux1=np.nansum(i_aux1>0).astype("int")
        if prior_values is None:
            # mu_P_aux1=1.0
            # sigma_P_aux1=3.0
            mu_mu_P_aux1=0.5
            sigma_mu_P_aux1=1.0
            mu_sigma_P_aux1=-0.5
            sigma_sigma_P_aux1=0.5

        elif not('mu_mu_P_aux1' in prior_values.keys()):
            # new training of mu_aux1
            
            mu_mu_P_aux1=0.5
            sigma_mu_P_aux1=1.0
            mu_sigma_P_aux1=-0.5
            sigma_sigma_P_aux1=0.5

            #mu_P_aux1=1.0
            #sigma_P_aux1=3.0
        else:
            mu_mu_P_aux1=prior_values['mu_mu_P_aux1']
            sigma_mu_P_aux1=prior_values['sigma_mu_P_aux1']
            mu_sigma_P_aux1=prior_values['mu_sigma_P_aux1_']
            sigma_sigma_P_aux1=prior_values['sigma_sigma_P_aux1_']

    else:
        i_aux1=np.zeros_like(T_out)
        n_i_aux1=0

    if ('i_fan1' in input_values.keys()):
        # 1st stage heat pump heating.
        i_fan1=input_values['i_fan1']
        n_i_fan1=np.nansum(i_fan1>0).astype("int")
        if prior_values is None:
            mu_P_fan1=-3.0
            sigma_P_fan1=1.0
        elif not('mu_P_fan1_' in prior_values.keys()):
            # new training of mu_fan1
            mu_P_fan1=-3.0
            sigma_P_fan1=1.0
        else:
            mu_P_fan1=prior_values['mu_P_fan1_']
            sigma_P_fan1=prior_values['sigma_P_fan1_']
    else:
        i_fan1=np.zeros_like(T_out)
        n_i_fan1=0


    if prior_values is None:
        
        mu_mu_P_nonhc=-3.0
        sigma_mu_P_nonhc=0.5
        mu_sigma_P_nonhc=-0.5
        sigma_sigma_P_nonhc=0.5

        mu_sigma_P_net=-4.0
        sigma_sigma_P_net=1.5
    
    else:

        mu_mu_P_nonhc=prior_values['mu_mu_P_nonhc']
        sigma_mu_P_nonhc=prior_values['sigma_mu_P_nonhc']
        mu_sigma_P_nonhc=prior_values['mu_sigma_P_nonhc_']
        sigma_sigma_P_nonhc=prior_values['sigma_sigma_P_nonhc_']
        
        mu_sigma_P_net=prior_values['mu_sigma_P_net_']
        sigma_sigma_P_net=prior_values['sigma_sigma_P_net_']

    hc_disaggregation=pm.Model()

    with hc_disaggregation:
       
        # first staging heatpump operation
        P_heat1_base=tt.zeros((n_N))
        P_df1_base=tt.zeros((n_N))
        if n_i_heat1_all>0:
            # i_heat1, T_out            
            
            # 1 x, x2, x3, y, y2, y3, xy, xy2, x2y, x2y2
            beta0_heat1=pm.Normal("beta0_heat1",mu=mu_beta0_heat1,sigma=sigma_beta0_heat1,testval=np.random.normal(mu_beta0_heat1,sigma_beta0_heat1,1),shape=1)
            beta1_heat1=pm.Normal("beta1_heat1",mu=mu_beta1_heat1,sigma=sigma_beta1_heat1,testval=np.random.normal(mu_beta1_heat1,sigma_beta1_heat1,1),shape=1)
            beta2_heat1=pm.Normal("beta2_heat1",mu=mu_beta2_heat1,sigma=sigma_beta2_heat1,testval=np.random.normal(mu_beta2_heat1,sigma_beta2_heat1,1),shape=1)
            beta3_heat1=pm.Normal("beta3_heat1",mu=mu_beta3_heat1,sigma=sigma_beta3_heat1,testval=np.random.normal(mu_beta3_heat1,sigma_beta3_heat1,1),shape=1)
            beta4_heat1=pm.Normal("beta4_heat1",mu=mu_beta4_heat1,sigma=sigma_beta4_heat1,testval=np.random.normal(mu_beta4_heat1,sigma_beta4_heat1,1),shape=1)
            beta5_heat1=pm.Normal("beta5_heat1",mu=mu_beta5_heat1,sigma=sigma_beta5_heat1,testval=np.random.normal(mu_beta5_heat1,sigma_beta5_heat1,1),shape=1)

            beta6_heat1=pm.Normal("beta6_heat1",mu=mu_beta6_heat1,sigma=sigma_beta6_heat1,testval=np.random.normal(mu_beta6_heat1,sigma_beta6_heat1,1),shape=1)
            beta7_heat1=pm.Normal("beta7_heat1",mu=mu_beta7_heat1,sigma=sigma_beta7_heat1,testval=np.random.normal(mu_beta7_heat1,sigma_beta7_heat1,1),shape=1)
            beta8_heat1=pm.Normal("beta8_heat1",mu=mu_beta8_heat1,sigma=sigma_beta8_heat1,testval=np.random.normal(mu_beta8_heat1,sigma_beta8_heat1,1),shape=1)
            beta9_heat1=pm.Normal("beta9_heat1",mu=mu_beta9_heat1,sigma=sigma_beta9_heat1,testval=np.random.normal(mu_beta9_heat1,sigma_beta9_heat1,1),shape=1)
            beta10_heat1=pm.Normal("beta10_heat1",mu=mu_beta10_heat1,sigma=sigma_beta10_heat1,testval=np.random.normal(mu_beta10_heat1,sigma_beta10_heat1,1),shape=1)

            # creates RVs whenever there is operation.
            #mu_heat1=pm.Deterministic("mu_heat1",beta0_heat1+beta1_heat1*T_out[i_heat1>0])
            mu_heat1=pm.Deterministic("mu_heat1",beta0_heat1+\
                                                beta1_heat1*T_in[i_heat1_all>0]+\
                                                beta2_heat1*(T_in[i_heat1_all>0])**2+\
                                                beta3_heat1*(T_in[i_heat1_all>0])**3+\
                                                beta4_heat1*T_out[i_heat1_all>0]+\
                                                beta5_heat1*(T_out[i_heat1_all>0])**2+\
                                                beta6_heat1*(T_out[i_heat1_all>0])**3+\
                                                beta7_heat1*T_in[i_heat1_all>0]*(T_out[i_heat1_all>0])+\
                                                beta8_heat1*(T_in[i_heat1_all>0])*(T_out[i_heat1_all>0]**2)+\
                                                beta9_heat1*(T_in[i_heat1_all>0]**2)*(T_out[i_heat1_all>0])+\
                                                beta10_heat1*(T_in[i_heat1_all>0]**2)*(T_out[i_heat1_all>0]**2)\
                                                )

            sigma_P_heat1_=(pm.Normal('sigma_P_heat1_',mu=mu_sigma_P_heat1,sigma=sigma_sigma_P_heat1,testval=np.random.normal(mu_sigma_P_heat1,sigma_sigma_P_heat1,1),shape=1))
            sigma_P_heat1=pm.Deterministic("sigma_P_heat1",tt.nnet.softplus(sigma_P_heat1_))
            
            P_heat1_=pm.Normal("P_heat1_",mu=mu_heat1,sigma=sigma_P_heat1,shape=n_i_heat1_all)
            P_heat1_base=tt.inc_subtensor(P_heat1_base[i_heat1_all>0],tt.nnet.sigmoid(P_heat1_)*heat_max/net_max)

            phi_df=pm.Normal("phi_df",mu=mu_phi_df,sigma=sigma_phi_df,testval=np.random.normal(mu_phi_df,sigma_phi_df,1),shape=1)

            
            # creates RVs whenever there is operation.
            # mu_df1=pm.Deterministic("mu_df1",beta0_df1+beta1_df1*T_out[i_heat1>0])
            beta0_df1=pm.Normal("beta0_df1",mu=mu_beta0_df1,sigma=sigma_beta0_df1,testval=np.random.normal(mu_beta0_df1,sigma_beta0_df1,1),shape=1)
            beta1_df1=pm.Normal("beta1_df1",mu=mu_beta1_df1,sigma=sigma_beta1_df1,testval=np.random.normal(mu_beta1_df1,sigma_beta1_df1,1),shape=1)
            beta2_df1=pm.Normal("beta2_df1",mu=mu_beta2_df1,sigma=sigma_beta2_df1,testval=np.random.normal(mu_beta2_df1,sigma_beta2_df1,1),shape=1)
            beta3_df1=pm.Normal("beta3_df1",mu=mu_beta3_df1,sigma=sigma_beta3_df1,testval=np.random.normal(mu_beta3_df1,sigma_beta3_df1,1),shape=1)
            beta4_df1=pm.Normal("beta4_df1",mu=mu_beta4_df1,sigma=sigma_beta4_df1,testval=np.random.normal(mu_beta4_df1,sigma_beta4_df1,1),shape=1)
            beta5_df1=pm.Normal("beta5_df1",mu=mu_beta5_df1,sigma=sigma_beta5_df1,testval=np.random.normal(mu_beta5_df1,sigma_beta5_df1,1),shape=1)

            beta6_df1=pm.Normal("beta6_df1",mu=mu_beta6_df1,sigma=sigma_beta6_df1,testval=np.random.normal(mu_beta6_df1,sigma_beta6_df1,1),shape=1)
            beta7_df1=pm.Normal("beta7_df1",mu=mu_beta7_df1,sigma=sigma_beta7_df1,testval=np.random.normal(mu_beta7_df1,sigma_beta7_df1,1),shape=1)
            beta8_df1=pm.Normal("beta8_df1",mu=mu_beta8_df1,sigma=sigma_beta8_df1,testval=np.random.normal(mu_beta8_df1,sigma_beta8_df1,1),shape=1)
            beta9_df1=pm.Normal("beta9_df1",mu=mu_beta9_df1,sigma=sigma_beta9_df1,testval=np.random.normal(mu_beta9_df1,sigma_beta9_df1,1),shape=1)
            beta10_df1=pm.Normal("beta10_df1",mu=mu_beta10_df1,sigma=sigma_beta10_df1,testval=np.random.normal(mu_beta10_df1,sigma_beta10_df1,1),shape=1)

            mu_df1=pm.Deterministic("mu_df1",beta0_df1+\
                                                beta1_df1*T_in[i_heat1_all>0]+\
                                                beta2_df1*(T_in[i_heat1_all>0])**2+\
                                                beta3_df1*(T_in[i_heat1_all>0])**3+\
                                                beta4_df1*T_out[i_heat1_all>0]+\
                                                beta5_df1*(T_out[i_heat1_all>0])**2+\
                                                beta6_df1*(T_out[i_heat1_all>0])**3+\
                                                beta7_df1*T_in[i_heat1_all>0]*(T_out[i_heat1_all>0])+\
                                                beta8_df1*(T_in[i_heat1_all>0])*(T_out[i_heat1_all>0]**2)+\
                                                beta9_df1*(T_in[i_heat1_all>0]**2)*(T_out[i_heat1_all>0])+\
                                                beta10_df1*(T_in[i_heat1_all>0]**2)*(T_out[i_heat1_all>0]**2)\
                                                )
            
            sigma_P_df1_=(pm.Normal('sigma_P_df1_',mu=mu_sigma_P_df1,sigma=sigma_sigma_P_df1,testval=np.random.normal(mu_sigma_P_df1,sigma_sigma_P_df1,1),shape=1))
            sigma_P_df1=pm.Deterministic("sigma_P_df1",tt.nnet.softplus(sigma_P_df1_))            
            #P_df1_=pm.Normal("P_df1_",mu=mu_df1,sigma=sigma_P_df1,shape=n_i_df1)
            #P_df1_base=tt.inc_subtensor(P_df1_base[i_df1>0],tt.exp(P_df1_))

            P_df1_=pm.Normal("P_df1_",mu=mu_df1,sigma=sigma_P_df1,shape=n_i_heat1_all)
            P_df1_base=tt.inc_subtensor(P_df1_base[i_heat1_all>0],tt.nnet.sigmoid(P_df1_)*df_max/net_max)
            i_df1=pm.Deterministic("i_df1",tt.iround(reverse_logistic(x=i_heat1_all,x0=phi_df))*i_heat1_all)
            i_heat1=pm.Deterministic("i_heat1",tt.abs_(tt.iround(reverse_logistic(x=i_heat1_all,x0=phi_df))-1)*i_heat1_all)
        else:
            i_df1=np.zeros_like(T_out)
            i_heat1=np.zeros_like(T_out)
        P_heat1=pm.Deterministic("P_heat1",P_heat1_base)
        P_df1=pm.Deterministic("P_df1",P_df1_base)

        # first staging heatpump operation
        P_heat2_base=tt.zeros((n_N))
        P_df2_base=tt.zeros((n_N))
        if n_i_heat2_all>0:
            # i_heat2, T_out            
            # beta0_heat2=pm.Normal("beta0_heat2",mu=mu_beta0_heat2,sigma=sigma_beta0_heat2,testval=np.random.normal(mu_beta0_heat2,sigma_beta0_heat2,1),shape=1)
            # beta1_heat2_=pm.Normal("beta1_heat2_",mu=mu_beta1_heat2,sigma=sigma_beta1_heat2,testval=np.random.normal(mu_beta1_heat2,sigma_beta1_heat2,1),shape=1)
            # beta1_heat2=pm.Deterministic("beta1_heat2",tt.nnet.softplus(beta1_heat2_))
            
            # 1 x, x2, x3, y, y2, y3, xy, xy2, x2y, x2y2
            beta0_heat2=pm.Normal("beta0_heat2",mu=mu_beta0_heat2,sigma=sigma_beta0_heat2,testval=np.random.normal(mu_beta0_heat2,sigma_beta0_heat2,1),shape=1)
            beta1_heat2=pm.Normal("beta1_heat2",mu=mu_beta1_heat2,sigma=sigma_beta1_heat2,testval=np.random.normal(mu_beta1_heat2,sigma_beta1_heat2,1),shape=1)
            beta2_heat2=pm.Normal("beta2_heat2",mu=mu_beta2_heat2,sigma=sigma_beta2_heat2,testval=np.random.normal(mu_beta2_heat2,sigma_beta2_heat2,1),shape=1)
            beta3_heat2=pm.Normal("beta3_heat2",mu=mu_beta3_heat2,sigma=sigma_beta3_heat2,testval=np.random.normal(mu_beta3_heat2,sigma_beta3_heat2,1),shape=1)
            beta4_heat2=pm.Normal("beta4_heat2",mu=mu_beta4_heat2,sigma=sigma_beta4_heat2,testval=np.random.normal(mu_beta4_heat2,sigma_beta4_heat2,1),shape=1)
            beta5_heat2=pm.Normal("beta5_heat2",mu=mu_beta5_heat2,sigma=sigma_beta5_heat2,testval=np.random.normal(mu_beta5_heat2,sigma_beta5_heat2,1),shape=1)

            beta6_heat2=pm.Normal("beta6_heat2",mu=mu_beta6_heat2,sigma=sigma_beta6_heat2,testval=np.random.normal(mu_beta6_heat2,sigma_beta6_heat2,1),shape=1)
            beta7_heat2=pm.Normal("beta7_heat2",mu=mu_beta7_heat2,sigma=sigma_beta7_heat2,testval=np.random.normal(mu_beta7_heat2,sigma_beta7_heat2,1),shape=1)
            beta8_heat2=pm.Normal("beta8_heat2",mu=mu_beta8_heat2,sigma=sigma_beta8_heat2,testval=np.random.normal(mu_beta8_heat2,sigma_beta8_heat2,1),shape=1)
            beta9_heat2=pm.Normal("beta9_heat2",mu=mu_beta9_heat2,sigma=sigma_beta9_heat2,testval=np.random.normal(mu_beta9_heat2,sigma_beta9_heat2,1),shape=1)
            beta10_heat2=pm.Normal("beta10_heat2",mu=mu_beta10_heat2,sigma=sigma_beta10_heat2,testval=np.random.normal(mu_beta10_heat2,sigma_beta10_heat2,1),shape=1)

            # creates RVs whenever there is operation.
            #mu_heat2=pm.Deterministic("mu_heat2",beta0_heat2+beta1_heat2*T_out[i_heat2>0])
            mu_heat2=pm.Deterministic("mu_heat2",beta0_heat2+\
                                                beta1_heat2*T_in[i_heat2_all>0]+\
                                                beta2_heat2*(T_in[i_heat2_all>0])**2+\
                                                beta3_heat2*(T_in[i_heat2_all>0])**3+\
                                                beta4_heat2*T_out[i_heat2_all>0]+\
                                                beta5_heat2*(T_out[i_heat2_all>0])**2+\
                                                beta6_heat2*(T_out[i_heat2_all>0])**3+\
                                                beta7_heat2*T_in[i_heat2_all>0]*(T_out[i_heat2_all>0])+\
                                                beta8_heat2*(T_in[i_heat2_all>0])*(T_out[i_heat2_all>0]**2)+\
                                                beta9_heat2*(T_in[i_heat2_all>0]**2)*(T_out[i_heat2_all>0])+\
                                                beta10_heat2*(T_in[i_heat2_all>0]**2)*(T_out[i_heat2_all>0]**2)\
                                                )

            sigma_P_heat2_=(pm.Normal('sigma_P_heat2_',mu=mu_sigma_P_heat2,sigma=sigma_sigma_P_heat2,testval=np.random.normal(mu_sigma_P_heat2,sigma_sigma_P_heat2,1),shape=1))
            sigma_P_heat2=pm.Deterministic("sigma_P_heat2",tt.nnet.softplus(sigma_P_heat2_))
            
            P_heat2_=pm.Normal("P_heat2_",mu=mu_heat2,sigma=sigma_P_heat2,shape=n_i_heat2_all)
            P_heat2_base=tt.inc_subtensor(P_heat2_base[i_heat2_all>0],tt.nnet.sigmoid(P_heat2_)*heat_max/net_max)

            phi_df=pm.Normal("phi_df",mu=mu_phi_df,sigma=sigma_phi_df,testval=np.random.normal(mu_phi_df,sigma_phi_df,1),shape=1)

            #beta0_df2=pm.Normal("beta0_df2",mu=mu_beta0_df2,sigma=sigma_beta0_df2,testval=np.random.normal(mu_beta0_df2,sigma_beta0_df2,1),shape=1)
            #beta1_df2_=pm.Normal("beta1_df2_",mu=mu_beta1_df2,sigma=sigma_beta1_df2,testval=np.random.normal(mu_beta1_df2,sigma_beta1_df2,1),shape=1)
            #beta1_df2=pm.Deterministic("beta1_df2",tt.nnet.softplus(beta1_df2_)*-1.)
            # creates RVs whenever there is operation.
            # mu_df2=pm.Deterministic("mu_df2",beta0_df2+beta1_df2*T_out[i_heat2>0])
            beta0_df2=pm.Normal("beta0_df2",mu=mu_beta0_df2,sigma=sigma_beta0_df2,testval=np.random.normal(mu_beta0_df2,sigma_beta0_df2,1),shape=1)
            beta1_df2=pm.Normal("beta1_df2",mu=mu_beta1_df2,sigma=sigma_beta1_df2,testval=np.random.normal(mu_beta1_df2,sigma_beta1_df2,1),shape=1)
            beta2_df2=pm.Normal("beta2_df2",mu=mu_beta2_df2,sigma=sigma_beta2_df2,testval=np.random.normal(mu_beta2_df2,sigma_beta2_df2,1),shape=1)
            beta3_df2=pm.Normal("beta3_df2",mu=mu_beta3_df2,sigma=sigma_beta3_df2,testval=np.random.normal(mu_beta3_df2,sigma_beta3_df2,1),shape=1)
            beta4_df2=pm.Normal("beta4_df2",mu=mu_beta4_df2,sigma=sigma_beta4_df2,testval=np.random.normal(mu_beta4_df2,sigma_beta4_df2,1),shape=1)
            beta5_df2=pm.Normal("beta5_df2",mu=mu_beta5_df2,sigma=sigma_beta5_df2,testval=np.random.normal(mu_beta5_df2,sigma_beta5_df2,1),shape=1)

            beta6_df2=pm.Normal("beta6_df2",mu=mu_beta6_df2,sigma=sigma_beta6_df2,testval=np.random.normal(mu_beta6_df2,sigma_beta6_df2,1),shape=1)
            beta7_df2=pm.Normal("beta7_df2",mu=mu_beta7_df2,sigma=sigma_beta7_df2,testval=np.random.normal(mu_beta7_df2,sigma_beta7_df2,1),shape=1)
            beta8_df2=pm.Normal("beta8_df2",mu=mu_beta8_df2,sigma=sigma_beta8_df2,testval=np.random.normal(mu_beta8_df2,sigma_beta8_df2,1),shape=1)
            beta9_df2=pm.Normal("beta9_df2",mu=mu_beta9_df2,sigma=sigma_beta9_df2,testval=np.random.normal(mu_beta9_df2,sigma_beta9_df2,1),shape=1)
            beta10_df2=pm.Normal("beta10_df2",mu=mu_beta10_df2,sigma=sigma_beta10_df2,testval=np.random.normal(mu_beta10_df2,sigma_beta10_df2,1),shape=1)

            mu_df2=pm.Deterministic("mu_df2",beta0_df2+\
                                                beta1_df2*T_in[i_heat2_all>0]+\
                                                beta2_df2*(T_in[i_heat2_all>0])**2+\
                                                beta3_df2*(T_in[i_heat2_all>0])**3+\
                                                beta4_df2*T_out[i_heat2_all>0]+\
                                                beta5_df2*(T_out[i_heat2_all>0])**2+\
                                                beta6_df2*(T_out[i_heat2_all>0])**3+\
                                                beta7_df2*T_in[i_heat2_all>0]*(T_out[i_heat2_all>0])+\
                                                beta8_df2*(T_in[i_heat2_all>0])*(T_out[i_heat2_all>0]**2)+\
                                                beta9_df2*(T_in[i_heat2_all>0]**2)*(T_out[i_heat2_all>0])+\
                                                beta10_df2*(T_in[i_heat2_all>0]**2)*(T_out[i_heat2_all>0]**2)\
                                                )
            
            sigma_P_df2_=(pm.Normal('sigma_P_df2_',mu=mu_sigma_P_df2,sigma=sigma_sigma_P_df2,testval=np.random.normal(mu_sigma_P_df2,sigma_sigma_P_df2,1),shape=1))
            sigma_P_df2=pm.Deterministic("sigma_P_df2",tt.nnet.softplus(sigma_P_df2_))            
            #P_df2_=pm.Normal("P_df2_",mu=mu_df2,sigma=sigma_P_df2,shape=n_i_df2)
            #P_df2_base=tt.inc_subtensor(P_df2_base[i_df2>0],tt.exp(P_df2_))

            P_df2_=pm.Normal("P_df2_",mu=mu_df2,sigma=sigma_P_df2,shape=n_i_heat2_all)
            P_df2_base=tt.inc_subtensor(P_df2_base[i_heat2_all>0],tt.nnet.sigmoid(P_df2_)*df_max/net_max)
            i_df2=pm.Deterministic("i_df2",tt.iround(reverse_logistic(x=i_heat2_all,x0=phi_df))*i_heat2_all)
            i_heat2=pm.Deterministic("i_heat2",tt.abs_(tt.iround(reverse_logistic(x=i_heat2_all,x0=phi_df))-1)*i_heat2_all)
        else:
            i_df2=np.zeros_like(T_out)
            i_heat2=np.zeros_like(T_out)
        P_heat2=pm.Deterministic("P_heat2",P_heat2_base)
        P_df2=pm.Deterministic("P_df2",P_df2_base)


        # first stage cooling operation
        P_cool1_base=tt.zeros((n_N))
        if n_i_cool1>0:
            # beta0_cool1=pm.Normal("beta0_cool1",mu=mu_beta0_cool1,sigma=sigma_beta0_cool1,testval=np.random.normal(mu_beta0_cool1,sigma_beta0_cool1,1),shape=1)
            # beta1_cool1_=pm.Normal("beta1_cool1_",mu=mu_beta1_cool1,sigma=sigma_beta1_cool1,testval=np.random.normal(mu_beta1_cool1,sigma_beta1_cool1,1),shape=1)
            # beta1_cool1=pm.Deterministic("beta1_cool1",tt.nnet.softplus(beta1_cool1_))
            # creates RVs whenever there is operation.
            # P_cool1_=pm.Deterministic("P_cool1_",beta0_cool1+beta1_cool1*T_out[i_cool1>1])
            # mu_cool1=pm.Deterministic("mu_cool1",beta0_cool1+beta1_cool1*T_out[i_cool1>0])

            beta0_cool1=pm.Normal("beta0_cool1",mu=mu_beta0_cool1,sigma=sigma_beta0_cool1,testval=np.random.normal(mu_beta0_cool1,sigma_beta0_cool1,1),shape=1)
            beta1_cool1=pm.Normal("beta1_cool1",mu=mu_beta1_cool1,sigma=sigma_beta1_cool1,testval=np.random.normal(mu_beta1_cool1,sigma_beta1_cool1,1),shape=1)
            beta2_cool1=pm.Normal("beta2_cool1",mu=mu_beta2_cool1,sigma=sigma_beta2_cool1,testval=np.random.normal(mu_beta2_cool1,sigma_beta2_cool1,1),shape=1)
            beta3_cool1=pm.Normal("beta3_cool1",mu=mu_beta3_cool1,sigma=sigma_beta3_cool1,testval=np.random.normal(mu_beta3_cool1,sigma_beta3_cool1,1),shape=1)
            beta4_cool1=pm.Normal("beta4_cool1",mu=mu_beta4_cool1,sigma=sigma_beta4_cool1,testval=np.random.normal(mu_beta4_cool1,sigma_beta4_cool1,1),shape=1)
            beta5_cool1=pm.Normal("beta5_cool1",mu=mu_beta5_cool1,sigma=sigma_beta5_cool1,testval=np.random.normal(mu_beta5_cool1,sigma_beta5_cool1,1),shape=1)

            beta6_cool1=pm.Normal("beta6_cool1",mu=mu_beta6_cool1,sigma=sigma_beta6_cool1,testval=np.random.normal(mu_beta6_cool1,sigma_beta6_cool1,1),shape=1)
            beta7_cool1=pm.Normal("beta7_cool1",mu=mu_beta7_cool1,sigma=sigma_beta7_cool1,testval=np.random.normal(mu_beta7_cool1,sigma_beta7_cool1,1),shape=1)
            beta8_cool1=pm.Normal("beta8_cool1",mu=mu_beta8_cool1,sigma=sigma_beta8_cool1,testval=np.random.normal(mu_beta8_cool1,sigma_beta8_cool1,1),shape=1)
            beta9_cool1=pm.Normal("beta9_cool1",mu=mu_beta9_cool1,sigma=sigma_beta9_cool1,testval=np.random.normal(mu_beta9_cool1,sigma_beta9_cool1,1),shape=1)
            beta10_cool1=pm.Normal("beta10_cool1",mu=mu_beta10_cool1,sigma=sigma_beta10_cool1,testval=np.random.normal(mu_beta10_cool1,sigma_beta10_cool1,1),shape=1)

            mu_cool1=pm.Deterministic("mu_cool1",beta0_cool1+\
                                                beta1_cool1*wb_in[i_cool1>0]+\
                                                beta2_cool1*(wb_in[i_cool1>0])**2+\
                                                beta3_cool1*(wb_in[i_cool1>0])**3+\
                                                beta4_cool1*T_out[i_cool1>0]+\
                                                beta5_cool1*(T_out[i_cool1>0])**2+\
                                                beta6_cool1*(T_out[i_cool1>0])**3+\
                                                beta7_cool1*(wb_in[i_cool1>0])*(T_out[i_cool1>0])+\
                                                beta8_cool1*(wb_in[i_cool1>0])*(T_out[i_cool1>0]**2)+\
                                                beta9_cool1*(wb_in[i_cool1>0]**2)*(T_out[i_cool1>0])+\
                                                beta10_cool1*(wb_in[i_cool1>0]**2)*(T_out[i_cool1>0]**2)\
                                                )
            
            sigma_P_cool1_=(pm.Normal('sigma_P_cool1_',mu=mu_sigma_P_cool1,sigma=sigma_sigma_P_cool1,testval=np.random.normal(mu_sigma_P_cool1,sigma_sigma_P_cool1,1),shape=1))
            sigma_P_cool1=pm.Deterministic("sigma_P_cool1",tt.nnet.softplus(sigma_P_cool1_))
            
            # P_cool1_=pm.Normal("P_cool1_",mu=mu_cool1,sigma=sigma_P_cool1,shape=n_i_cool1)
            # P_cool1_base=tt.inc_subtensor(P_cool1_base[i_cool1>0],tt.exp(P_cool1_))
            

            P_cool1_=pm.Normal("P_cool1_",mu=mu_cool1,sigma=sigma_P_cool1,shape=n_i_cool1)
            P_cool1_base=tt.inc_subtensor(P_cool1_base[i_cool1>0],tt.nnet.sigmoid(P_cool1_)*cool_max/net_max)

            
            #P_cool1_=pm.Beta("P_cool1_",mu=mu_cool1,sigma=sigma_P_cool1,shape=n_i_cool1)
            #P_cool1_base=tt.inc_subtensor(P_cool1_base[i_cool1>0],(P_cool1_))

        P_cool1=pm.Deterministic("P_cool1",P_cool1_base)

        # first stage cooling operation
        P_cool2_base=tt.zeros((n_N))
        if n_i_cool2>0:
            # beta0_cool2=pm.Normal("beta0_cool2",mu=mu_beta0_cool2,sigma=sigma_beta0_cool2,testval=np.random.normal(mu_beta0_cool2,sigma_beta0_cool2,1),shape=1)
            # beta1_cool2_=pm.Normal("beta1_cool2_",mu=mu_beta1_cool2,sigma=sigma_beta1_cool2,testval=np.random.normal(mu_beta1_cool2,sigma_beta1_cool2,1),shape=1)
            # beta1_cool2=pm.Deterministic("beta1_cool2",tt.nnet.softplus(beta1_cool2_))
            # creates RVs whenever there is operation.
            # P_cool2_=pm.Deterministic("P_cool2_",beta0_cool2+beta1_cool2*T_out[i_cool2>1])
            # mu_cool2=pm.Deterministic("mu_cool2",beta0_cool2+beta1_cool2*T_out[i_cool2>0])

            beta0_cool2=pm.Normal("beta0_cool2",mu=mu_beta0_cool2,sigma=sigma_beta0_cool2,testval=np.random.normal(mu_beta0_cool2,sigma_beta0_cool2,1),shape=1)
            beta1_cool2=pm.Normal("beta1_cool2",mu=mu_beta1_cool2,sigma=sigma_beta1_cool2,testval=np.random.normal(mu_beta1_cool2,sigma_beta1_cool2,1),shape=1)
            beta2_cool2=pm.Normal("beta2_cool2",mu=mu_beta2_cool2,sigma=sigma_beta2_cool2,testval=np.random.normal(mu_beta2_cool2,sigma_beta2_cool2,1),shape=1)
            beta3_cool2=pm.Normal("beta3_cool2",mu=mu_beta3_cool2,sigma=sigma_beta3_cool2,testval=np.random.normal(mu_beta3_cool2,sigma_beta3_cool2,1),shape=1)
            beta4_cool2=pm.Normal("beta4_cool2",mu=mu_beta4_cool2,sigma=sigma_beta4_cool2,testval=np.random.normal(mu_beta4_cool2,sigma_beta4_cool2,1),shape=1)
            beta5_cool2=pm.Normal("beta5_cool2",mu=mu_beta5_cool2,sigma=sigma_beta5_cool2,testval=np.random.normal(mu_beta5_cool2,sigma_beta5_cool2,1),shape=1)

            beta6_cool2=pm.Normal("beta6_cool2",mu=mu_beta6_cool2,sigma=sigma_beta6_cool2,testval=np.random.normal(mu_beta6_cool2,sigma_beta6_cool2,1),shape=1)
            beta7_cool2=pm.Normal("beta7_cool2",mu=mu_beta7_cool2,sigma=sigma_beta7_cool2,testval=np.random.normal(mu_beta7_cool2,sigma_beta7_cool2,1),shape=1)
            beta8_cool2=pm.Normal("beta8_cool2",mu=mu_beta8_cool2,sigma=sigma_beta8_cool2,testval=np.random.normal(mu_beta8_cool2,sigma_beta8_cool2,1),shape=1)
            beta9_cool2=pm.Normal("beta9_cool2",mu=mu_beta9_cool2,sigma=sigma_beta9_cool2,testval=np.random.normal(mu_beta9_cool2,sigma_beta9_cool2,1),shape=1)
            beta10_cool2=pm.Normal("beta10_cool2",mu=mu_beta10_cool2,sigma=sigma_beta10_cool2,testval=np.random.normal(mu_beta10_cool2,sigma_beta10_cool2,1),shape=1)

            mu_cool2=pm.Deterministic("mu_cool2",beta0_cool2+\
                                                beta1_cool2*wb_in[i_cool2>0]+\
                                                beta2_cool2*(wb_in[i_cool2>0])**2+\
                                                beta3_cool2*(wb_in[i_cool2>0])**3+\
                                                beta4_cool2*T_out[i_cool2>0]+\
                                                beta5_cool2*(T_out[i_cool2>0])**2+\
                                                beta6_cool2*(T_out[i_cool2>0])**3+\
                                                beta7_cool2*(wb_in[i_cool2>0])*(T_out[i_cool2>0])+\
                                                beta8_cool2*(wb_in[i_cool2>0])*(T_out[i_cool2>0]**2)+\
                                                beta9_cool2*(wb_in[i_cool2>0]**2)*(T_out[i_cool2>0])+\
                                                beta10_cool2*(wb_in[i_cool2>0]**2)*(T_out[i_cool2>0]**2)\
                                                )
            
            sigma_P_cool2_=(pm.Normal('sigma_P_cool2_',mu=mu_sigma_P_cool2,sigma=sigma_sigma_P_cool2,testval=np.random.normal(mu_sigma_P_cool2,sigma_sigma_P_cool2,1),shape=1))
            sigma_P_cool2=pm.Deterministic("sigma_P_cool2",tt.nnet.softplus(sigma_P_cool2_))
            
            # P_cool2_=pm.Normal("P_cool2_",mu=mu_cool2,sigma=sigma_P_cool2,shape=n_i_cool2)
            # P_cool2_base=tt.inc_subtensor(P_cool2_base[i_cool2>0],tt.exp(P_cool2_))
            

            P_cool2_=pm.Normal("P_cool2_",mu=mu_cool2,sigma=sigma_P_cool2,shape=n_i_cool2)
            P_cool2_base=tt.inc_subtensor(P_cool2_base[i_cool2>0],tt.nnet.sigmoid(P_cool2_)*cool_max/net_max)

            
            #P_cool2_=pm.Beta("P_cool2_",mu=mu_cool2,sigma=sigma_P_cool2,shape=n_i_cool2)
            #P_cool2_base=tt.inc_subtensor(P_cool2_base[i_cool2>0],(P_cool2_))

        P_cool2=pm.Deterministic("P_cool2",P_cool2_base)


        P_aux1_base=tt.zeros((n_N))
        if n_i_aux1>0:
            mu_P_aux1=(pm.Normal('mu_P_aux1',mu=mu_mu_P_aux1,sigma=sigma_mu_P_aux1,testval=np.random.normal(mu_mu_P_aux1,sigma_mu_P_aux1,1),shape=1))
            sigma_P_aux1_=(pm.Normal('sigma_P_aux1_',mu=mu_sigma_P_aux1,sigma=sigma_sigma_P_aux1,testval=np.random.normal(mu_sigma_P_aux1,sigma_sigma_P_aux1,1),shape=1))
            sigma_P_aux1=pm.Deterministic("sigma_P_aux1",tt.nnet.softplus(sigma_P_aux1_))
            P_aux1_=pm.Normal("P_aux1_",mu=mu_P_aux1,sigma=sigma_P_aux1,shape=n_i_aux1)
            P_aux1_base=tt.inc_subtensor(P_aux1_base[i_aux1>0],tt.nnet.sigmoid(P_aux1_)*aux_max/net_max)
        P_aux1=pm.Deterministic("P_aux1",P_aux1_base)

        P_fan1_base=tt.zeros((n_N))
        if n_i_fan1>0:
            P_fan1_=pm.Normal("P_aux1_",mu=mu_P_fan1,sigma=sigma_P_fan1,shape=n_i_aux1)
            P_fan1_base=tt.inc_subtensor(P_fan1_base[i_fan1>0],tt.nnet.softplus(P_fan1_))
        P_fan1=pm.Deterministic("P_fan1",P_fan1_base)

        mu_P_nonhc=pm.Normal('mu_P_nonhc',mu=mu_mu_P_nonhc,sigma=sigma_mu_P_nonhc,testval=np.random.normal(mu_mu_P_nonhc,sigma_mu_P_nonhc,1),shape=1)
        sigma_P_nonhc_=pm.Normal('sigma_P_nonhc_',mu=mu_sigma_P_nonhc,sigma=sigma_sigma_P_nonhc,testval=np.random.normal(mu_sigma_P_nonhc,sigma_sigma_P_nonhc,1),shape=1)
        sigma_P_nonhc=pm.Deterministic("sigma_P_nonhc",tt.nnet.softplus(sigma_P_nonhc_))
        P_nonhc_=pm.Normal("P_nonhc_",mu=mu_P_nonhc,sigma=sigma_P_nonhc,shape=n_N)
        P_nonhc=pm.Deterministic("P_nonhc",tt.nnet.softplus(P_nonhc_))
        
        #P_hc=pm.Deterministic("P_hc",i_heat1*P_heat1+i_heat2*P_heat2 +i_cool1*P_cool1+i_cool2*P_cool2 + i_aux1*P_aux1 +i_df1*P_df1+i_df2*P_df2+P_fan1*i_fan1)
        # reverse_logistic(x,x0)
        
        P_hc=pm.Deterministic("P_hc",i_heat1*P_heat1+i_df1*P_df1 +i_cool1*P_cool1 +i_heat2*P_heat2+i_df2*P_df2 +i_cool2*P_cool2 + i_aux1*P_aux1 +P_fan1*i_fan1)
        
        mu_net=pm.Deterministic("mu_net",P_hc+ P_nonhc)
        
        sigma_P_net_=(pm.Normal('sigma_P_net_',mu=mu_sigma_P_net,sigma=sigma_sigma_P_net,testval=np.random.normal(mu_sigma_P_net,sigma_sigma_P_net,1),shape=1))
        sigma_P_net=pm.Deterministic("sigma_P_net",tt.nnet.softplus(sigma_P_net_))
        
        obs=pm.TruncatedNormal('obs', mu=mu_net, sigma=sigma_P_net, lower=0,observed=y_net)
        advi=pm.ADVI()

        return advi


