__all__=['trainig_loop','ResNIHCM']

import math
import pandas as pd
import os
import pickle
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim


import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO


import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader


import pyro
from pyro.optim import MultiStepLR, ExponentialLR
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO,TraceMeanField_ELBO

def trainig_loop(n_epochs, optimizer, model, loss_fn, train_loader,cuda=False,priors=None,prior_network=False,new_data=False,missing_data=False):
    
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device=="cpu":
            print("Cuda is not available. CPU is used.")
    else:
        device="cpu"

    model=model.to(device)
    svi = SVI(model.model, model.guide, optimizer, loss=loss_fn)
    loss_list=[]
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for  ix, (y_net, t_out,i_heat,i_heat_on,i_heat_off,i_cool,i_cool_on,i_cool_off,i_aux,i_aux_on,i_aux_off,i_heat_df) in enumerate(train_loader):
        
            y_net = y_net.to(device=device)  # <1>
            t_out = t_out.to(device=device)  # <1>
            i_heat = i_heat.to(device=device)  # <1>
            i_heat_on = i_heat_on.to(device=device)  # <1>
            i_heat_off = i_heat_off.to(device=device)  # <1>
            i_cool = i_cool.to(device=device)  # <1>
            i_cool_on = i_cool_on.to(device=device)  # <1>
            i_cool_off = i_cool_off.to(device=device)  # <1>
            i_aux = i_aux.to(device=device)  # <1>
            i_aux_on = i_aux_on.to(device=device)  # <1>
            i_aux_off = i_aux_off.to(device=device)  # <1>
            i_heat_df =i_heat_df.to(device=device)

            if priors is None:
                loss=svi.step(y_net=y_net, t_out=t_out,i_heat=i_heat,i_heat_on=i_heat_on,i_heat_off=i_heat_off,i_cool=i_cool,i_cool_on=i_cool_on,i_cool_off=i_cool_off,i_aux=i_aux,i_aux_on=i_aux_on,i_aux_off=i_aux_off,i_heat_df=i_heat_df,priors=None)
            else:
                # no model update. Just in-prior computation (actually, it doesn't exist)
                raise ValueError("Training is not in any case. check priors, new_data, prior_network params")
                
            loss_train += loss
        
        loss_list.append(loss_train / len(train_loader))
        if epoch == 1 or epoch % 5 == 0:
            
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
        if (epoch==1 or epoch%10==0) and (type(optimizer)==pyro.optim.lr_scheduler.PyroLRScheduler) :
            optimizer.step()
            #print(f'learning rate {next(iter(svi.optim.optim_objs.values())).get_last_lr()[0]}')
    return loss_list


class ResNIHCM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define dimensions
        
        # Use ELU see Murphy p.397
        self.elu=nn.ELU()
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()
        self.softplus=nn.Softplus()
    
    def calculate_concentration(self,mu,sigma):
        concentration_alpha=((1-mu)/(sigma**2)-1/mu)*(mu**2)
        concentration_beta=concentration_alpha*(1/mu-1)
        return concentration_alpha, concentration_beta

    def model(self, y_net, t_out,
                    i_heat,i_heat_on,i_heat_off,
                    i_cool,i_cool_on,i_cool_off,
                    i_aux,i_aux_on,i_aux_off,i_heat_df,
                    priors=None):
        # it is hard to generalize the process. 
        # we may have matrix, 

        # initial network
        self.batch_sz=t_out.shape[0]
        device=t_out.device
        
        if priors is None:
            add_noise=0 # no noise addition for priors
            noise_scale=0.01
            noise_mean=0
            priors={
                "mu_misc":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_misc~logN(-3,2.5) [0.0004,0.05,6.783]
                "sigma_misc":np.array([1.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_beta0_heat":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
                "sigma_beta0_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
                "mu_beta1_heat":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
                "sigma_beta1_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
                "sigma_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_heat~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
                "mu_heat_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_on~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
                "sigma_heat_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_heat_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_off~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
                "sigma_heat_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_beta0_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
                "sigma_beta0_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
                "mu_beta1_cool":np.array([-4.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
                "sigma_beta1_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
                "sigma_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_cool~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
                "mu_cool_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_on~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
                "sigma_cool_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_cool_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_off~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
                "sigma_cool_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_aux":np.array([-3.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
                "sigma_aux":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_heat_df":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
                "sigma_heat_df":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
                "mu_aux_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_on~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
                "sigma_aux_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_aux_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_off~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
                "sigma_aux_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
                # "mu_phi_df":np.array([-1/3])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # phi~N(-1/3,1/6) give [-2/3,-1/3,0] which is [-10,0,10] in real scale
                # "sigma_phi_df":np.array([1/6])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                # "mu_psi":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # psi~Beta(mu_psi=0.5,sigma_psi=1/12) # 0~1 flat 
                # "sigma_psi":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,     # 
                
                "mu_sigma_net":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,  # 
                "sigma_sigma_net":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise # 

            }
            
  
        mu_misc=torch.tensor(priors['mu_misc'],dtype=torch.float32).to(device)
        sigma_misc=torch.tensor(priors['sigma_misc'],dtype=torch.float32).to(device)
        
        mu_beta0_heat=torch.tensor(priors['mu_beta0_heat'],dtype=torch.float32).to(device)
        sigma_beta0_heat=torch.tensor(priors['sigma_beta0_heat'],dtype=torch.float32).to(device)
        mu_beta1_heat=torch.tensor(priors['mu_beta1_heat'],dtype=torch.float32).to(device)
        sigma_beta1_heat=torch.tensor(priors['sigma_beta1_heat'],dtype=torch.float32).to(device)
        sigma_heat=torch.tensor(priors['sigma_heat'],dtype=torch.float32).to(device)

        mu_heat_on=torch.tensor(priors['mu_heat_on'],dtype=torch.float32).to(device)
        sigma_heat_on=torch.tensor(priors['sigma_heat_on'],dtype=torch.float32).to(device)
        mu_heat_off=torch.tensor(priors['mu_heat_off'],dtype=torch.float32).to(device)
        sigma_heat_off=torch.tensor(priors['sigma_heat_off'],dtype=torch.float32).to(device)

        mu_beta0_cool=torch.tensor(priors['mu_beta0_cool'],dtype=torch.float32).to(device)
        sigma_beta0_cool=torch.tensor(priors['sigma_beta0_cool'],dtype=torch.float32).to(device)
        mu_beta1_cool=torch.tensor(priors['mu_beta1_cool'],dtype=torch.float32).to(device)
        sigma_beta1_cool=torch.tensor(priors['sigma_beta1_cool'],dtype=torch.float32).to(device)
        sigma_cool=torch.tensor(priors['sigma_cool'],dtype=torch.float32).to(device)

        mu_cool_on=torch.tensor(priors['mu_cool_on'],dtype=torch.float32).to(device)
        sigma_cool_on=torch.tensor(priors['sigma_cool_on'],dtype=torch.float32).to(device)
        mu_cool_off=torch.tensor(priors['mu_cool_off'],dtype=torch.float32).to(device)
        sigma_cool_off=torch.tensor(priors['sigma_cool_off'],dtype=torch.float32).to(device)

        mu_aux=torch.tensor(priors['mu_aux'],dtype=torch.float32).to(device)
        sigma_aux=torch.tensor(priors['sigma_aux'],dtype=torch.float32).to(device)

        mu_aux_on=torch.tensor(priors['mu_aux_on'],dtype=torch.float32).to(device)
        sigma_aux_on=torch.tensor(priors['sigma_aux_on'],dtype=torch.float32).to(device)
        mu_aux_off=torch.tensor(priors['mu_aux_off'],dtype=torch.float32).to(device)
        sigma_aux_off=torch.tensor(priors['sigma_aux_off'],dtype=torch.float32).to(device)


        mu_heat_df=torch.tensor(priors['mu_heat_df'],dtype=torch.float32).to(device)
        sigma_heat_df=torch.tensor(priors['sigma_heat_df'],dtype=torch.float32).to(device)

        #mu_phi_df=torch.tensor(priors['mu_phi_df'],dtype=torch.float32).to(device)
        #sigma_phi_df=torch.tensor(priors['sigma_phi_df'],dtype=torch.float32).to(device)
        
        #mu_psi=torch.tensor(priors['mu_psi'],dtype=torch.float32).to(device)
        #sigma_psi=torch.tensor(priors['sigma_psi'],dtype=torch.float32).to(device)

        mu_sigma_net=torch.tensor(priors['mu_sigma_net'],dtype=torch.float32).to(device)
        sigma_sigma_net=torch.tensor(priors['sigma_sigma_net'],dtype=torch.float32).to(device)
        
        
        E_misc=self.softplus(pyro.sample("E_misc",dist.Normal(mu_misc,sigma_misc).to_event(1)))
        

        # here mu_heat is not real scale. E_heat~LogNormal(mu_heat,sigma_heat)
        beta0_heat=pyro.sample("beta0_heat",dist.Normal(mu_beta0_heat,sigma_beta0_heat).to_event(1))
        beta1_heat=self.softplus(pyro.sample("beta1_heat",dist.Normal(mu_beta1_heat,sigma_beta1_heat).to_event(1)))
        mu_heat=pyro.deterministic("mu_heat",beta0_heat+beta1_heat*t_out)
        E_heat=self.softplus(pyro.sample("E_heat",dist.Normal(mu_heat,sigma_heat).to_event(1)))
        #print(f"E_heat shape is {E_heat.shape}")
        



        beta0_cool=pyro.sample("beta0_cool",dist.Normal(mu_beta0_cool,sigma_beta0_cool).to_event(1))
        beta1_cool=self.softplus(pyro.sample("beta1_cool",dist.Normal(mu_beta1_cool,sigma_beta1_cool).to_event(1)))
        mu_cool=pyro.deterministic("mu_cool",beta0_cool+beta1_cool*t_out)
        E_cool=self.softplus(pyro.sample("E_cool",dist.Normal(mu_cool,sigma_cool).to_event(1)))

        E_aux=self.softplus(pyro.sample("E_aux",dist.Normal(mu_aux,sigma_aux).to_event(1)))

        # phi_df=pyro.sample("phi_df",dist.Normal(mu_phi_df,sigma_phi_df).to_event(1))
        # mu_psi_alpha,mu_psi_beta=self.calculate_concentration(mu=mu_psi,sigma=sigma_psi)
        # psi=pyro.sample("psi",dist.Beta(concentration1=mu_psi_alpha ,concentration0=mu_psi_beta).to_event(1))
        
        eta_heat=i_heat.clone()
        mu_heat_on_alpha,mu_heat_on_beta=self.calculate_concentration(mu=mu_heat_on,sigma=sigma_heat_on)
        mu_heat_off_alpha,mu_heat_off_beta=self.calculate_concentration(mu=mu_heat_off,sigma=sigma_heat_off)
        
        # print(f'mu_heat_on_alpha is {mu_heat_on_alpha}')
        # print(f'mu_heat_on_beta is {mu_heat_on_beta}')

        eta_heat_on=pyro.sample("eta_heat_on",dist.Beta(concentration1=mu_heat_on_alpha ,concentration0=mu_heat_on_beta).to_event(1))
        eta_heat_off=pyro.sample("eta_heat_off",dist.Beta(concentration1=mu_heat_off_alpha ,concentration0=mu_heat_off_beta).to_event(1))
        eta_heat[i_heat_on==1]=eta_heat_on#[i_heat_on==1]
        eta_heat[i_heat_off==1]=eta_heat_off#[i_heat_off==1]
        
        eta_cool=i_cool.clone()
        mu_cool_on_alpha,mu_cool_on_beta=self.calculate_concentration(mu=mu_cool_on,sigma=sigma_cool_on)
        mu_cool_off_alpha,mu_cool_off_beta=self.calculate_concentration(mu=mu_cool_off,sigma=sigma_cool_off)
        eta_cool_on=pyro.sample("eta_cool_on",dist.Beta(concentration1=mu_cool_on_alpha ,concentration0=mu_cool_on_beta).to_event(1))
        eta_cool_off=pyro.sample("eta_cool_off",dist.Beta(concentration1=mu_cool_off_alpha ,concentration0=mu_cool_off_beta).to_event(1))
        eta_cool[i_cool_on==1]=eta_cool_on#[i_cool_on==1]
        eta_cool[i_cool_off==1]=eta_cool_off#[i_cool_off==1]
        
        eta_aux=i_aux.clone()
        mu_aux_on_alpha,mu_aux_on_beta=self.calculate_concentration(mu=mu_aux_on,sigma=sigma_aux_on)
        mu_aux_off_alpha,mu_aux_off_beta=self.calculate_concentration(mu=mu_aux_off,sigma=sigma_aux_off)
        eta_aux_on=pyro.sample("eta_aux_on",dist.Beta(concentration1=mu_aux_on_alpha ,concentration0=mu_aux_on_beta).to_event(1))
        eta_aux_off=pyro.sample("eta_aux_off",dist.Beta(concentration1=mu_aux_off_alpha ,concentration0=mu_aux_off_beta).to_event(1))
        eta_aux[i_aux_on==1]=eta_aux_on#[i_aux_on==1]
        eta_aux[i_aux_off==1]=eta_aux_off#[i_aux_off==1]

        E_heat_df=self.softplus(pyro.sample("E_heat_df",dist.Normal(mu_heat_df,sigma_heat_df).to_event(1)))
        #i_df=torch.zeros_like(i_heat).to(device)
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta.concentration1
        # concentration1 (float or Tensor) – 1st concentration parameter of the distribution (often referred to as alpha)
        # concentration0 (float or Tensor) – 2nd concentration parameter of the distribution (often referred to as beta)

        #with pyro.plate("Emisc", size=t_out.shape[0]):
        
        
        #i_df_on=pyro.sample("i_df_on",dist.Binomial(total_count=1,probs=psi))

        #i_df=torch.where((i_heat==torch.tensor(1,dtype=torch.float32))&(t_out<phi_df),i_df_on,i_df)
        


        y_nan=torch.any(torch.cat([torch.isnan(i_heat)[:,None],
                                torch.isnan(i_cool)[:,None],
                                torch.isnan(i_aux)[:,None],
                                torch.isnan(t_out)[:,None],
                                torch.isnan(i_heat_df)[:,None],
                                torch.isnan(y_net)[:,None]
                                ],dim=1),axis=1)

        #print(f'y_nan is {y_nan}')
        # print(f'eta_heat is {eta_heat}')
        # print(f'i_heat is {i_heat}')

        
        mu_net_=eta_heat*i_heat*E_heat+eta_cool*i_cool*E_cool+(eta_aux*i_aux)*E_aux+(i_heat_df)*E_heat_df+E_misc
        
        mu_net=pyro.deterministic("mu_net",mu_net_[~y_nan])
        sigma_net = self.softplus(pyro.sample("sigma_t_unit", dist.Normal(mu_sigma_net,sigma_sigma_net).to_event(1)))
        #print(f"sigma_net is {sigma_net}")
        y_net_=y_net.flatten()[~y_nan]

        with pyro.plate("data", size=mu_net.shape[0]):
            obs_net=pyro.sample("obs_net", dist.Normal(mu_net, sigma_net).to_event(1), obs=y_net_.flatten())   # .to_event(1)

        return mu_net,priors

    
    def guide(self, y_net, t_out,
                    i_heat,i_heat_on,i_heat_off,
                    i_cool,i_cool_on,i_cool_off,
                    i_aux,i_aux_on,i_aux_off,i_heat_df,
                    priors=None):
        
        
        self.batch_sz=t_out.shape[0]
        device=t_out.device
        
        if priors is None:
            # add noise  for priors
            add_noise=1
            noise_scale=0.001
            noise_mean=0
            priors={
                "mu_misc":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_misc~logN(-3,2.5) [0.0004,0.05,6.783]
                "sigma_misc":np.array([1.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_beta0_heat":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
                "sigma_beta0_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
                "mu_beta1_heat":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
                "sigma_beta1_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
                "sigma_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_heat~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
                "mu_heat_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_on~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
                "sigma_heat_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_heat_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_off~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
                "sigma_heat_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_beta0_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
                "sigma_beta0_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
                "mu_beta1_cool":np.array([-4.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
                "sigma_beta1_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
                "sigma_cool":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_cool~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
                "mu_cool_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_on~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
                "sigma_cool_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_cool_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_off~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
                "sigma_cool_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_aux":np.array([-3.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
                "sigma_aux":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

                "mu_heat_df":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
                "sigma_heat_df":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
                "mu_aux_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_on~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
                "sigma_aux_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
                "mu_aux_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_off~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
                "sigma_aux_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
                # "mu_phi_df":np.array([-1/3])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # phi~N(-1/3,1/6) give [-2/3,-1/3,0] which is [-10,0,10] in real scale
                # "sigma_phi_df":np.array([1/6])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                # "mu_psi":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # psi~Beta(mu_psi=0.5,sigma_psi=1/12) # 0~1 flat 
                # "sigma_psi":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,     # 
                
                "mu_sigma_net":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,  # 
                "sigma_sigma_net":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise # 

            }
            

        ################### params ########################33

        mu_misc=torch.tensor(priors['mu_misc'],dtype=torch.float32).to(device)
        sigma_misc=torch.tensor(priors['sigma_misc'],dtype=torch.float32).to(device)
        
        mu_beta0_heat=torch.tensor(priors['mu_beta0_heat'],dtype=torch.float32).to(device)
        sigma_beta0_heat=torch.tensor(priors['sigma_beta0_heat'],dtype=torch.float32).to(device)
        mu_beta1_heat=torch.tensor(priors['mu_beta1_heat'],dtype=torch.float32).to(device)
        sigma_beta1_heat=torch.tensor(priors['sigma_beta1_heat'],dtype=torch.float32).to(device)
        sigma_heat=torch.tensor(priors['sigma_heat'],dtype=torch.float32).to(device)

        mu_heat_on=torch.tensor(priors['mu_heat_on'],dtype=torch.float32).to(device)
        sigma_heat_on=torch.tensor(priors['sigma_heat_on'],dtype=torch.float32).to(device)
        mu_heat_off=torch.tensor(priors['mu_heat_off'],dtype=torch.float32).to(device)
        sigma_heat_off=torch.tensor(priors['sigma_heat_off'],dtype=torch.float32).to(device)

        mu_beta0_cool=torch.tensor(priors['mu_beta0_cool'],dtype=torch.float32).to(device)
        sigma_beta0_cool=torch.tensor(priors['sigma_beta0_cool'],dtype=torch.float32).to(device)
        mu_beta1_cool=torch.tensor(priors['mu_beta1_cool'],dtype=torch.float32).to(device)
        sigma_beta1_cool=torch.tensor(priors['sigma_beta1_cool'],dtype=torch.float32).to(device)
        sigma_cool=torch.tensor(priors['sigma_cool'],dtype=torch.float32).to(device)

        mu_cool_on=torch.tensor(priors['mu_cool_on'],dtype=torch.float32).to(device)
        sigma_cool_on=torch.tensor(priors['sigma_cool_on'],dtype=torch.float32).to(device)
        mu_cool_off=torch.tensor(priors['mu_cool_off'],dtype=torch.float32).to(device)
        sigma_cool_off=torch.tensor(priors['sigma_cool_off'],dtype=torch.float32).to(device)

        mu_aux=torch.tensor(priors['mu_aux'],dtype=torch.float32).to(device)
        sigma_aux=torch.tensor(priors['sigma_aux'],dtype=torch.float32).to(device)

        mu_aux_on=torch.tensor(priors['mu_aux_on'],dtype=torch.float32).to(device)
        sigma_aux_on=torch.tensor(priors['sigma_aux_on'],dtype=torch.float32).to(device)
        mu_aux_off=torch.tensor(priors['mu_aux_off'],dtype=torch.float32).to(device)
        sigma_aux_off=torch.tensor(priors['sigma_aux_off'],dtype=torch.float32).to(device)


        mu_heat_df=torch.tensor(priors['mu_heat_df'],dtype=torch.float32).to(device)
        sigma_heat_df=torch.tensor(priors['sigma_heat_df'],dtype=torch.float32).to(device)

        #mu_phi_df=torch.tensor(priors['mu_phi_df'],dtype=torch.float32).to(device)
        #sigma_phi_df=torch.tensor(priors['sigma_phi_df'],dtype=torch.float32).to(device)
        
        #mu_psi=torch.tensor(priors['mu_psi'],dtype=torch.float32).to(device)
        #sigma_psi=torch.tensor(priors['sigma_psi'],dtype=torch.float32).to(device)

        mu_sigma_net=torch.tensor(priors['mu_sigma_net'],dtype=torch.float32).to(device)
        sigma_sigma_net=torch.tensor(priors['sigma_sigma_net'],dtype=torch.float32).to(device)
        
        
        E_misc=self.softplus(pyro.sample("E_misc",dist.Normal(mu_misc,sigma_misc).to_event(1)))
        

        # here mu_heat is not real scale. E_heat~LogNormal(mu_heat,sigma_heat)
        beta0_heat=pyro.sample("beta0_heat",dist.Normal(mu_beta0_heat,sigma_beta0_heat).to_event(1))
        beta1_heat=self.softplus(pyro.sample("beta1_heat",dist.Normal(mu_beta1_heat,sigma_beta1_heat).to_event(1)))
        mu_heat=pyro.deterministic("mu_heat",beta0_heat+beta1_heat*t_out)
        E_heat=self.softplus(pyro.sample("E_heat",dist.Normal(mu_heat,sigma_heat).to_event(1)))
        #print(f"E_heat shape is {E_heat.shape}")
        



        beta0_cool=pyro.sample("beta0_cool",dist.Normal(mu_beta0_cool,sigma_beta0_cool).to_event(1))
        beta1_cool=self.softplus(pyro.sample("beta1_cool",dist.Normal(mu_beta1_cool,sigma_beta1_cool).to_event(1)))
        mu_cool=pyro.deterministic("mu_cool",beta0_cool+beta1_cool*t_out)
        E_cool=self.softplus(pyro.sample("E_cool",dist.Normal(mu_cool,sigma_cool).to_event(1)))

        E_aux=self.softplus(pyro.sample("E_aux",dist.Normal(mu_aux,sigma_aux).to_event(1)))

        # phi_df=pyro.sample("phi_df",dist.Normal(mu_phi_df,sigma_phi_df).to_event(1))
        # mu_psi_alpha,mu_psi_beta=self.calculate_concentration(mu=mu_psi,sigma=sigma_psi)
        # psi=pyro.sample("psi",dist.Beta(concentration1=mu_psi_alpha ,concentration0=mu_psi_beta).to_event(1))
        
        eta_heat=i_heat.clone()
        mu_heat_on_alpha,mu_heat_on_beta=self.calculate_concentration(mu=mu_heat_on,sigma=sigma_heat_on)
        mu_heat_off_alpha,mu_heat_off_beta=self.calculate_concentration(mu=mu_heat_off,sigma=sigma_heat_off)
        
        # print(f'mu_heat_on_alpha is {mu_heat_on_alpha}')
        # print(f'mu_heat_on_beta is {mu_heat_on_beta}')

        eta_heat_on=pyro.sample("eta_heat_on",dist.Beta(concentration1=mu_heat_on_alpha ,concentration0=mu_heat_on_beta).to_event(1))
        eta_heat_off=pyro.sample("eta_heat_off",dist.Beta(concentration1=mu_heat_off_alpha ,concentration0=mu_heat_off_beta).to_event(1))
        eta_heat[i_heat_on==1]=eta_heat_on#[i_heat_on==1]
        eta_heat[i_heat_off==1]=eta_heat_off#[i_heat_off==1]
        
        eta_cool=i_cool.clone()
        mu_cool_on_alpha,mu_cool_on_beta=self.calculate_concentration(mu=mu_cool_on,sigma=sigma_cool_on)
        mu_cool_off_alpha,mu_cool_off_beta=self.calculate_concentration(mu=mu_cool_off,sigma=sigma_cool_off)
        eta_cool_on=pyro.sample("eta_cool_on",dist.Beta(concentration1=mu_cool_on_alpha ,concentration0=mu_cool_on_beta).to_event(1))
        eta_cool_off=pyro.sample("eta_cool_off",dist.Beta(concentration1=mu_cool_off_alpha ,concentration0=mu_cool_off_beta).to_event(1))
        eta_cool[i_cool_on==1]=eta_cool_on#[i_cool_on==1]
        eta_cool[i_cool_off==1]=eta_cool_off#[i_cool_off==1]
        
        eta_aux=i_aux.clone()
        mu_aux_on_alpha,mu_aux_on_beta=self.calculate_concentration(mu=mu_aux_on,sigma=sigma_aux_on)
        mu_aux_off_alpha,mu_aux_off_beta=self.calculate_concentration(mu=mu_aux_off,sigma=sigma_aux_off)
        eta_aux_on=pyro.sample("eta_aux_on",dist.Beta(concentration1=mu_aux_on_alpha ,concentration0=mu_aux_on_beta).to_event(1))
        eta_aux_off=pyro.sample("eta_aux_off",dist.Beta(concentration1=mu_aux_off_alpha ,concentration0=mu_aux_off_beta).to_event(1))
        eta_aux[i_aux_on==1]=eta_aux_on#[i_aux_on==1]
        eta_aux[i_aux_off==1]=eta_aux_off#[i_aux_off==1]

        E_heat_df=self.softplus(pyro.sample("E_heat_df",dist.Normal(mu_heat_df,sigma_heat_df).to_event(1)))
        #i_df=torch.zeros_like(i_heat).to(device)
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta.concentration1
        # concentration1 (float or Tensor) – 1st concentration parameter of the distribution (often referred to as alpha)
        # concentration0 (float or Tensor) – 2nd concentration parameter of the distribution (often referred to as beta)

        #with pyro.plate("Emisc", size=t_out.shape[0]):
        
        
        #i_df_on=pyro.sample("i_df_on",dist.Binomial(total_count=1,probs=psi))

        #i_df=torch.where((i_heat==torch.tensor(1,dtype=torch.float32))&(t_out<phi_df),i_df_on,i_df)
        


        y_nan=torch.any(torch.cat([torch.isnan(i_heat)[:,None],
                                torch.isnan(i_cool)[:,None],
                                torch.isnan(i_aux)[:,None],
                                torch.isnan(t_out)[:,None],
                                torch.isnan(i_heat_df)[:,None],
                                torch.isnan(y_net)[:,None]
                                ],dim=1),axis=1)

        #print(f'y_nan is {y_nan}')
        # print(f'eta_heat is {eta_heat}')
        # print(f'i_heat is {i_heat}')

        
        mu_net_=eta_heat*i_heat*E_heat+eta_cool*i_cool*E_cool+(eta_aux*i_aux)*E_aux+(i_heat_df)*E_heat_df+E_misc
        
        mu_net=pyro.deterministic("mu_net",mu_net_[~y_nan])
        sigma_net = self.softplus(pyro.sample("sigma_t_unit", dist.Normal(mu_sigma_net,sigma_sigma_net).to_event(1)))
        #print(f"sigma_net is {sigma_net}")
        y_net_=y_net.flatten()[~y_nan]

        


# class ResNIHCM(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # define dimensions
        
#         # Use ELU see Murphy p.397
#         self.elu=nn.ELU()
#         self.relu=nn.ReLU()
#         self.softmax=nn.Softmax(dim=1)
#         self.tanh=nn.Tanh()
#         self.softplus=nn.Softplus()
    
#     def calculate_concentration(self,mu,sigma):
#         concentration_alpha=((1-mu)/(sigma**2)-1/mu)*(mu**2)
#         concentration_beta=concentration_alpha*(1/mu-1)
#         return concentration_alpha, concentration_beta

#     def model(self, y_net, t_out,
#                     i_heat,i_heat_on,i_heat_off,
#                     i_cool,i_cool_on,i_cool_off,
#                     i_aux,i_aux_on,i_aux_off,
#                     priors=None):
#         # it is hard to generalize the process. 
#         # we may have matrix, 

#         # initial network
#         self.batch_sz=t_out.shape[0]
#         device=t_out.device
        
#         if priors is None:
#             add_noise=0 # no noise addition for priors
#             noise_scale=0.01
#             noise_mean=0
#             priors={
#                 "mu_misc":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_misc~logN(-3,2.5) [0.0004,0.05,6.783]
#                 "sigma_misc":np.array([3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_beta0_heat":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
#                 "sigma_beta0_heat":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
#                 "mu_beta1_heat":np.array([-2.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
#                 "sigma_beta1_heat":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
#                 "sigma_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_heat~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
#                 "mu_heat_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_on~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
#                 "sigma_heat_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_heat_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_off~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
#                 "sigma_heat_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_beta0_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
#                 "sigma_beta0_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
#                 "mu_beta1_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
#                 "sigma_beta1_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
#                 "sigma_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_cool~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
#                 "mu_cool_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_on~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
#                 "sigma_cool_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_cool_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_off~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
#                 "sigma_cool_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_aux":np.array([-0.4])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
#                 "sigma_aux":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_heat_df":np.array([-3.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
#                 "sigma_heat_df":np.array([1.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
#                 "mu_aux_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_on~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
#                 "sigma_aux_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_aux_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_off~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
#                 "sigma_aux_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
#                 "mu_phi_df":np.array([-1/3])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # phi~N(-1/3,1/6) give [-2/3,-1/3,0] which is [-10,0,10] in real scale
#                 "sigma_phi_df":np.array([1/6])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
#                 "mu_psi":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # psi~Beta(mu_psi=0.5,sigma_psi=1/12) # 0~1 flat 
#                 "sigma_psi":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,     # 
                
#                 "mu_sigma_net":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,  # 
#                 "sigma_sigma_net":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise # 

#             }
            
  
#         mu_misc=torch.tensor(priors['mu_misc'],dtype=torch.float32).to(device)
#         sigma_misc=torch.tensor(priors['sigma_misc'],dtype=torch.float32).to(device)
        
#         mu_beta0_heat=torch.tensor(priors['mu_beta0_heat'],dtype=torch.float32).to(device)
#         sigma_beta0_heat=torch.tensor(priors['sigma_beta0_heat'],dtype=torch.float32).to(device)
#         mu_beta1_heat=torch.tensor(priors['mu_beta1_heat'],dtype=torch.float32).to(device)
#         sigma_beta1_heat=torch.tensor(priors['sigma_beta1_heat'],dtype=torch.float32).to(device)
#         sigma_heat=torch.tensor(priors['sigma_heat'],dtype=torch.float32).to(device)

#         mu_heat_on=torch.tensor(priors['mu_heat_on'],dtype=torch.float32).to(device)
#         sigma_heat_on=torch.tensor(priors['sigma_heat_on'],dtype=torch.float32).to(device)
#         mu_heat_off=torch.tensor(priors['mu_heat_off'],dtype=torch.float32).to(device)
#         sigma_heat_off=torch.tensor(priors['sigma_heat_off'],dtype=torch.float32).to(device)

#         mu_beta0_cool=torch.tensor(priors['mu_beta0_cool'],dtype=torch.float32).to(device)
#         sigma_beta0_cool=torch.tensor(priors['sigma_beta0_cool'],dtype=torch.float32).to(device)
#         mu_beta1_cool=torch.tensor(priors['mu_beta1_cool'],dtype=torch.float32).to(device)
#         sigma_beta1_cool=torch.tensor(priors['sigma_beta1_cool'],dtype=torch.float32).to(device)
#         sigma_cool=torch.tensor(priors['sigma_cool'],dtype=torch.float32).to(device)

#         mu_cool_on=torch.tensor(priors['mu_cool_on'],dtype=torch.float32).to(device)
#         sigma_cool_on=torch.tensor(priors['sigma_cool_on'],dtype=torch.float32).to(device)
#         mu_cool_off=torch.tensor(priors['mu_cool_off'],dtype=torch.float32).to(device)
#         sigma_cool_off=torch.tensor(priors['sigma_cool_off'],dtype=torch.float32).to(device)

#         mu_aux=torch.tensor(priors['mu_aux'],dtype=torch.float32).to(device)
#         sigma_aux=torch.tensor(priors['sigma_aux'],dtype=torch.float32).to(device)

#         mu_aux_on=torch.tensor(priors['mu_aux_on'],dtype=torch.float32).to(device)
#         sigma_aux_on=torch.tensor(priors['sigma_aux_on'],dtype=torch.float32).to(device)
#         mu_aux_off=torch.tensor(priors['mu_aux_off'],dtype=torch.float32).to(device)
#         sigma_aux_off=torch.tensor(priors['sigma_aux_off'],dtype=torch.float32).to(device)

#         #mu_phi_df=torch.tensor(priors['mu_phi_df'],dtype=torch.float32).to(device)
#         #sigma_phi_df=torch.tensor(priors['sigma_phi_df'],dtype=torch.float32).to(device)
        
#         #mu_psi=torch.tensor(priors['mu_psi'],dtype=torch.float32).to(device)
#         #sigma_psi=torch.tensor(priors['sigma_psi'],dtype=torch.float32).to(device)

#         mu_sigma_net=torch.tensor(priors['mu_sigma_net'],dtype=torch.float32).to(device)
#         sigma_sigma_net=torch.tensor(priors['sigma_sigma_net'],dtype=torch.float32).to(device)
        
        
#         E_misc=pyro.sample("E_misc",dist.LogNormal(mu_misc,sigma_misc).to_event(1))
        

#         # here mu_heat is not real scale. E_heat~LogNormal(mu_heat,sigma_heat)
#         beta0_heat=pyro.sample("beta0_heat",dist.Normal(mu_beta0_heat,sigma_beta0_heat).to_event(1))
#         beta1_heat=pyro.sample("beta1_heat",dist.LogNormal(mu_beta1_heat,sigma_beta1_heat).to_event(1))
#         mu_heat=pyro.deterministic("mu_heat",beta0_heat+beta1_heat*t_out)
#         E_heat=pyro.sample("E_heat",dist.LogNormal(mu_heat,sigma_heat).to_event(1))
#         #print(f"E_heat shape is {E_heat.shape}")
        



#         beta0_cool=pyro.sample("beta0_cool",dist.Normal(mu_beta0_cool,sigma_beta0_cool).to_event(1))
#         beta1_cool=pyro.sample("beta1_cool",dist.LogNormal(mu_beta1_cool,sigma_beta1_cool).to_event(1))
#         mu_cool=pyro.deterministic("mu_cool",beta0_cool+beta1_cool*t_out)
#         E_cool=pyro.sample("E_cool",dist.LogNormal(mu_cool,sigma_cool).to_event(1))

#         E_aux=pyro.sample("E_aux",dist.LogNormal(mu_aux,sigma_aux).to_event(1))

#         # phi_df=pyro.sample("phi_df",dist.Normal(mu_phi_df,sigma_phi_df).to_event(1))
#         # mu_psi_alpha,mu_psi_beta=self.calculate_concentration(mu=mu_psi,sigma=sigma_psi)
#         # psi=pyro.sample("psi",dist.Beta(concentration1=mu_psi_alpha ,concentration0=mu_psi_beta).to_event(1))
        
#         eta_heat=i_heat.clone()
#         mu_heat_on_alpha,mu_heat_on_beta=self.calculate_concentration(mu=mu_heat_on,sigma=sigma_heat_on)
#         mu_heat_off_alpha,mu_heat_off_beta=self.calculate_concentration(mu=mu_heat_off,sigma=sigma_heat_off)
        
#         # print(f'mu_heat_on_alpha is {mu_heat_on_alpha}')
#         # print(f'mu_heat_on_beta is {mu_heat_on_beta}')

#         eta_heat_on=pyro.sample("eta_heat_on",dist.Beta(concentration1=mu_heat_on_alpha ,concentration0=mu_heat_on_beta).to_event(1))
#         eta_heat_off=pyro.sample("eta_heat_off",dist.Beta(concentration1=mu_heat_off_alpha ,concentration0=mu_heat_off_beta).to_event(1))
#         eta_heat[i_heat_on==1]=eta_heat_on#[i_heat_on==1]
#         eta_heat[i_heat_off==1]=eta_heat_off#[i_heat_off==1]
        
#         eta_cool=i_cool.clone()
#         mu_cool_on_alpha,mu_cool_on_beta=self.calculate_concentration(mu=mu_cool_on,sigma=sigma_cool_on)
#         mu_cool_off_alpha,mu_cool_off_beta=self.calculate_concentration(mu=mu_cool_off,sigma=sigma_cool_off)
#         eta_cool_on=pyro.sample("eta_cool_on",dist.Beta(concentration1=mu_cool_on_alpha ,concentration0=mu_cool_on_beta).to_event(1))
#         eta_cool_off=pyro.sample("eta_cool_off",dist.Beta(concentration1=mu_cool_off_alpha ,concentration0=mu_cool_off_beta).to_event(1))
#         eta_cool[i_cool_on==1]=eta_cool_on#[i_cool_on==1]
#         eta_cool[i_cool_off==1]=eta_cool_off#[i_cool_off==1]
        
#         eta_aux=i_aux.clone()
#         mu_aux_on_alpha,mu_aux_on_beta=self.calculate_concentration(mu=mu_aux_on,sigma=sigma_aux_on)
#         mu_aux_off_alpha,mu_aux_off_beta=self.calculate_concentration(mu=mu_aux_off,sigma=sigma_aux_off)
#         eta_aux_on=pyro.sample("eta_aux_on",dist.Beta(concentration1=mu_aux_on_alpha ,concentration0=mu_aux_on_beta).to_event(1))
#         eta_aux_off=pyro.sample("eta_aux_off",dist.Beta(concentration1=mu_aux_off_alpha ,concentration0=mu_aux_off_beta).to_event(1))
#         eta_aux[i_aux_on==1]=eta_aux_on#[i_aux_on==1]
#         eta_aux[i_aux_off==1]=eta_aux_off#[i_aux_off==1]


#         #i_df=torch.zeros_like(i_heat).to(device)
#         # https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta.concentration1
#         # concentration1 (float or Tensor) – 1st concentration parameter of the distribution (often referred to as alpha)
#         # concentration0 (float or Tensor) – 2nd concentration parameter of the distribution (often referred to as beta)

#         #with pyro.plate("Emisc", size=t_out.shape[0]):
        
        
#         #i_df_on=pyro.sample("i_df_on",dist.Binomial(total_count=1,probs=psi))

#         #i_df=torch.where((i_heat==torch.tensor(1,dtype=torch.float32))&(t_out<phi_df),i_df_on,i_df)
        


#         y_nan=torch.any(torch.cat([torch.isnan(i_heat)[:,None],
#                                 torch.isnan(i_cool)[:,None],
#                                 torch.isnan(i_aux)[:,None],
#                                 torch.isnan(t_out)[:,None],
#                                 torch.isnan(y_net)[:,None]
#                                 ],dim=1),axis=1)

#         #print(f'y_nan is {y_nan}')
#         # print(f'eta_heat is {eta_heat}')
#         # print(f'i_heat is {i_heat}')

        
#         mu_net_=eta_heat*i_heat*E_heat+eta_cool*i_cool*E_cool+(eta_aux*i_aux+i_df)*E_aux+E_misc
        
#         mu_net=pyro.deterministic("mu_net",mu_net_[~y_nan])
#         sigma_net = pyro.sample("sigma_t_unit", dist.LogNormal(mu_sigma_net,sigma_sigma_net).to_event(1))
#         #print(f"sigma_net is {sigma_net}")
#         y_net_=y_net.flatten()[~y_nan]

#         with pyro.plate("data", size=mu_net.shape[0]):
#             obs_net=pyro.sample("obs_net", dist.Normal(mu_net, sigma_net).to_event(1), obs=y_net_.flatten())   # .to_event(1)

#         return mu_net,priors

    
#     def guide(self, y_net, t_out,
#                     i_heat,i_heat_on,i_heat_off,
#                     i_cool,i_cool_on,i_cool_off,
#                     i_aux,i_aux_on,i_aux_off,
#                     priors=None):
        
        
#         self.batch_sz=t_out.shape[0]
#         device=t_out.device
        
#         if priors is None:
#             # add noise  for priors
#             add_noise=1
#             noise_scale=0.001
#             noise_mean=0
#             priors={
#                 "mu_misc":np.array([-3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_misc~logN(-3,2.5) [0.0004,0.05,6.783]
#                 "sigma_misc":np.array([3.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_beta0_heat":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
#                 "sigma_beta0_heat":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
#                 "mu_beta1_heat":np.array([-2.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
#                 "sigma_beta1_heat":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
#                 "sigma_heat":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_heat~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
#                 "mu_heat_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_on~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
#                 "sigma_heat_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_heat_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_heat_off~Beta(mu_heat_on=0.5,sigma_heat_on=1/12) # 0~1 flat 
#                 "sigma_heat_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_beta0_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta0~N(-2,1.0) [-4.0~0.0]
#                 "sigma_beta0_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, # exp(beta0) [0.018~1]
#                 "mu_beta1_cool":np.array([-2.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,   # beta1~logN(-1.5,0.8) [0.04~1.0]
#                 "sigma_beta1_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise, #  
#                 "sigma_cool":np.array([1.0])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # E_cool~logN(-1.2,0.6)  [0.093,0.30,1.0]
                
#                 "mu_cool_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_on~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
#                 "sigma_cool_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_cool_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_cool_off~Beta(mu_cool_on=0.5,sigma_cool_on=1/12) # 0~1 flat 
#                 "sigma_cool_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,

#                 "mu_aux":np.array([-0.4])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # E_aux~logN(-0.4,0.4) [0.3,0.67,1.43]
#                 "sigma_aux":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
#                 "mu_aux_on":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_on~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
#                 "sigma_aux_on":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,  
#                 "mu_aux_off":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,      # eta_aux_off~Beta(mu_aux_on=0.5,sigma_aux_on=1/12) # 0~1 flat 
#                 "sigma_aux_off":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,
                
#                 "mu_phi_df":np.array([-1/3])+np.random.normal(noise_mean,noise_scale,1)*add_noise,       # phi~N(-1/3,1/6) give [-2/3,-1/3,0] which is [-10,0,10] in real scale
#                 "sigma_phi_df":np.array([1/6])+np.random.normal(noise_mean,noise_scale,1)*add_noise,
#                 "mu_psi":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise,         # psi~Beta(mu_psi=0.5,sigma_psi=1/12) # 0~1 flat 
#                 "sigma_psi":np.sqrt(np.array([1/12]))+np.random.normal(noise_mean,noise_scale,1)*add_noise,     # 
                
#                 "mu_sigma_net":np.array([-4.])+np.random.normal(noise_mean,noise_scale,1)*add_noise,  # 
#                 "sigma_sigma_net":np.array([0.5])+np.random.normal(noise_mean,noise_scale,1)*add_noise # 

#             }
            

#         ################### params ########################33

#         mu_misc=pyro.param("mu_misc",torch.tensor(priors['mu_misc'],dtype=torch.float32).to(device))
#         sigma_misc=pyro.param("sigma_misc",torch.tensor(priors['sigma_misc'],dtype=torch.float32).to(device),constraints.positive)

#         mu_beta0_heat=pyro.param("mu_beta0_heat",torch.tensor(priors['mu_beta0_heat'],dtype=torch.float32).to(device))
#         sigma_beta0_heat=pyro.param("sigma_beta0_heat",torch.tensor(priors['sigma_beta0_heat'],dtype=torch.float32).to(device),constraints.positive)
#         mu_beta1_heat=pyro.param("mu_beta1_heat",torch.tensor(priors['mu_beta1_heat'],dtype=torch.float32).to(device))
#         sigma_beta1_heat=pyro.param("sigma_beta1_heat",torch.tensor(priors['sigma_beta1_heat'],dtype=torch.float32).to(device),constraints.positive)
#         sigma_heat=pyro.param("sigma_heat",torch.tensor(priors['sigma_heat'],dtype=torch.float32).to(device),constraints.positive)

#         mu_heat_on=pyro.param("mu_heat_on",torch.tensor(priors['mu_heat_on'],dtype=torch.float32).to(device))
#         sigma_heat_on=pyro.param("sigma_heat_on",torch.tensor(priors['sigma_heat_on'],dtype=torch.float32).to(device),constraints.positive)
#         mu_heat_off=pyro.param("mu_heat_off",torch.tensor(priors['mu_heat_off'],dtype=torch.float32).to(device))
#         sigma_heat_off=pyro.param("sigma_heat_off",torch.tensor(priors['sigma_heat_off'],dtype=torch.float32).to(device),constraints.positive)

#         mu_beta0_cool=pyro.param("mu_beta0_cool",torch.tensor(priors['mu_beta0_cool'],dtype=torch.float32).to(device))
#         sigma_beta0_cool=pyro.param("sigma_beta0_cool",torch.tensor(priors['sigma_beta0_cool'],dtype=torch.float32).to(device),constraints.positive)
#         mu_beta1_cool=pyro.param("mu_beta1_cool",torch.tensor(priors['mu_beta1_cool'],dtype=torch.float32).to(device))
#         sigma_beta1_cool=pyro.param("sigma_beta1_cool",torch.tensor(priors['sigma_beta1_cool'],dtype=torch.float32).to(device),constraints.positive)

#         sigma_cool=pyro.param("sigma_cool",torch.tensor(priors['sigma_cool'],dtype=torch.float32).to(device),constraints.positive)

#         mu_cool_on=pyro.param("mu_cool_on",torch.tensor(priors['mu_cool_on'],dtype=torch.float32).to(device))
#         sigma_cool_on=pyro.param("sigma_cool_on",torch.tensor(priors['sigma_cool_on'],dtype=torch.float32).to(device),constraints.positive)
#         mu_cool_off=pyro.param("mu_cool_off",torch.tensor(priors['mu_cool_off'],dtype=torch.float32).to(device))
#         sigma_cool_off=pyro.param("sigma_cool_off",torch.tensor(priors['sigma_cool_off'],dtype=torch.float32).to(device),constraints.positive)

#         mu_aux=pyro.param("mu_aux",torch.tensor(priors['mu_aux'],dtype=torch.float32).to(device))
#         sigma_aux=pyro.param("sigma_aux",torch.tensor(priors['sigma_aux'],dtype=torch.float32).to(device),constraints.positive)

#         mu_aux_on=pyro.param("mu_aux_on",torch.tensor(priors['mu_aux_on'],dtype=torch.float32).to(device))
#         sigma_aux_on=pyro.param("sigma_aux_on",torch.tensor(priors['sigma_aux_on'],dtype=torch.float32).to(device),constraints.positive)
#         mu_aux_off=pyro.param("mu_aux_off",torch.tensor(priors['mu_aux_off'],dtype=torch.float32).to(device))
#         sigma_aux_off=pyro.param("sigma_aux_off",torch.tensor(priors['sigma_aux_off'],dtype=torch.float32).to(device),constraints.positive)

#         mu_phi_df=pyro.param("mu_phi_df",torch.tensor(priors['mu_phi_df'],dtype=torch.float32).to(device))
#         sigma_phi_df=pyro.param("sigma_phi_df",torch.tensor(priors['sigma_phi_df'],dtype=torch.float32).to(device),constraints.positive)


        
#         mu_psi=pyro.param("mu_psi",torch.tensor(priors['mu_psi'],dtype=torch.float32).to(device))
#         sigma_psi=pyro.param("sigma_psi",torch.tensor(priors['sigma_psi'],dtype=torch.float32).to(device),constraints.positive)


#         mu_sigma_net=pyro.param("mu_sigma_net",torch.tensor(priors['mu_sigma_net'],dtype=torch.float32).to(device))
#         sigma_sigma_net=pyro.param("sigma_sigma_net",torch.tensor(priors['sigma_sigma_net'],dtype=torch.float32).to(device),constraints.positive)
    

#         ########33# samples

#         E_misc=pyro.sample("E_misc",dist.LogNormal(mu_misc,sigma_misc).to_event(1))
#         #print(E_misc.dtype)
        
#         # here mu_heat is not real scale. E_heat~LogNormal(mu_heat,sigma_heat)
#         # here mu_heat is not real scale. E_heat~LogNormal(mu_heat,sigma_heat)
#         beta0_heat=pyro.sample("beta0_heat",dist.Normal(mu_beta0_heat,sigma_beta0_heat).to_event(1))
#         beta1_heat=pyro.sample("beta1_heat",dist.LogNormal(mu_beta1_heat,sigma_beta1_heat).to_event(1))
#         mu_heat=pyro.deterministic("mu_heat",beta0_heat+beta1_heat*t_out)
#         E_heat=pyro.sample("E_heat",dist.LogNormal(mu_heat,sigma_heat).to_event(1))
#         #print(f"E_heat shape is {E_heat.shape}")
        



#         beta0_cool=pyro.sample("beta0_cool",dist.Normal(mu_beta0_cool,sigma_beta0_cool).to_event(1))
#         beta1_cool=pyro.sample("beta1_cool",dist.LogNormal(mu_beta1_cool,sigma_beta1_cool).to_event(1))
#         mu_cool=pyro.deterministic("mu_cool",beta0_cool+beta1_cool*t_out)
#         E_cool=pyro.sample("E_cool",dist.LogNormal(mu_cool,sigma_cool).to_event(1))

#         E_aux=pyro.sample("E_aux",dist.LogNormal(mu_aux,sigma_aux).to_event(1))

#         phi_df=pyro.sample("phi_df",dist.Normal(mu_phi_df,sigma_phi_df).to_event(1))
        
#         mu_psi_alpha,mu_psi_beta=self.calculate_concentration(mu=mu_psi,sigma=sigma_psi)

#         psi=pyro.sample("psi",dist.Beta(concentration1=mu_psi_alpha ,concentration0=mu_psi_beta).to_event(1))
        
#         eta_heat=i_heat.clone()
#         mu_heat_on_alpha,mu_heat_on_beta=self.calculate_concentration(mu=mu_heat_on,sigma=sigma_heat_on)
#         mu_heat_off_alpha,mu_heat_off_beta=self.calculate_concentration(mu=mu_heat_off,sigma=sigma_heat_off)
        
#         # print(f'mu_heat_on_alpha is {mu_heat_on_alpha}')
#         # print(f'mu_heat_on_beta is {mu_heat_on_beta}')

#         eta_heat_on=pyro.sample("eta_heat_on",dist.Beta(concentration1=mu_heat_on_alpha ,concentration0=mu_heat_on_beta).to_event(1))
#         eta_heat_off=pyro.sample("eta_heat_off",dist.Beta(concentration1=mu_heat_off_alpha ,concentration0=mu_heat_off_beta).to_event(1))
#         eta_heat[i_heat_on==1]=eta_heat_on#[i_heat_on==1]
#         eta_heat[i_heat_off==1]=eta_heat_off#[i_heat_off==1]
        

        
#         eta_cool=i_cool.clone()
#         mu_cool_on_alpha,mu_cool_on_beta=self.calculate_concentration(mu=mu_cool_on,sigma=sigma_cool_on)
#         mu_cool_off_alpha,mu_cool_off_beta=self.calculate_concentration(mu=mu_cool_off,sigma=sigma_cool_off)
#         eta_cool_on=pyro.sample("eta_cool_on",dist.Beta(concentration1=mu_cool_on_alpha ,concentration0=mu_cool_on_beta).to_event(1))
#         eta_cool_off=pyro.sample("eta_cool_off",dist.Beta(concentration1=mu_cool_off_alpha ,concentration0=mu_cool_off_beta).to_event(1))
#         eta_cool[i_cool_on==1]=eta_cool_on#[i_cool_on==1]
#         eta_cool[i_cool_off==1]=eta_cool_off#[i_cool_off==1]
        


#         eta_aux=i_aux.clone()
#         mu_aux_on_alpha,mu_aux_on_beta=self.calculate_concentration(mu=mu_aux_on,sigma=sigma_aux_on)
#         mu_aux_off_alpha,mu_aux_off_beta=self.calculate_concentration(mu=mu_aux_off,sigma=sigma_aux_off)
#         eta_aux_on=pyro.sample("eta_aux_on",dist.Beta(concentration1=mu_aux_on_alpha ,concentration0=mu_aux_on_beta).to_event(1))
#         eta_aux_off=pyro.sample("eta_aux_off",dist.Beta(concentration1=mu_aux_off_alpha ,concentration0=mu_aux_off_beta).to_event(1))
#         eta_aux[i_aux_on==1]=eta_aux_on#[i_aux_on==1]
#         eta_aux[i_aux_off==1]=eta_aux_off#[i_aux_off==1]


#         i_df=torch.zeros_like(i_heat).to(device)
#         # https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta.concentration1
#         # concentration1 (float or Tensor) – 1st concentration parameter of the distribution (often referred to as alpha)
#         # concentration0 (float or Tensor) – 2nd concentration parameter of the distribution (often referred to as beta)

#         #with pyro.plate("Emisc", size=t_out.shape[0]):
        
        
#         #i_df_on=pyro.sample("i_df_on",dist.Binomial(total_count=1,probs=psi))

#         #i_df=torch.where((i_heat==torch.tensor(1,dtype=torch.float32))&(t_out<phi_df),i_df_on,i_df)
        


#         y_nan=torch.any(torch.cat([torch.isnan(i_heat)[:,None],
#                                 torch.isnan(i_cool)[:,None],
#                                 torch.isnan(i_aux)[:,None],
#                                 torch.isnan(t_out)[:,None],
#                                 torch.isnan(y_net)[:,None]
#                                 ],dim=1),axis=1)

#         #print(f'y_nan is {y_nan}')
#         # print(f'eta_heat is {eta_heat}')
#         # print(f'i_heat is {i_heat}')

        
#         mu_net_=eta_heat*i_heat*E_heat+eta_cool*i_cool*E_cool+(eta_aux*i_aux+i_df)*E_aux+E_misc
        
#         mu_net=pyro.deterministic("mu_net",mu_net_[~y_nan])
#         sigma_net = pyro.sample("sigma_t_unit", dist.LogNormal(mu_sigma_net,sigma_sigma_net).to_event(1))
#         #print(f"sigma_net is {sigma_net}")
#         y_net_=y_net.flatten()[~y_nan]


# mu_misc=(-3.0*torch.ones(1)).to(device)
# sigma_misc=(2.5*torch.ones(1)).to(device)
# beta0_heat=(-2*torch.ones(1)).to(device) # t_out -1~1 slope probably - for heating. not large (or decide after plotting)
# beta1_heat=(-2*torch.ones(1)).to(device)  # t_out -1~1 slope probably - for heating. not large (or decide after plotting)
# sigma_heat=(1.0*torch.ones(1)).to(device)
# mu_heat_on=(1*torch.ones(1)).to(device)
# sigma_heat_on=(0.25*torch.ones(1)).to(device)
# mu_heat_off=(1*torch.ones(1)).to(device)
# sigma_heat_off=(0.25*torch.ones(1)).to(device)
# beta0_cool=(-2*torch.ones(1)).to(device) # t_out -1~1 slope probably - for heating. not large (or decide after plotting)
# beta1_cool=(-2*torch.ones(1)).to(device)  # t_out -1~1 slope probably - for heating. not large (or decide after plotting)
# sigma_cool=(1.0*torch.ones(1)).to(device)
# mu_cool_on=(1*torch.ones(1)).to(device)
# sigma_cool_on=(0.25*torch.ones(1)).to(device)
# mu_cool_off=(1*torch.ones(1)).to(device)
# sigma_cool_off=(0.25*torch.ones(1)).to(device)
# mu_aux=(-0.7*torch.ones(1)).to(device)
# sigma_aux=(0.6*torch.ones(1)).to(device)
# mu_aux_on=(1*torch.ones(1)).to(device)
# sigma_aux_on=(0.25*torch.ones(1)).to(device)
# mu_aux_off=(1*torch.ones(1)).to(device)
# sigma_aux_off=(0.25*torch.ones(1)).to(device)

# mu_phi_df=(0*torch.ones(1)).to(device)
# sigma_phi_df=(0*torch.ones(1)).to(device)

# mu_psi=(1*torch.ones(1)).to(device)
# sigma_psi=(0.25*torch.ones(1)).to(device)

# mu_sigma_net=(1*torch.ones(1)).to(device)
# sigma_sigma_net=(0.25*torch.ones(1)).to(device)

##########Guide

# mu_misc=pyro.param("mu_misc",(0.01*torch.randn(1)).to(device))
# sigma_misc=pyro.param("sigma_misc",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_beta0_heat=pyro.param("mu_beta0_heat",(0.01*torch.randn(1)).to(device))
# sigma_beta0_heat=pyro.param("sigma_beta0_heat",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# mu_beta1_heat=pyro.param("mu_beta1_heat",(0.01*torch.randn(1)).to(device))
# sigma_beta1_heat=pyro.param("sigma_beta1_heat",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# sigma_heat=pyro.param("sigma_heat",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_heat_on=pyro.param("mu_heat_on",(0.01*torch.randn(1)).to(device))
# sigma_heat_on=pyro.param("sigma_heat_on",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# mu_heat_off=pyro.param("mu_heat_off",(0.01*torch.randn(1)).to(device))
# sigma_heat_off=pyro.param("sigma_heat_off",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_beta0_cool=pyro.param("mu_beta0_cool",(0.01*torch.randn(1)).to(device))
# sigma_beta0_cool=pyro.param("sigma_beta0_cool",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# mu_beta1_cool=pyro.param("mu_beta1_cool",(0.01*torch.randn(1)).to(device))
# sigma_beta1_cool=pyro.param("sigma_beta1_cool",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# sigma_cool=pyro.param("sigma_cool",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_cool_on=pyro.param("mu_cool_on",(0.01*torch.randn(1)).to(device))
# sigma_cool_on=pyro.param("sigma_cool_on",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# mu_cool_off=pyro.param("mu_cool_off",(0.01*torch.randn(1)).to(device))
# sigma_cool_off=pyro.param("sigma_cool_off",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_aux=pyro.param("mu_aux",(0.01*torch.randn(1)).to(device))
# sigma_aux=pyro.param("sigma_aux",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_aux_on=pyro.param("mu_aux_on",(0.01*torch.randn(1)).to(device))
# sigma_aux_on=pyro.param("sigma_aux_on",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)
# mu_aux_off=pyro.param("mu_aux_off",(0.01*torch.randn(1)).to(device))
# sigma_aux_off=pyro.param("sigma_aux_off",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)

# mu_phi_df=pyro.param("mu_phi_df",(0.01*torch.randn(1)).to(device))
# sigma_phi_df=pyro.param("sigma_phi_df",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)


# mu_psi=pyro.param("mu_psi",(0.01*torch.randn(1)).to(device))
# sigma_psi=pyro.param("sigma_psi",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)


# mu_sigma_net=pyro.param("mu_sigma_net",(0.01*torch.randn(1)).to(device))
# sigma_sigma_net=pyro.param("sigma_sigma_net",(0.05*torch.abs(torch.randn(1))).to(device),constraints.positive)