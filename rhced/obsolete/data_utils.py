__all__=["detect_on_off","min_max_scale","min_max_recover","scale_by_name","CustomDataloader"]

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataloader(Dataset):
    def __init__(self, y_net, t_out,i_heat,i_heat_on,i_heat_off,i_cool,i_cool_on,i_cool_off,i_aux,i_aux_on,i_aux_off,i_heat_df):
        if not torch.is_tensor(y_net):
            self.y_net = torch.from_numpy(y_net).type(torch.float32)

        if not torch.is_tensor(t_out):
            self.t_out = torch.from_numpy(t_out).type(torch.float32)

        if not torch.is_tensor(i_heat):
            self.i_heat = torch.from_numpy(i_heat).type(torch.float32)

        if not torch.is_tensor(i_heat_on):
            self.i_heat_on = torch.from_numpy(i_heat_on).type(torch.float32)

        if not torch.is_tensor(i_heat_off):
            self.i_heat_off = torch.from_numpy(i_heat_off).type(torch.float32)

        if not torch.is_tensor(i_cool):
            self.i_cool = torch.from_numpy(i_cool).type(torch.float32)

        if not torch.is_tensor(i_cool_on):
            self.i_cool_on = torch.from_numpy(i_cool_on).type(torch.float32)

        if not torch.is_tensor(i_cool_off):
            self.i_cool_off = torch.from_numpy(i_cool_off).type(torch.float32)

        if not torch.is_tensor(i_aux):
            self.i_aux = torch.from_numpy(i_aux).type(torch.float32)

        if not torch.is_tensor(i_aux_on):
            self.i_aux_on = torch.from_numpy(i_aux_on).type(torch.float32)

        if not torch.is_tensor(i_aux_off):
            self.i_aux_off = torch.from_numpy(i_aux_off).type(torch.float32)
        if not torch.is_tensor(i_heat_df):
            self.i_heat_df = torch.from_numpy(i_heat_df).type(torch.float32)

        

    def __len__(self):
        return len(self.t_out)

    def __getitem__(self, idx):
        return self.y_net[idx], self.t_out[idx], self.i_heat[idx], self.i_heat_on[idx], self.i_heat_off[idx], self.i_cool[idx], self.i_cool_on[idx], self.i_cool_off[idx], self.i_aux[idx], self.i_aux_on[idx], self.i_aux_off[idx], self.i_heat_df[idx]


########################### Utility functions used for energyplus ############################
def min_max_scale(x,x_min,x_max,lower=0.,upper=1.):
    # min max scale from lower to upper.
    # it can hanldes np.nan.
    # helper function
    if type(x)!=np.ndarray:
        raise ValueError("Put data as numpy ndarray.format.")
    x=np.array(x)
    return (lower+((x-x_min)*(upper-lower))/(x_max-x_min))

def min_max_recover(x,x_min,x_max,lower=0.,upper=1.):
    
    # it can hanldes np.nan.

    if type(x)!=np.ndarray:
        raise ValueError("Put data as numpy ndarray.format.")

    x=np.array(x)
    return ((x-lower)*(x_max-x_min)/(upper-lower)+x_min)

def scale_by_name(df,scale_const,var_name):
    if np.isin(var_name,df.columns).item():
        pass
    else:
        raise ValueError(f"{var_name} does not exist in the input dataframe.")

    x_max=scale_const[f'{var_name}_max']
    x_min=scale_const[f'{var_name}_min']
    lower=scale_const[f'{var_name}_lower']
    upper=scale_const[f'{var_name}_upper']
    x=df[var_name].to_numpy()
    x_scale=min_max_scale(x,x_min,x_max,lower,upper)
    #df[var_name]=x_scale
#     df[var_name]=\
#             df_scale.apply(lambda x :min_max_scale(x=x['sp_cool'],x_min=t_min,x_max=t_max,lower=-1.,upper=1.),axis=1)
    return x_scale
    
############################################################################33



def detect_on_off(operation,operation_state):
    # operation is vector of operation_state (numpy object vector with string data)
    # operation_state is hc system operation state \in {heat,cool,aux,heat_aux,idle}
    # To detect heatpump heating, it should be ['heat','heat_aux']
    
    if operation.dtype=="O":
        pass
    else:
        # change dtype to Object to have str and nan together
        operation=operation.astype("O")    
    # check if any 'nan' and put np.nan for the Object type vector 
    if np.any(operation==str(np.nan)):
        operation[operation==str(np.nan)]=np.nan
    
    if type(operation_state)==np.ndarray:
        if operation_state.dtype=="O":
            pass
        else:
            operation_state=operation_state.astype("O")
    else:
        operation_state=np.array([operation_state],dtype=object).flatten()


    i_vec=np.zeros_like(operation).astype("float32")
    # put nan for missing inputs
    i_vec[pd.isna(operation)]=np.nan
    i_vec_on=i_vec.copy()
    i_vec_off=i_vec.copy()
    
    i_vec[(np.isin(operation,operation_state)&(~pd.isna(operation)))]=1.0
    for t in np.arange(1,len(i_vec)):
        if (~np.isnan(i_vec[t-1])&~np.isnan(i_vec[t])) & ((i_vec[t-1]==0)&(i_vec[t]==1)):
            # check device idle -> on
            i_vec_on[t]=1
        if (~np.isnan(i_vec[t-1])&~np.isnan(i_vec[t])) & ((i_vec[t-1]==1)&(i_vec[t]==0)):
            # check device on -> idle
            if (t==1):
                i_vec_off[t-1]=1
            elif ((i_vec[t-2]==0)&(i_vec[t-1]==1)&(i_vec[t]==0)):
                # check if idle -> on -> idle. In this case it is just on state
                pass
            else:
                i_vec_off[t-1]=1
    i_vec[i_vec_on==1]=0
    i_vec[i_vec_off==1]=0
    
    
    return i_vec,i_vec_on,i_vec_off
