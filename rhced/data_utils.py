'''
Collections of several functions that used to data pre-processing.
'''

__all__=["detect_operation","min_max_scale","min_max_recover","filter_na_outlier","scale_by_name","detect_defrost","nan_mean","create_unit_input","create_input_values","get_scale_const_template","get_last_outputs","get_wetbulb"]
import numpy as np
import pandas as pd
import re
import pickle
#from metpy.calc import dewpoint_from_relative_humidity,wet_bulb_temperature
#from metpy.units import units
import psypy.psySI as SI 

def psypy_wb(temp,rh,p_atm=101325):
    '''Calculate Wet-bulb temperature from dry-bulb temperature and relative humidity via psypy package.
    Args:
        temp (numeric): temperature in celsius [C].
        rh (numeric): relative humidity in 0-1 scale [-].
        p_atm (numeric) : atmospheric pressure [Pa]. Default is 101325 Pa.
    Raises: return np.nan when wet-bulb is too low.
    Returns:
        wb (numeric): wet-bulb temperature in celsius. 
    '''
    temp_k=temp+273.15 # Kelvin
    try:
        S=SI.state("DBT",temp_k,"RH",rh,p_atm) 
        wb=S[5]-273.15 # Kelvin to Celsius
    except:
        wb=np.nan # return np.nan when wet-bulb is too low.
    return wb


def get_wetbulb(temp,rh,p_atm=101325):
    '''Vectorization of wet-bulb calculation (psypy_wb function) while skipping nan.
    Args:
        temp (np.array [1d]): temperature in celsius [C].
        rh (np.array [1d]): relative humidity in 0-1 scale [-].
        p_atm (np.array [1d] or numeric) : atmospheric pressure [Pa]. Default is 101325 Pa.
    Raises: 
    Returns:
        wb (np.array [1d]): wet-bulb temperature in np.array 
    '''
    nan_index=np.any(np.isnan(np.stack([temp,rh],axis=1)),axis=1) # find nan data in row
    base_=np.zeros([temp.shape[0]]) # original data dimension
    base_[nan_index]=np.nan # put nan for nan values.
    n_data=temp[~nan_index].shape[0] # non-nan data
    vec_psypy_wb=np.vectorize(psypy_wb) # vectorize psypy_wb function 
    wb=vec_psypy_wb(temp=temp[~nan_index],rh=rh[~nan_index],p_atm=p_atm) # calculate wet-bulb for non-nan data 
    base_[~nan_index]=wb # put calculated wet-bulb of non-nan data for the original data dimension
    return(base_)

def min_max_scale(x,x_min,x_max,lower=0.,upper=1.):
    '''
    min_max_scale (see rescaling in https://en.wikipedia.org/wiki/Feature_scaling).
    The data in [x_min,x_max] is linearly rescaled to [lower,upper].
    # it can hanldes np.nan.
    Args:
        x (1d np.array): data that needs to be rescaled.
        x_min (numeric): minimum value of x. 
        x_max (numeric): maximum value of x.
        lower (numeric): minimum value of rescaled x. default is 0. 
        upper (numeric): maximum value of rescaled x. default is 1.
    Raises: 
    Returns:
        rescaled x (numeric): rescaled x from [x_min,x_max] to [lower,upper] 
    '''    
    if type(x)!=np.ndarray:
        raise ValueError("Put data as numpy ndarray.format.")
    x=np.array(x)
    return (lower+((x-x_min)*(upper-lower))/(x_max-x_min))

def min_max_recover(x,x_min,x_max,lower=0.,upper=1.):
    '''
    reverse of min_max_scale function. The rescaled x is recovered to original scale.
    The data in [lower,upper] is recovered to [x_min,x_max] range.
    # it can hanldes np.nan.
    Args:
        x (1d np.array): data that needs to be rescaled.
        x_min (numeric): minimum value of x. 
        x_max (numeric): maximum value of x.
        lower (numeric): minimum value of rescaled x. default is 0. 
        upper (numeric): maximum value of rescaled x. default is 1.
    Raises: 
    Returns:
        original scale x (numeric): rescaled x is recovered to [lower,upper] range from [x_min,x_max]. 
    '''
    if type(x)!=np.ndarray:
        raise ValueError("Put data as numpy ndarray.format.")
    x=np.array(x)
    return ((x-lower)*(x_max-x_min)/(upper-lower)+x_min)

def scale_by_name(df,scale_const,var_name):
    '''
    Apply min_max_scale for df['var_name']
    Args:
        df (pandas dataframe): data frame that contains various variables that needs to be scaled.
        scale_const (dictionary): dictionary that contains x_min, x_max, lower, and upper for var_name variable in df.
        var_name (str): a variable that needs to be scaled. One of the columns in df.
    Raises: 
    Returns:
        scaled df['var_name] (numeric): scaled df['var_name] as a numpy array. 
    '''
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
    #df[var_name]=df_scale.apply(lambda x :min_max_scale(x=x['sp_cool'],x_min=t_min,x_max=t_max,lower=-1.,upper=1.),axis=1)
    return x_scale

def get_scale_const_template():
    '''
    The tempelate for scale_const. Feel free to revise for your case.

    Args:
    Raises: 
    Returns:
        scale_const (dictionary): dictionary that contains min, max, lower, upper information of T_out, T_in, wb_in, net, heat, cool,df,aux. 
    '''
    scale_const={"T_out_max":40.,"T_out_min":-20.,"T_out_upper":1.,"T_out_lower":-1.,\
                "T_in_max":30.,"T_in_min":10.,"T_in_upper":1.,"T_in_lower":-1.,\
                "wb_in_max":30.,"wb_in_min":0.,"wb_in_upper":1.,"wb_in_lower":-1.,\
             "net_max":10000.0,"net_min":0.0,"heat_max":2000,"cool_max":2000,"df_max":4000,"aux_max":6000}
    # what is 0 in scaled T_out?
    #-1/3=-1+((0+20)*(1+1))/(40+20) 
    return scale_const

def get_last_outputs(output_path):
    # get the latest input_values and prior_values file path given the output_path
    xx=output_path.glob("*")
    xxx=[x for x in xx if (x.is_file()) and (re.search('prior_values',x.__str__())) ]
    last_prior_values_path=xxx[-1]
    xx=output_path.glob("*")
    xxx2=[x for x in xx if (x.is_file()) and (re.search('input_values',x.__str__())) ]
    last_input_values_path=xxx2[-1]
    print(f'last_prior_values_path is {last_prior_values_path}')
    print(f'last_input_values_path is {last_input_values_path}')
    with open(last_input_values_path.__str__(),"rb") as handle:
        last_input_values=pickle.load(handle)
    with open(last_prior_values_path.__str__(),"rb") as handle:
        last_prior_values=pickle.load(handle)

    return last_input_values, last_prior_values,last_input_values_path,last_prior_values_path


############################################################################33
def detect_operation(operation,operation_state):
    # operation is vector of operation_state (numpy object vector with string data)
    # operation_state is hc system operation state \in {heat1,cool1,aux1,heat1_aux1,idle}
    # To detect heatpump heating, it should be ['heat1','heat1_aux1']
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
    
    i_vec[(np.isin(operation,operation_state)&(~pd.isna(operation)))]=1.0
    return i_vec


# ,T_out=None,df_cutpoint=0.0
def detect_defrost(i_vec,T_out,df_cutpoint=0):
    i_heat=i_vec.copy()
    i_df=i_vec.copy()
    i_heat[(T_out<=df_cutpoint) ]=0.0
    i_df[(T_out>df_cutpoint) ]=0.0
    return i_heat, i_df

def filter_na_outlier(x,x_max,x_min):
    x[x>(x_max)]=np.nan
    x[x<(x_min)]=np.nan
    return x

def nan_mean(vec):
    if np.all(np.isnan(vec)):
        return(np.nan)
    else:
        vec[np.isnan(vec)]=0.

        return(np.mean(vec))

def create_unit_input(thermostat_data,meter_data,scale_const,time_interval=15,training=False):

    # checking timestamp is pd.Timestamp object.
    if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(thermostat_data['timestamp']):
        pass
    else:
        thermostat_data['timestamp']=pd.to_datetime(thermostat_data['timestamp'])
    if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(meter_data['timestamp']):
        pass
    else:
        meter_data['timestamp']=pd.to_datetime(meter_data['timestamp'])

    start_date=thermostat_data['timestamp'][0].strftime("%Y-%m-%d")
    end_date=thermostat_data.tail(1)['timestamp'].reset_index(drop=True)[0].strftime("%Y-%m-%d")
    
    
    # create regular time inerval data
    # 5min time grid
    num_time_grid=((pd.Timestamp(end_date)-pd.Timestamp(start_date)+pd.Timedelta("1day")).total_seconds())/300
    time_grid=pd.date_range(pd.Timestamp(start_date), periods=num_time_grid, freq='5min')
    thermostat_data_base=pd.DataFrame(data={"timestamp":time_grid})
    thermostat_data=pd.merge(thermostat_data_base, thermostat_data, how='left', on=['timestamp'])

    thermostat_time_delta=((thermostat_data['timestamp'][1]-thermostat_data['timestamp'][0]).total_seconds())/60 # in minutes
    if (thermostat_time_delta!=5):
        raise ValueError("thermostat_data time interval should be 5 minute interval.")


    # time grid for meter_data
    meter_time_delta=((meter_data['timestamp'][1]-meter_data['timestamp'][0]).total_seconds())/60 # in minutes
    num_time_grid_meter=((pd.Timestamp(end_date)-pd.Timestamp(start_date)+pd.Timedelta("1day")).total_seconds())/(meter_time_delta*60) #
    time_grid_meter=pd.date_range(pd.Timestamp(start_date), periods=num_time_grid_meter, freq=f'{meter_time_delta}min')
    meter_data_base=pd.DataFrame(data={"timestamp":time_grid_meter})
    meter_data=pd.merge(meter_data_base, meter_data, how='left', on=['timestamp'])

    # make meter_data to 5 minute interval data
    if (meter_time_delta>5) and (meter_time_delta%5==0):
        meter_data=meter_data.iloc[np.repeat(np.arange(len(meter_data)), int(meter_time_delta/5))].reset_index(drop=True).copy()
        meter_data['timestamp']=time_grid
    elif (meter_time_delta==5):
        pass
    else:
        raise ValueError("meter_data time interval should be 5 minute interval or larger interval one with multiple of 5.")


    wb_in=get_wetbulb(temp=thermostat_data['T_in'].to_numpy(),rh=thermostat_data['rh_in'].to_numpy())
    wb_out=get_wetbulb(temp=thermostat_data['T_out'].to_numpy(),rh=thermostat_data['rh_out'].to_numpy())
    thermostat_data['wb_in']=wb_in
    thermostat_data['wb_out']=wb_out
    T_out=thermostat_data['T_out'].to_numpy()

    i_heat1_all=detect_operation(operation=thermostat_data['operation'],operation_state=['heat1','heat1_aux1'])
    i_heat2_all=detect_operation(operation=thermostat_data['operation'],operation_state=['heat2','heat2_aux1'])
    i_cool1=detect_operation(operation=thermostat_data['operation'],operation_state=['cool1'])
    i_cool2=detect_operation(operation=thermostat_data['operation'],operation_state=['cool2'])
    i_aux1=detect_operation(operation=thermostat_data['operation'],operation_state=['aux1','heat1_aux1','heat2_aux1'])
    i_fan1=detect_operation(operation=thermostat_data['operation'],operation_state=['fan1'])

    unit_data=pd.merge(thermostat_data, meter_data, how='left', on=['timestamp', 'unitcode'])

    net_max2=scale_const['net_max']*2.
    heat_max2=scale_const['heat_max']*2.
    cool_max2=scale_const['cool_max']*2.
    aux_max2=scale_const['aux_max']*2.
    df_max2=scale_const['df_max']*2.

    if type(unit_data['timestamp'][0])==str:
        unit_data['timestamp']=pd.to_datetime(unit_data['timestamp'].str.extract(r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)')[0])
    

    unit_input=pd.DataFrame(data={"timestamp":unit_data['timestamp'].reset_index(drop=True),
                        "T_out":unit_data['T_out'].to_numpy(),
                        "T_in":unit_data['T_in'].to_numpy(),
                        "wb_in":unit_data['wb_in'].to_numpy(),
                        "net":unit_data['net'].to_numpy(),
                        "i_heat1_all":i_heat1_all,
                        "i_heat2_all":i_heat2_all,
                        "i_cool1":i_cool1,
                        "i_cool2":i_cool2,
                        "i_aux1":i_aux1,
                        "i_fan1":i_fan1,
                        "operation":unit_data['operation']
                        })

    unit_input['net']=filter_na_outlier(x=unit_input['net'].to_numpy(),x_max=net_max2,x_min=0.)
    if training:
        # training only no heatpump ahu data
        pass
    else:
        unit_input['ahu']=unit_data['ahu'].to_numpy()
        unit_input['heatpump']=unit_data['heatpump'].to_numpy()
        unit_input['misc']=unit_data['net'].to_numpy()-unit_data['ahu'].to_numpy()-unit_data['heatpump'].to_numpy()
        ahu=unit_input['ahu'].to_numpy()
        unit_input['ahu']=filter_na_outlier(x=ahu,x_max=aux_max2,x_min=0.)
        heatpump=unit_input['heatpump'].to_numpy()
        unit_input['heatpump']=filter_na_outlier(x=heatpump,x_max=heat_max2,x_min=0.)
        #unit_input['misc']=fill_na_outlier(x=unit_input['misc'].to_numpy(),x_max=misc_max2,x_min=0.)


    if time_interval==5:
        output=unit_input.copy()
        #pass
    else:
        #unit_input_15=unit_input.copy()
        unit_input['timestamp']=unit_input['timestamp'].dt.floor(f'{time_interval}min')
        output=unit_input.groupby('timestamp').agg(nan_mean).reset_index().copy()
        #output=unit_input.groupby('timestamp').mean().reset_index().copy()
    return output

def create_input_values(unit_input,scale_const=None,training=True):
    
    if training:
        print("Drop NA data to create training data.")
        nan_index=np.any(unit_input.isnull().to_numpy(),axis=1)
        timestamp=unit_input['timestamp']
        unit_input=unit_input.dropna()
    else:
        timestamp=unit_input['timestamp']
        nan_index=np.repeat(True,unit_input.shape[0])

    time_delta=(unit_input['timestamp'][1]-unit_input['timestamp'][0])/np.timedelta64(1,'m')
    net=unit_input['net'].to_numpy()
    
    T_out=unit_input['T_out'].to_numpy()
    T_in=unit_input['T_in'].to_numpy()
    wb_in=unit_input['wb_in'].to_numpy()
    i_heat1_all=unit_input['i_heat1_all'].to_numpy()
    i_heat2_all=unit_input['i_heat2_all'].to_numpy()
    i_cool1=unit_input['i_cool1'].to_numpy()
    i_cool2=unit_input['i_cool2'].to_numpy()
    
    i_aux1=unit_input['i_aux1'].to_numpy()
    i_fan1=unit_input['i_fan1'].to_numpy()
    i_hc=i_heat1_all+i_heat2_all+i_cool1+i_cool2+i_aux1+i_fan1 # hc signal
    #i_hc=i_heat1_all+i_cool1+i_aux1+i_fan1 # hc signal

    if scale_const is None:
        scale_const=get_scale_const_template()
        print(f"Template scale_const is used. The values are {scale_const}. Please specify scale_const as a function input by calling `get_scale_const_template()` function.")  
    else:
        print(f'scale const is {scale_const}.')
    
    s_T_out=min_max_scale(T_out,x_min=scale_const['T_out_min'],x_max=scale_const['T_out_max'],
                  lower=scale_const['T_out_lower'],upper=scale_const['T_out_upper'])
    
    s_T_in=min_max_scale(T_in,x_min=scale_const['T_in_min'],x_max=scale_const['T_in_max'],
                  lower=scale_const['T_in_lower'],upper=scale_const['T_in_upper'])

    s_wb_in=min_max_scale(wb_in,x_min=scale_const['wb_in_min'],x_max=scale_const['wb_in_max'],
                  lower=scale_const['wb_in_lower'],upper=scale_const['wb_in_upper'])

    s_net=net/scale_const['net_max'] # 0-1 scale. Watt

    input_values={"net":net,
                  "s_net":s_net,
                  "T_out":T_out,
                  "s_T_out":s_T_out,
                  "T_in":T_in,
                  "s_T_in":s_T_in,
                  "wb_in":wb_in,
                  "s_wb_in":s_wb_in,
                  "i_heat1_all":i_heat1_all, 
                  "i_heat2_all":i_heat2_all,
                  "i_cool1":i_cool1,
                  "i_cool2":i_cool2,
                  "i_aux1":i_aux1,
                  "i_fan1":i_fan1,
                  "i_hc":i_hc,
                  "nan_index":nan_index,
                  "scale_const":scale_const,
                  "timestamp":timestamp,
                  "time_delta":time_delta
                  }
    # in case we have validation data
    if ('ahu' in unit_input.columns) and ('heatpump' in unit_input.columns):
        hc=unit_input['heatpump'].to_numpy()+unit_input['ahu'].to_numpy()
        input_values.update({"hc":hc})

    return input_values

