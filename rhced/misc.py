__all__=["query_unit_data","query_all_data"]

import numpy as np
import pandas as pd
import sqlite3
import pathlib

# this is internal use. 

def query_unit_data(unitcode,start_date,end_date=None,train_days=7):
    #print(pathlib.Path(__file__).parents[0].joinpath('data','overlook','temp_db.db').__str__())
    conn=sqlite3.connect(pathlib.Path(__file__).parents[1].joinpath('data','bldg1','temp_db.db').__str__())
    strftime_format="%Y-%m-%d %H:%M:%S" #filter time in this format
    start_date_utc=pd.Timestamp(start_date,tz="America/Indianapolis").tz_convert("UTC")
    if end_date is None:
        end_date_utc=(start_date_utc+pd.Timedelta(days=train_days))
    else:
        end_date_utc=pd.Timestamp(end_date,tz="America/Indianapolis").tz_convert("UTC")
    query_weather=f"SELECT * from WEATHER where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}'"
    data_weather= pd.read_sql_query(query_weather, conn)
    query_gem=f"SELECT * from GEM where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}' and unitcode='{unitcode}' "
    data_gem= pd.read_sql_query(query_gem, conn)
    query_ecobee=f"SELECT * from ECOBEE where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}' and unitcode='{unitcode}' "
    data_ecobee= pd.read_sql_query(query_ecobee, conn)
    conn.close()
    # convert timestamp to INDY time
    data_gem['timestamp']=pd.to_datetime(data_gem['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    data_weather['timestamp']=pd.to_datetime(data_weather['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    data_ecobee['timestamp']=pd.to_datetime(data_ecobee['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    # select columns
    data_gem['hvac']=data_gem['heatpump']+data_gem['ahu']
    data_gem=data_gem[['timestamp','unitcode','ahu','heatpump','hvac','net']]
    # wattsecond to watt 
    dtime=pd.Timedelta(data_gem['timestamp'][1]-data_gem['timestamp'][0]).seconds
    #print(dtime)
    data_gem=data_gem.apply(lambda x: x/dtime if x.name in ['ahu', 'heatpump','hvac','net'] else x)
    data_ecobee=data_ecobee[['timestamp','unitcode','operation','t_unit','rh_unit']]
    data_ecobee['t_unit']=((data_ecobee['t_unit'].to_numpy())-32)/1.8 # F to C
    data_ecobee['rh_unit']=((data_ecobee['rh_unit'].to_numpy()/100)) # % to -
    vec=data_ecobee['operation'].to_numpy()
    vec[vec=="heat"]="heat1"
    vec[vec=="cool"]="cool1"
    vec[vec=="aux"]="aux1"
    vec[vec=="heat_aux"]="heat1_aux1"
    data_ecobee['operation']=vec

    #data_ecobee['t_rs_m']=((data_ecobee['t_rs_m'].to_numpy())-32)/1.8 # F to C
    #data_ecobee['sp_heat']=((data_ecobee['sp_heat'].to_numpy())-32)/1.8 # F to C
    #data_ecobee['sp_cool']=((data_ecobee['sp_cool'].to_numpy())-32)/1.8 # F to C

    data_weather=data_weather[['timestamp','t_out','rh_out']]
    data_weather['t_out']=((data_weather['t_out'].to_numpy())-32)/1.8 # F to C
    data_weather['rh_out']=((data_weather['rh_out'].to_numpy()/100)) # % to -
    # join tables
    data_unit=pd.merge(pd.merge(data_gem,data_ecobee,on=['timestamp','unitcode'],how='left'),data_weather,on='timestamp',how='left')
    data_unit=data_unit.rename(columns={"t_out":"T_out"})
    data_unit=data_unit.rename(columns={"t_unit":"T_in"})
    data_unit=data_unit.rename(columns={"rh_unit":"rh_in"})
    #self.observed_model_input=data_unit[['timestamp','function','state','setpoint_cooling','setpoint_heating','t_out']].copy()
    
    return data_unit



def query_all_data(start_date,train_days=7):
    #print(pathlib.Path(__file__).parents[0].joinpath('data','overlook','temp_db.db').__str__())
    conn=sqlite3.connect(pathlib.Path(__file__).parents[1].joinpath('data','bldg1','temp_db.db').__str__())
    strftime_format="%Y-%m-%d %H:%M:%S" #filter time in this format
    start_date_utc=pd.Timestamp(start_date,tz="America/Indianapolis").tz_convert("UTC")
    end_date_utc=(start_date_utc+pd.Timedelta(days=train_days))
    query_weather=f"SELECT * from WEATHER where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}'"
    data_weather= pd.read_sql_query(query_weather, conn)
    query_gem=f"SELECT * from GEM where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}'"
    data_gem= pd.read_sql_query(query_gem, conn)
    query_ecobee=f"SELECT * from ECOBEE where timestamp>= '{start_date_utc.strftime(strftime_format)}'  and timestamp <'{end_date_utc.strftime(strftime_format)}'"
    data_ecobee= pd.read_sql_query(query_ecobee, conn)
    conn.close()
    # convert timestamp to INDY time
    data_gem['timestamp']=pd.to_datetime(data_gem['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    print(data_gem.head())
    data_weather['timestamp']=pd.to_datetime(data_weather['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    data_ecobee['timestamp']=pd.to_datetime(data_ecobee['timestamp'],utc=True).dt.tz_convert("America/Indianapolis")
    # select columns
    data_gem['hvac']=data_gem['heatpump']+data_gem['ahu']
    data_gem=data_gem[['timestamp','unitcode','ahu','heatpump','hvac','net']]
    # wattsecond to watt 
    dtime=300#pd.Timedelta(data_gem['timestamp'][1]-data_gem['timestamp'][0]).seconds
    #print(dtime)
    data_gem=data_gem.apply(lambda x: x/dtime if x.name in ['ahu', 'heatpump','hvac','net'] else x)
    data_ecobee=data_ecobee[['timestamp','unitcode','operation','t_unit','rh_unit']]
    data_ecobee['t_unit']=((data_ecobee['t_unit'].to_numpy())-32)/1.8 # F to C
    data_ecobee['rh_unit']=((data_ecobee['rh_unit'].to_numpy()/100)) # % to -
    
    vec=data_ecobee['operation'].to_numpy()
    vec[vec=="heat"]="heat1"
    vec[vec=="cool"]="cool1"
    vec[vec=="aux"]="aux1"
    vec[vec=="heat_aux"]="heat1_aux1"
    data_ecobee['operation']=vec

    #data_ecobee['t_rs_m']=((data_ecobee['t_rs_m'].to_numpy())-32)/1.8 # F to C
    #data_ecobee['sp_heat']=((data_ecobee['sp_heat'].to_numpy())-32)/1.8 # F to C
    #data_ecobee['sp_cool']=((data_ecobee['sp_cool'].to_numpy())-32)/1.8 # F to C

    data_weather=data_weather[['timestamp','t_out','rh_out']]
    data_weather['t_out']=((data_weather['t_out'].to_numpy())-32)/1.8 # F to C
    data_weather['rh_out']=((data_weather['rh_out'].to_numpy()/100)) # % to -

    # join tables
    data_unit=pd.merge(pd.merge(data_gem,data_ecobee,on=['timestamp','unitcode'],how='left'),data_weather,on='timestamp',how='left')
    data_unit=data_unit.rename(columns={"t_out":"T_out"})
    #self.observed_model_input=data_unit[['timestamp','function','state','setpoint_cooling','setpoint_heating','t_out']].copy()
    return data_unit