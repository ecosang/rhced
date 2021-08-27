

rm(list=ls())

library(tidyverse)
library(ggplot2)
library(patchwork)
library(lubridate)


#library(RcppCNPy)


#outputs\bldg1\a10\update


# weeks


#start_date="2021-03-08"
#end_date="2021-03-14"
get_cal=function(file_path){
  df=arrow::read_feather(file_path)
  
  df=df%>%mutate(timestamp=ymd_hms(timestamp))
  df=df%>%mutate(ihc=hc*i_hc)
  
  df_boxplot=df%>%select(-timestamp)%>%summarize_all(list(~mean(.,na.rm=T)/1000*24*7))
  df_boxplot_hc=df_boxplot%>%select(contains("P_hc"))
  colnames(df_boxplot_hc)=(df_boxplot_hc%>%colnames()%>%str_replace("P+_+\\w+_+",""))
  
  
  obs_P_hc=mean(df$hc*df$i_hc,na.rm=T)/1000*24*7
  
  
  
  i_heat1=sum(df$i_heat1[!is.na(df$i_heat1)])/2
  i_cool1=sum(df$i_cool1[!is.na(df$i_cool1)])/2
  i_df1=sum(df$i_df1[!is.na(df$i_df1)])/2
  i_aux1=sum(df$i_aux1[!is.na(df$i_aux1)])/2
  
  print(paste0("heat hour: ",round(i_heat1,2), " cool hour: ",round(i_cool1,2)," df hour: ",round(i_df1,2), " aux hour: ",round(i_aux1,2)," hc hour: ",round(i_heat1+i_cool1+i_df1+i_aux1,1) ))
  print(paste0("median prediction: ",round(df_boxplot_hc$mid,2),"kwh obs: ",round(obs_P_hc,2),"kwh"))
  
  
  
  return(list(df=df_boxplot_hc,obs=obs_P_hc))
  
}





# outputs_2018-01-28_2018-02-03.feather
# outputs_2018-02-04_2018-02-10.feather
# outputs_2018-02-11_2018-02-17.feather
# outputs_2018-02-25_2018-03-03.feather
# outputs_2018-03-04_2018-03-10.feather
# outputs_2018-03-11_2018-03-17.feather
# outputs_2018-03-18_2018-03-24.feather
# outputs_2018-03-25_2018-03-31.feather

file_list=c("outputs_2018-01-28_2018-02-03.feather",
            "outputs_2018-02-04_2018-02-10.feather",
            "outputs_2018-02-11_2018-02-17.feather",
            "outputs_2018-02-25_2018-03-03.feather",
            "outputs_2018-03-04_2018-03-10.feather",
            "outputs_2018-03-11_2018-03-17.feather",
            "outputs_2018-03-18_2018-03-24.feather",
            "outputs_2018-03-25_2018-03-31.feather")

#obs=1:length(file_list)


df_list_one_day=list()
obs=rep(0,length(file_list))
i=1
for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/single_day_model/",file_list[i])
  
  l1=get_cal(file_path)
  df_list_one_day[[i]]=l1$df%>%mutate(week=paste0("week",i),data="1day")
  obs[i]=l1$obs
  rm(l1)
}
df_one_day=bind_rows(df_list_one_day)
write_rds(df_one_day,"outputs/figures/df_one_day.rds")


df_list_two_day=list()
obs=rep(0,length(file_list))

for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/two_day_model/",file_list[i])
  
  l1=get_cal(file_path)
  df_list_two_day[[i]]=l1$df%>%mutate(week=paste0("week",i),data="2days")
  obs[i]=l1$obs
  rm(l1)
}
df_two_day=bind_rows(df_list_two_day)
write_rds(df_two_day,"outputs/figures/df_two_day.rds")



df_list_four_day=list()
obs=rep(0,length(file_list))
i=1
for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/four_day_model/",file_list[i])
  
  l1=get_cal(file_path)
  df_list_four_day[[i]]=l1$df%>%mutate(week=paste0("week",i),data="4days")
  obs[i]=l1$obs
  rm(l1)
}
df_four_day=bind_rows(df_list_four_day)
write_rds(df_four_day,"outputs/figures/df_four_day.rds")




df_list_one_week=list()
for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/single_unit_analysis/",file_list[i])
  l1=get_cal(file_path)
  df_list_one_week[[i]]=l1$df%>%mutate(week=paste0("week",i),data="1week")
  rm(l1)
}

df_one_week=bind_rows(df_list_one_week)
write_rds(df_one_week,"outputs/figures/df_one_week.rds")


df_list_two_week=list()
for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/two_weeks_model/",file_list[i])
  l1=get_cal(file_path)
  df_list_two_week[[i]]=l1$df%>%mutate(week=paste0("week",i),data="2weeks")
  rm(l1)
}

df_two_week=bind_rows(df_list_two_week)
write_rds(df_two_week,"outputs/figures/df_two_week.rds")


df_list_three_week=list()
for (i in 1:length(file_list)){
  file_path=paste0("outputs/bldg1/a10/three_weeks_model/",file_list[i])
  l1=get_cal(file_path)
  df_list_three_week[[i]]=l1$df%>%mutate(week=paste0("week",i),data="3weeks")
  rm(l1)
}

df_three_week=bind_rows(df_list_three_week)
write_rds(df_three_week,"outputs/figures/df_three_week.rds")

df_one_day=read_rds("outputs/figures/df_one_day.rds")
df_two_day=read_rds("outputs/figures/df_two_day.rds")
df_four_day=read_rds("outputs/figures/df_four_day.rds")
df_one_week=read_rds("outputs/figures/df_one_week.rds")
df_two_week=read_rds("outputs/figures/df_two_week.rds")
df_three_week=read_rds("outputs/figures/df_three_week.rds")


full_df=bind_rows(df_one_day,df_two_day,df_four_day,df_one_week,df_two_week,df_three_week)

full_df$data=factor(full_df$data,levels = c("1day","2days","4days","1week","2weeks","3weeks"))

df_boxplot2=tibble(week=df_one_day$week,value=obs)#%>%mutate(week="")

ggplot(full_df,aes(x=week,group=interaction(week,data)))+
  geom_boxplot(aes(fill=data,ymin=lower,ymax=upper,lower=lq,middle=mid,upper=uq),alpha=0.3,width=0.5, stat = "identity")+
  theme(axis.text.x = element_text(size=10), 
        axis.text.y = element_text(size=10),
        axis.title.y= element_blank(),
        axis.title.x= element_blank(),#element_blank(),
        legend.position=c(0.8,.8),
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0),limits=c(0,250))

ggsave("outputs/figures/update1.jpg",dpi=300,width=8,height=5)

ggplot(data=df_boxplot2)+geom_point(aes(week,value),shape=4,size=5,color="red")+
  theme(axis.text.x = element_text(size=10), 
        axis.text.y = element_text(size=10),
        axis.title.y= element_blank(),
        axis.title.x= element_blank(),#element_blank(),
        legend.position="right",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0),limits=c(0,250))

ggsave("outputs/figures/update2.jpg",dpi=300,width=8,height=5)
