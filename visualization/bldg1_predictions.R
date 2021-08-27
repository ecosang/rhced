


rm(list=ls())

library(tidyverse)
library(ggplot2)
library(patchwork)
library(lubridate)


# weeks



#dd=tibble(x=c('1','1','2','2'),ymin_=c(1,2,3,4),ymax_=c(6,7,8,9),lower_=c(2,3,4,5),upper_=c(4,5,6,7),middle_=c(3,4,5,6),mark=c("c","d","c","d"))
#ggplot(dd,aes(x=x,group=interaction(x,mark)))+geom_boxplot(aes(fill=mark,ymin=ymin_,ymax=ymax_,lower=lower_,middle=middle_,upper=upper_),width=0.5, stat = "identity")


#start_date="2021-03-08"
#end_date="2021-03-14"
get_cal=function(file_path){
  df2=arrow::read_feather(file_path)
  df=df2
  rm(df2)
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
list_temp=list.files('outputs/bldg1/a10/single_unit_analysis/',full.names = T)
file_list=list_temp[list_temp%>%str_detect("outputs_")]


#obs=1:length(file_list)


df_list=list()
obs=rep(0,length(file_list))

for (i in 1:length(file_list)){
  #file_path=paste0("outputs/bldg1/a10/single_day_model/",file_list[i])
  file_path=file_list[i]
  l1=get_cal(file_path)
  start_date=file_list[i]%>%str_extract("\\d*-\\d*-\\d*")
  df_list[[i]]=l1$df%>%mutate(start_date=start_date)
  obs[i]=l1$obs
  rm(l1)
}
df_=bind_rows(df_list)

write_rds(df_,"outputs/figures/bldg1/for_figures/df_a10.rds")

df_=read_rds("outputs/figures/bldg1/for_figures/df_a10.rds")

library(ggplot2)
ggplot(df_,aes(x=start_date,group=start_date))+
  geom_boxplot(aes(ymin=lower,ymax=upper,lower=lq,middle=mid,upper=uq),color="darkblue",width=0.5, stat = "identity")+
  geom_point(aes(x=start_date,y=obs),shape=4,size=5,color="red")+
  labs(y="Electricity [kWh]")+
  theme(axis.text.x = element_text(size=10,angle = 90, hjust = 1,vjust=.5), 
        axis.text.y = element_text(size=10),
        #axis.title.y= element_blank(),
        axis.title.x= element_blank(),#element_blank(),
        legend.position=c(0.9,.8),
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0)) # ,limits=c(0,250)
ggsave("outputs/figures/bldg1_a10_multiple_weeks.png",dpi=300,width=12,height=4)
