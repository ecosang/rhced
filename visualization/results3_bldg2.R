
rm(list=ls())

library(tidyverse)
library(patchwork)
library(lubridate)
library(arrow)
library(RcppCNPy)
library(Rfast)
# result figure #1 time series plot 

getwd()

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





file_list=c("outputs_2021-03-08_2021-03-14.feather",
            "outputs_2021-03-15_2021-03-21.feather",
            "outputs_2021-03-22_2021-03-28.feather",
            "outputs_2021-03-29_2021-04-04.feather",
            "outputs_2021-04-05_2021-04-11.feather",
            "outputs_2021-04-12_2021-04-18.feather")
unitcode_vec=c("na03","nb06","nb08","sa03","sb01","sb05")
#obs=1:length(file_list)

#file_path=paste0("outputs/bldg1/","b17","/single_unit_analysis/","outputs_2018-02-18_2018-02-24.feather")
#l1=get_cal(file_path)



#obs=rep(0,length(file_list))
#i=1
df_list=list()


for (j in 1:length(unitcode_vec)){
  for (i in 1:length(file_list)){
    file_path=paste0("outputs/bldg2/",unitcode_vec[j],"/single_unit_analysis/",file_list[i])
    
    l1=get_cal(file_path)
    df_list[[(length(unitcode_vec)*j+i)]]=l1$df%>%mutate(week=paste0("week",i),code=unitcode_vec[j],obs=l1$obs)
    #obs[i]=l1$obs
    rm(l1)
  }
  
}
df_=bind_rows(df_list)
df_
write_rds(df_,"outputs/figures/df_bldg2.rds")

df_=read_rds("outputs/figures/df_bldg2.rds")


for (i in 1:length(unitcode_vec)){
  unit_df=df_%>%filter(code==unitcode_vec[i])
  assign(paste0("g",i),
         ggplot(df_%>%filter(code==unitcode_vec[i]),aes(x=week,group=week))+
           geom_boxplot(aes(ymin=lower,ymax=upper,lower=lq,middle=mid,upper=uq),color="darkblue",width=0.5, stat = "identity")+
           geom_point(aes(x=week,y=obs),shape=4,size=5,color="red")+
           labs(title = unitcode_vec[i],y="Electricity [kWh]")+
           theme(axis.text.x = element_text(size=10), 
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
           )+scale_y_continuous(expand=c(0,0),limits=c(0,round(max(unit_df$upper+5),0))) # ,limits=c(0,250)
  )
  
}

ggsave("outputs/figures/bldg2_multiple_units.png",dpi=300,width=12,height=6,(g1+g2+g3+g4+g5+g6))


