rm(list=ls())

library(tidyverse)
library(patchwork)
library(lubridate)
library(arrow)
library(RcppCNPy)
library(Rfast)
# result figure #1 time series plot 

getwd()

start_date="2018-01-28"
end_date="2018-02-03"

#df=feather::read_feather(paste0("outputs/bldg1/a10/single_unit_analysis/outputs_",start_date,"_",end_date,".feather"))
df=arrow::read_feather(paste0("outputs/bldg1/a10/single_unit_analysis/outputs_",start_date,"_",end_date,".feather"))

lubridate::yday
df=df%>%mutate(yday=yday(timestamp))
day_idx=df$yday%>%unique()
df=df%>%mutate(timestamp=ymd_hms(timestamp))
num_days=7
#df=df%>%filter(yday==day_idx[2])

df=df%>%mutate(ihc=hc*i_hc)
# ,color="red"
g1=ggplot(df,aes(x=timestamp))+geom_ribbon(aes(ymin=P_hc_lower,ymax=P_hc_upper),fill="lightskyblue2",alpha=0.8)+
  geom_line(aes(y=P_hc_mid),color="skyblue4",linetype = "longdash",alpha=0.8)+
  geom_line(aes(y=ihc),color="red3",linetype="solid",size=.5,alpha=1.0)+
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
  )+scale_y_continuous(expand=c(0,0))




df_boxplot=df%>%select(-timestamp)%>%summarize_all(list(~mean(.,na.rm=T)/1000*24*num_days))

df_boxplot_hc=df_boxplot%>%select(contains("P_hc"))
colnames(df_boxplot_hc)=(df_boxplot_hc%>%colnames()%>%str_replace("P+_+\\w+_+",""))

df_boxplot_heat1=df_boxplot%>%select(contains("P_heat1"))
colnames(df_boxplot_heat1)=(df_boxplot_heat1%>%colnames()%>%str_replace("P+_+\\w+_+",""))

df_boxplot_df1=df_boxplot%>%select(contains("P_df1"))
colnames(df_boxplot_df1)=(df_boxplot_df1%>%colnames()%>%str_replace("P+_+\\w+_+",""))
df_boxplot_aux1=df_boxplot%>%select(contains("P_aux1"))
colnames(df_boxplot_aux1)=(df_boxplot_aux1%>%colnames()%>%str_replace("P+_+\\w+_+",""))


df_boxplot_=bind_rows(df_boxplot_hc%>%mutate(name="c01"),
          df_boxplot_heat1%>%mutate(name="c02"),
          df_boxplot_df1%>%mutate(name="c03"),
          df_boxplot_aux1%>%mutate(name="c04")
          )

obs_P_hc=mean(df$hc*df$i_hc,na.rm=T)/1000*24*num_days
obs_P_heat1=mean(df$hc*df$i_heat1,na.rm=T)/1000*24*num_days
obs_P_df1=mean(df$hc*df$i_df1,na.rm=T)/1000*24*num_days
obs_P_aux1=mean(df$hc*df$i_aux1,na.rm=T)/1000*24*num_days
#df_boxplot=tibble(c01=P_hc_sample,c02=P_heat1_sample,c03=P_df1_sample,c04=P_aux1_sample)%>%pivot_longer(cols=everything())
df_boxplot2=tibble(c01=obs_P_hc,c02=obs_P_heat1,c03=obs_P_df1,c04=obs_P_aux1)%>%pivot_longer(cols=everything())


g2=ggplot(df_boxplot_,aes(x=name,group=name))+geom_boxplot(aes(ymin=lower,ymax=upper,lower=lq,middle=mid,upper=uq),lwd =.5,alpha=0.5,width=0.5,fill="lightskyblue2",color="black", stat = "identity")+geom_point(data=df_boxplot2,aes(name,value),shape=4,size=5,color="red")+
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
  )+scale_y_continuous(expand=c(0,0))




g1+g2+plot_layout(width = c(1,0.25))
#ggsave("outputs/figures/result1_a10_2018-01-28_2018-01-28.jpg",dpi=300,width=12,height=3.5)
#ggsave("outputs/figures/result1_a10_2018-01-29_2018-01-29.jpg",dpi=300,width=12,height=3.5)
ggsave("outputs/figures/result1_a10_2018-01-28_2018-02-03.jpg",dpi=300,width=12,height=3.5)


i_heat1=sum(df$i_heat1[!is.na(df$i_heat1)])/2
i_cool1=sum(df$i_cool1[!is.na(df$i_cool1)])/2
i_df1=sum(df$i_df1[!is.na(df$i_df1)])/2
i_aux1=sum(df$i_aux1[!is.na(df$i_aux1)])/2

print(paste0("heat hour: ",round(i_heat1,2), " cool hour: ",round(i_cool1,2)," df hour: ",round(i_df1,2), " aux hour: ",round(i_aux1,2)," hc hour: ",round(i_heat1+i_cool1+i_df1+i_aux1,1) ))
print(paste0("median prediction: ",round(df_boxplot_hc$mid,2),"kwh obs: ",round(obs_P_hc,2),"kwh"))


rm(list=ls())

library(tidyverse)
library(patchwork)
library(lubridate)
library(arrow)
#library(RcppCNPy)
library(Rfast)
# result figure #1 time series plot 

getwd()

start_date="2018-01-28"
end_date="2018-02-03"
df=arrow::read_feather(paste0("outputs/bldg1/a10/single_unit_analysis/outputs_",start_date,"_",end_date,".feather"))

lubridate::yday
df=df%>%mutate(yday=yday(timestamp))
day_idx=df$yday%>%unique()
df=df%>%mutate(timestamp=ymd_hms(timestamp))
num_days=2 
df=df%>%filter(yday%in%day_idx[1:2])

df=df%>%mutate(ihc=hc*i_hc)
# ,color="red"
g1=ggplot(df,aes(x=timestamp))+geom_ribbon(aes(ymin=P_hc_lower,ymax=P_hc_upper),fill="lightskyblue2",alpha=0.8)+
  geom_line(aes(y=P_hc_mid),color="skyblue4",linetype = "longdash",alpha=0.8)+
  geom_line(aes(y=ihc),color="red3",linetype="solid",size=.5,alpha=1.0)+
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
  )+scale_y_continuous(expand=c(0,0))

ggsave("outputs/figures/result1_a10_2018-01-28_2018-01-29.jpg",dpi=300,width=9,height=2.5)



##############################33
rm(list=ls())
# outputs_2018-02-18_2018-02-24
#outputs_2018-02-11_2018-02-17
# outputs_2018-02-04_2018-02-10
start_date="2018-02-18"
end_date="2018-02-24"
df=arrow::read_feather(paste0("outputs/bldg1/a10/single_unit_analysis/outputs_",start_date,"_",end_date,".feather"))



df=df%>%mutate(timestamp=ymd_hms(timestamp))
df=df%>%mutate(ihc=hc*i_hc)
# ,color="red"
g1=ggplot(df,aes(x=timestamp))+geom_ribbon(aes(ymin=P_hc_lower,ymax=P_hc_upper),fill="lightskyblue2",alpha=0.8)+
  geom_line(aes(y=P_hc_mid),color="skyblue4",linetype = "longdash",alpha=0.8)+
  geom_line(aes(y=ihc),color="red3",linetype="solid",size=.5,alpha=1.0)+
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
  )+scale_y_continuous(expand=c(0,0))




df_boxplot=df%>%select(-timestamp)%>%summarize_all(list(~mean(.,na.rm=T)/1000*24*7))

df_boxplot_hc=df_boxplot%>%select(contains("P_hc"))
colnames(df_boxplot_hc)=(df_boxplot_hc%>%colnames()%>%str_replace("P+_+\\w+_+",""))

df_boxplot_heat1=df_boxplot%>%select(contains("P_heat1"))
colnames(df_boxplot_heat1)=(df_boxplot_heat1%>%colnames()%>%str_replace("P+_+\\w+_+",""))

df_boxplot_df1=df_boxplot%>%select(contains("P_df1"))
colnames(df_boxplot_df1)=(df_boxplot_df1%>%colnames()%>%str_replace("P+_+\\w+_+",""))
df_boxplot_aux1=df_boxplot%>%select(contains("P_aux1"))
colnames(df_boxplot_aux1)=(df_boxplot_aux1%>%colnames()%>%str_replace("P+_+\\w+_+",""))


df_boxplot_=bind_rows(df_boxplot_hc%>%mutate(name="c01"),
                      df_boxplot_heat1%>%mutate(name="c02"),
                      df_boxplot_df1%>%mutate(name="c03"),
                      df_boxplot_aux1%>%mutate(name="c04")
)

obs_P_hc=mean(df$hc*df$i_hc,na.rm=T)/1000*24*7
obs_P_heat1=mean(df$hc*df$i_heat1,na.rm=T)/1000*24*7
obs_P_df1=mean(df$hc*df$i_df1,na.rm=T)/1000*24*7
obs_P_aux1=mean(df$hc*df$i_aux1,na.rm=T)/1000*24*7
#df_boxplot=tibble(c01=P_hc_sample,c02=P_heat1_sample,c03=P_df1_sample,c04=P_aux1_sample)%>%pivot_longer(cols=everything())
df_boxplot2=tibble(c01=obs_P_hc,c02=obs_P_heat1,c03=obs_P_df1,c04=obs_P_aux1)%>%pivot_longer(cols=everything())


g2=ggplot(df_boxplot_,aes(x=name,group=name))+geom_boxplot(aes(ymin=lower,ymax=upper,lower=lq,middle=mid,upper=uq),lwd =.5,alpha=0.5,width=0.5,fill="lightskyblue2",color="black", stat = "identity")+geom_point(data=df_boxplot2,aes(name,value),shape=4,size=5,color="red")+
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
  )+scale_y_continuous(expand=c(0,0))




g1+g2+plot_layout(width = c(1,0.25))
ggsave(paste0("outputs/figures/result1_a10_",start_date,"_",end_date,".jpg"),dpi=300,width=12,height=2.5)


i_heat1=sum(df$i_heat1[!is.na(df$i_heat1)])/2
i_cool1=sum(df$i_cool1[!is.na(df$i_cool1)])/2
i_df1=sum(df$i_df1[!is.na(df$i_df1)])/2
i_aux1=sum(df$i_aux1[!is.na(df$i_aux1)])/2

print(paste0("heat hour: ",round(i_heat1,2), " cool hour: ",round(i_cool1,2)," df hour: ",round(i_df1,2), " aux hour: ",round(i_aux1,2)," hc hour: ",round(i_heat1+i_cool1+i_df1+i_aux1,1) ))
print(paste0("median prediction: ",round(df_boxplot_hc$mid,2),"kwh obs: ",round(obs_P_hc,2),"kwh"))




# print(paste0("cool hour: ",round(i_cool1,2)))
# print(paste0("df hour: ",round(i_df1,2)))
# print(paste0("aux hour: ",round(i_aux1,2)))
# print(paste0("hc hour: ",round(i_heat1+i_cool1+i_df1+i_aux1,1)))





#P_aux1=RcppCNPy::npyLoad("outputs/bldg1/a10/single_unit_analysis/P_aux1_2018-01-28_2018-02-03.npy")
#P_heat1=RcppCNPy::npyLoad("outputs/bldg1/a10/single_unit_analysis/P_heat1_2018-01-28_2018-02-03.npy")
#P_df1=RcppCNPy::npyLoad("outputs/bldg1/a10/single_unit_analysis/P_df1_2018-01-28_2018-02-03.npy")
#P_hc=P_aux1+P_heat1+P_df1
#summary(P_hc)
#P_hc%>%as.vector()%>%quantile(probs=c(0.025,0.5,0.975))

# i_heat1=df$i_heat1[!is.na(df$i_heat1)]
# #P_heat1_sample=((P_heat1[,i_heat1==1]%>%rowMeans())/1000*24*7)
# P_heat1_sample=((P_heat1%>%rowMeans())/1000*24*7)
# summary(P_heat1)
# P_heat1_sample
# i_df1=df$i_df1[!is.na(df$i_df1)]
# P_df1_sample=((P_df1%>%rowMeans())/1000*24*7)
# 
# i_aux1=df$i_aux1[!is.na(df$i_aux1)]
# P_aux1_sample=((P_aux1%>%rowMeans())/1000*24*7)
# 
# P_hc_sample=P_heat1_sample+P_df1_sample+P_aux1_sample
# 
# dd=tibble(x_=c(1,2,3,4),ymin_=c(1,2,3,4),ymax_=c(6,7,8,9),lower_=c(2,3,4,5),upper_=c(4,5,6,7),middle_=c(3,4,5,6))
# 
# ggplot(dd,aes(x=x_,group=x_))+geom_boxplot(aes(ymin=ymin_,ymax=ymax_,lower=lower_,middle=middle_,upper=upper_), stat = "identity")
# 
# 
# #ggplot()+geom_boxplot(aes(x=1, y = 2:5, lower = 3, upper = 4, middle = 3.5, ymin=2, ymax=5))
# 
# 
# obs_P_hc=mean(df$hc*df$i_hc,na.rm=T)/1000*24*7
# obs_P_heat1=mean(df$hc*df$i_heat1,na.rm=T)/1000*24*7
# obs_P_df1=mean(df$hc*df$i_df1,na.rm=T)/1000*24*7
# obs_P_aux1=mean(df$hc*df$i_aux1,na.rm=T)/1000*24*7
# df_boxplot=tibble(c01=P_hc_sample,c02=P_heat1_sample,c03=P_df1_sample,c04=P_aux1_sample)%>%pivot_longer(cols=everything())
# df_boxplot2=tibble(c01=obs_P_hc,c02=obs_P_heat1,c03=obs_P_df1,c04=obs_P_aux1)%>%pivot_longer(cols=everything())
# ggplot(df_boxplot,aes(name,value,color=name))+geom_boxplot(color="black")+geom_point(data=df_boxplot2,aes(name,value),shape=4,size=5,color="red")+
#   theme(axis.text.x = element_text(size=10), 
#         axis.text.y = element_text(size=10),
#         axis.title.y= element_blank(),
#         axis.title.x= element_blank(),#element_blank(),
#         legend.position="right",
#         legend.direction="vertical",
#         legend.text=element_text(size=10),
#         panel.background = element_rect(fill  = "white", colour = "grey80"),
#         panel.grid.major.y  = element_line(colour = "grey80",size=.3),
#         panel.grid.minor.y= element_line(colour = "grey80",size=.2),
#         panel.grid.major.x= element_line(colour = "grey80",size=.3)
#   )+scale_y_continuous(expand=c(0,0))
# 
# 
# summary(P_heat1_sample)
# obs_P_hc
# 
# obs_P_heat1
# quantile(P_heat1_sample,probs=c(0.025,0.5,0.975))
# quantile(P_df1_sample,probs=c(0.025,0.5,0.975))
# quantile(P_aux1_sample,probs=c(0.025,0.5,0.975))
# # 