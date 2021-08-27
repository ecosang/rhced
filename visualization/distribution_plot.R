library(tidyverse)
library(patchwork)
library(lubridate)
# figure for 
#df=read_csv("data/overlook/a10/raw_data/unit_data_a10_2018-01-14_2018-01-20.csv")%>%select(timestamp,ahu,heatpump,net,T_out,operation)%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))


df=read_csv("data/private/overlook_all_2018-01-14_2018-09-01.csv")%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))

df=df%>%rename(hc=hvac)%>%mutate(month=month(timestamp))%>%mutate(misc=net-hc)%>%mutate(month=month(timestamp))
df$month%>%unique()
df=df%>%mutate(season=case_when(month%in%c(1,2,3)~"heating",
                             month%in%c(4,5,6)~"transition",
                             month%in%c(7,8,9)~"cooling",
                             TRUE~NA_character_))

#df_check=df%>%filter(!is.na(net)&operation!="ERROR")
# remove outliers
df=df%>%filter(!(is.na(net)|operation=="ERROR"|is.na(operation)))
df=df%>%group_by(unitcode)%>%distinct(timestamp,.keep_all=TRUE)%>%ungroup()
df=df%>%filter(misc>0&hc>0&hc<45000)

df_check=df
df_count=df_check%>%group_by(unitcode,month)%>%summarize(n=n())
df_count=df_count%>%ungroup()%>%pivot_wider(names_from=month,values_from=n)
View(df_count)
# View(df_count)


#a11(2bed) a10(1bed) # b17 corner #c37 top floor #c39 top floor # c49 corner

#df_unit=df%>%mutate(month=month(timestamp))%>%filter(unitcode=="a3")%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))
df
df_density=df%>%filter(unitcode%in%c("a11","a10","b17","c37","c49"))%>%select(unitcode,hc,misc,season,operation)%>%pivot_longer(cols=c(hc,misc))
df_density=df_density%>%filter(!(name=="hc"&operation%in%c("idle","ERROR")))

df_density
# a11(2bed) a10(1bed) # b17 corner #c37 top floor #c39 top floor # c49 corner


cols <- c("cooling"="#619CFF","heating"="#F8766D","transition"="#00BA38")
#df_density%>%mutate(value=log(value/1))
g1=ggplot(df_density%>%mutate(value=(value/1)),aes(x=value,color=season,group=season,fill=season))+geom_density(alpha=0.1)+
  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  facet_grid(name~unitcode,scales="free")+
  theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.92,.3), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0),limits=c(0,0.005)) +scale_x_continuous(expand=c(0,0),limits=c(0,5000)) # 

ggsave("outputs/figures/fig_3_distribution.png",dpi=300,width=10,height=5.0,g1)


#df_t_out%>%group_by(season,unitcode)%>%sample_n(5000)%>%ungroup()
df_t_out=df%>%filter(unitcode%in%c("a11","a10","b17","c37","c49"))
df_t_out%>%select(unitcode,net,T_out,season)#%>%pivot_longer(cols=c(season))



cols <- c("cooling"="#619CFF","heating"="#F8766D","transition"="#00BA38")

ggplot(df_t_out%>%group_by(season,unitcode)%>%sample_n(5000)%>%ungroup(),aes(T_out,net))+geom_point(aes(color=season),alpha=0.3)+
  facet_grid(season~unitcode,scales="free")+
  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  theme(axis.text.x = element_text(size=10,angle=0,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.865,.85), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0),limits=c(0,8000)) +scale_x_continuous(expand=c(0,0)) # ,limits=c(0,5000)


ggsave("outputs/figures/fig_t_out_net.png",dpi=300,width=11,height=5)

cols <- c("cooling"="#619CFF","heating"="#F8766D","transition"="#00BA38")
ggplot(df_t_out%>%group_by(season,unitcode)%>%sample_n(5000)%>%ungroup(),aes(T_out,hc))+geom_point(aes(color=season),alpha=0.3)+
  facet_grid(season~unitcode,scales="free")+
  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  theme(axis.text.x = element_text(size=10,angle=0,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.865,.85), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0),limits=c(0,8000)) +scale_x_continuous(expand=c(0,0)) # ,limits=c(0,5000)

ggsave("outputs/figures/fig_t_out_hc.png",dpi=300,width=11,height=5)



ggplot(df_t_out%>%group_by(season,unitcode)%>%sample_n(5000)%>%ungroup(),aes(T_out,misc))+geom_point(aes(color=season),alpha=0.3)+
  facet_grid(season~unitcode,scales="free")+
  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  theme(axis.text.x = element_text(size=10,angle=0,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.865,.85), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0),limits=c(0,8000)) +scale_x_continuous(expand=c(0,0)) # ,limits=c(0,5000)

ggsave("outputs/figures/fig_t_out_misc.png",dpi=300,width=11,height=5)

# df_density
# g11=ggplot(df_density%>%mutate(value=(value/1))%>%filter(name=="hc"),aes(x=value,color=season,group=season,fill=season))+geom_density(alpha=0.1)+
#   scale_fill_manual(values = cols)+
#   scale_color_manual(values = cols)+
#   facet_grid(cols=vars(unitcode),scales="free")+
#   theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
#         axis.text.y = element_text(size=10),
#         axis.title.y= element_text(size=12),
#         axis.title.x= element_text(size=12),#element_blank(),
#         legend.position=c(0.72,.7), #"right"
#         legend.direction="vertical",
#         legend.text=element_text(size=8),
#         panel.background = element_rect(fill  = "white", colour = "grey80"),
#         panel.grid.major.y  = element_line(colour = "grey80",size=.3),
#         panel.grid.minor.y= element_line(colour = "grey80",size=.2),
#         panel.grid.major.x= element_line(colour = "grey80",size=.3),
#         panel.spacing = unit(1, "lines")
#   )+scale_y_continuous(expand=c(0,0)) +scale_x_continuous(expand=c(0,0),limits=c(0,5000)) # 
# 
# g12=ggplot(df_density%>%mutate(value=(value/1))%>%filter(name=="misc"),aes(x=value,color=season,group=season,fill=season))+geom_density(alpha=0.1)+
#   scale_fill_manual(values = cols)+
#   scale_color_manual(values = cols)+
#   facet_grid(cols=vars(unitcode),scales="free")+
#   theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
#         axis.text.y = element_text(size=10),
#         axis.title.y= element_text(size=12),
#         axis.title.x= element_text(size=12),#element_blank(),
#         legend.position=c(0.72,.7), #"right"
#         legend.direction="vertical",
#         legend.text=element_text(size=8),
#         panel.background = element_rect(fill  = "white", colour = "grey80"),
#         panel.grid.major.y  = element_line(colour = "grey80",size=.3),
#         panel.grid.minor.y= element_line(colour = "grey80",size=.2),
#         panel.grid.major.x= element_line(colour = "grey80",size=.3),
#         panel.spacing = unit(1, "lines")
#   )+scale_y_continuous(expand=c(0,0),limits=c(0,0.006)) +scale_x_continuous(expand=c(0,0),limits=c(0,6000)) # 

#cols = vars(cyl)

#df_density%>%filter(unitcode=="a10")%>%filter(name=="misc")%>%.$value%>%plot()



##################### Heatpump outdoor air temperatrue #######################################3

df_hc=df%>%filter(unitcode%in%c("a11","a10","b17","c37","c49"))%>%select(timestamp,unitcode,heatpump,misc,season,operation,T_out)#%>%pivot_longer(cols=c(hc,misc))

df_hc1=df_hc%>%filter(operation%in%c("heat1","cool1"))%>%select(unitcode,heatpump,operation,T_out)

df_hc=df_hc%>%mutate(i_heat1=if_else(operation=="heat1",1,0),i_cool1=if_else(operation=="cool1",1,0))
df_hc=df_hc%>%mutate(timestamp30=floor_date(timestamp,unit="30min"))

# df_hc30=df_hc%>%group_by(unitcode,timestamp30)%>%summarize(hc=mean(heatpump,na.rm=T),T_out=mean(T_out,na.rm=T),i_heat1=mean(i_heat1,na.rm=T),i_cool1=mean(i_cool1,na.rm=T))%>%ungroup()
# df_hc30
# df_hc30=df_hc30%>%mutate(operation=case_when(i_heat1>0~"heat",i_cool1>0~"cool",TRUE~NA_character_))
# df_hc30=df_hc30%>%filter(!is.na(operation))

df_hc1=df_hc1%>%mutate(operation=case_when(operation=="heat1"~"heat",operation=="cool1"~"cool",TRUE~NA_character_))
cols <- c("cool"="#619CFF","heat"="#F8766D")
g3=ggplot(df_hc1,aes(x=T_out,y=heatpump,color=operation,group=operation,fill=operation))+geom_point(alpha=0.3)+
  facet_grid(operation~unitcode,scales="free")+  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.92,.3), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        legend.title=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0)) +scale_x_continuous(expand=c(0,0)) 


ggsave("outputs/figures/fig_5_distribution.png",dpi=300,width=12,height=4.5,g3)

sigmoid=function(x){
  return(1/(1+exp(-x)))
}
hist(log10(max(x+1) - x),breaks=100) 
hist(1/(max(x+1) - x),breaks=100)
x=(df_hc1%>%filter(operation=="heat"&unitcode=="a10")%>%.$heatpump)/4000
#x=(df_hc1%>%filter(operation=="heat"&unitcode=="a10")%>%.$ahu)/4000

par(mfrow=c(1,2))
hist(x,breaks=100)
hist(exp(x),breaks=100)

hist(((df_hc1%>%filter(operation=="heat"&unitcode=="a10")%>%.$heatpump)/15000),breaks=100)


#########################################################################
########################### BLDG2 from here ########################

library(tidyverse)
library(patchwork)
library(lubridate)
library(feather)
# figure for 
#df=read_csv("data/overlook/a10/raw_data/unit_data_a10_2018-01-14_2018-01-20.csv")%>%select(timestamp,ahu,heatpump,net,T_out,operation)%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))

df=arrow::read_feather("data/private/all_data.feather")%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))
#df=read_csv("data/private/overlook_all_2018-01-14_2018-09-01.csv")%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))
df=df%>%filter(unitcode%>%str_detect("sb|nb|sa|na")) #filter posterioy data

# df%>%filter(unitcode=="nb06")%>%distinct(timestamp,.keep_all=TRUE)%>%arrange(timestamp)%>%write_csv("nb06.csv")

df=df%>%mutate(hvac=ahu+heatpump)
df=df%>%rename(hc=hvac)%>%mutate(month=month(timestamp))%>%mutate(misc=net-hc)%>%mutate(month=month(timestamp))
df$month%>%unique()
df=df%>%mutate(season=case_when(month%in%c(1,2,3,11,12)~"heating",
                                month%in%c(4,5,6,10)~"transition",
                                month%in%c(7,8,9)~"cooling",
                                TRUE~NA_character_))

df=df%>%mutate(mode=case_when(operation%in%c("heat1","heat1_aux1","aux1")~"heating",
                              operation%in%c("cool1")~"cooling",
                                TRUE~NA_character_))

df=df%>%filter(!(is.na(net)|operation=="ERROR"|is.na(operation)))

df=df%>%group_by(unitcode)%>%distinct(timestamp,.keep_all=TRUE)%>%ungroup()
df=df%>%filter(misc>0&hc>0&hc<45000)

#ggplot(df%>%filter(unitcode=="sb01"),aes(timestamp,T_out))+geom_line()

#View(df%>%filter(unitcode=="nb07"))
# df%>%filter(unitcode=="sb10"&month==5)
df_check=df

df_count=df_check%>%group_by(unitcode,month)%>%summarize(n=n())
df_count=df_count%>%ungroup()%>%pivot_wider(names_from=month,values_from=n)

View(df_count%>%mutate(total=rowSums(df_count%>%select(-unitcode),na.rm=T)))

# 3bed
# na04, na05, na10, na11, nb01, nb06, nb07, nb12
# sa04, sa05, sa10, sa11, sb01, sb06, sb07, sb12

# # na10(3bed) ,  nb08, sb01(3bed),  sa03, sb09



#df_unit=df%>%mutate(month=month(timestamp))%>%filter(unitcode=="a3")%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))
df=df%>%mutate(month=month(timestamp,label=T,abbr=T))%>%filter(!(is.na(net)|operation=="ERROR"|is.na(operation)))
df_density=df%>%filter(unitcode%in%c("nb06","nb08","sb01","sa03","sb05"))%>%select(unitcode,hc,misc,month,operation)%>%pivot_longer(cols=c(hc,misc))
df_density=df_density%>%filter(!(name=="hc"&operation%in%c("idle","ERROR")))
df_density=df_density%>%filter(month%in%c("Jan","Apr","May"))


#df_density%>%filter(name=="hc"&unitcode=="sb09")%>%View()

cols <- c("May"="#619CFF","Jan"="#F8766D","Apr"="#00BA38")

g2=ggplot(df_density,aes(x=value,color=month,group=month,fill=month))+geom_density(alpha=0.1)+
  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  facet_grid(name~unitcode,scales="free")+
  theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.92,.3), #"right"
        legend.direction="vertical",
        legend.text=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0)) +scale_x_continuous(expand=c(0,0),limits=c(0,9000)) 

ggsave("outputs/figures/fig_4_distribution.png",dpi=300,width=12,height=4.5,g2)


df_density%>%filter()




#####################################################################
#############3##################### Heatpump outdoor air temperatrue #######################################3



df_hc=df%>%filter(unitcode%in%c("nb06","nb08","sb01","sa03","sb05"))%>%select(timestamp,unitcode,heatpump,misc,season,operation,T_out)#%>%pivot_longer(cols=c(hc,misc))

df_hc1=df_hc%>%filter(operation%in%c("heat1","cool1"))%>%select(unitcode,heatpump,operation,T_out)

df_hc1=df_hc1%>%mutate(operation=case_when(operation=="heat1"~"heat",operation=="cool1"~"cool",TRUE~NA_character_))

cols <- c("cool"="#619CFF","heat"="#F8766D")
g4=ggplot(df_hc1,aes(x=T_out,y=heatpump,color=operation,group=operation,fill=operation))+geom_point(alpha=0.5)+
  facet_grid(operation~unitcode,scales="free")+  scale_fill_manual(values = cols)+
  scale_color_manual(values = cols)+
  theme(axis.text.x = element_text(size=9,angle=90,hjust=0,vjust=0),  # angle = 0, hjust = 1,vjust=.5,size=10
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=12),#element_blank(),
        legend.position=c(0.9,.4), #"right"
        legend.direction="horizontal",
        legend.text=element_text(size=8),
        legend.title=element_text(size=8),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3),
        panel.spacing = unit(1, "lines")
  )+scale_y_continuous(expand=c(0,0)) +scale_x_continuous(expand=c(0,0)) 
g4
ggsave("outputs/figures/fig_6_distribution.png",dpi=300,width=12,height=4.5,g4)


