library(tidyverse)
library(patchwork)
library(lubridate)
# figure for 
df=read_csv("data/bldg1/raw_data/unit_data_a10_2018-01-14_2018-01-20.csv")%>%select(timestamp,ahu,heatpump,net,T_out,operation)#%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))
df=df%>%mutate(days=yday(timestamp))


#all_data=feather::read_feather("data/private/all_data.feather")
#all_data=all_data%>%mutate(timestamp=with_tz(timestamp,tzone="America/Indianapolis"))

#df=all_data%>%filter(unitcode=="nb05")%>%distinct(timestamp,.keep_all=TRUE)%>%arrange(timestamp)

#all_data$unitcode%>%unique()

# unit_data_a10_2018-01-07_2018-05-07.csv



days_vec=df$days%>%unique()

df$operation[df$operation=="heat_aux"]="aux"

df_day=df%>%filter(days==days_vec[6])
#df_net=df_day%>%select(timestamp,net)%>%pivot_longer(cols=c(net),names_to="name",values_to="value")
df_power=df_day%>%select(timestamp,ahu,heatpump)%>%pivot_longer(cols=c(ahu,heatpump),names_to="name",values_to="value")
df_t_out=df_day%>%select(timestamp,T_out)%>%pivot_longer(cols=c(T_out),names_to="name",values_to="value")
df_operation=df_day%>%select(timestamp,operation)%>%mutate(value=1)


# g1=ggplot(df_net,aes(timestamp,value,group=name,color=name))+geom_line()+
#   theme(axis.text.x = element_blank(), 
#         axis.text.y = element_text(size=10),
#         axis.title.y= element_text(size=12),
#         axis.title.x= element_blank(),#element_blank(),
#         legend.position=c(0.10,.7),
#         legend.direction="vertical",
#         legend.text=element_text(size=10),
#         panel.background = element_rect(fill  = "white", colour = "grey80"),
#         panel.grid.major.y  = element_line(colour = "grey80",size=.3),
#         panel.grid.minor.y= element_line(colour = "grey80",size=.2),
#         panel.grid.major.x= element_line(colour = "grey80",size=.3)
#   )


g2=ggplot(df_power,aes(timestamp,value,group=name,color=name))+geom_line()+
  theme(axis.text.x = element_blank(), 
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_blank(),#element_blank(),
        axis.ticks.x=element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_continuous(expand=c(0,0))

g3=ggplot(df_operation,aes(timestamp,value,group=operation,color=operation,fill=operation))+geom_col()+
  theme(axis.text.x = element_blank(), 
        axis.text.y = element_blank(),
        axis.title.y= element_blank(),
        axis.title.x= element_blank(),#element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_continuous(expand=c(0,0))

g4=ggplot(df_t_out,aes(timestamp,value))+geom_line(color="black")+
  theme(axis.text.x = element_text(angle = 0, size=10), # hjust = 1,vjust=.5,
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=10),#element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_datetime(expand=c(0,0))


g2+g3+g4+plot_layout(heights = c(0.5,0.1,0.2))
ggsave("outputs/figures/timeseries.png",dpi=300,width=12,height=6)




calculate_mode <- function(x) {
  if (length(sum(!is.na(x)))==0){
    out=NA_character_
  }else{
    x=x[!is.na(x)]
    
    if (length(x[x!="idle"])==0){
      out="idle"
    }else{
      x=x[x!="idle"]
      uniqx <- unique(na.omit(x))
      out=uniqx[which.max(tabulate(match(x, uniqx)))]
    }
  }
  return(out)
}


days_vec
df30=df%>%mutate(timestamp=floor_date(timestamp,unit="15min"))
df30_day=df30%>%filter(days==days_vec[6])


df30_day=df30_day%>%select(-c(days))%>%group_by(timestamp)%>%summarize(ahu=mean(ahu,na.rm=T),
                                                                       heatpump=mean(heatpump,na.rm=T),
                                                                       net=mean(net,na.rm=T),
                                                                       T_out=mean(T_out,na.rm=T),
                                                                       operation=calculate_mode(operation))


df_net=df30_day%>%select(timestamp,net)%>%pivot_longer(cols=c(net),names_to="name",values_to="value")
df_power=df30_day%>%select(timestamp,ahu,heatpump)%>%pivot_longer(cols=c(ahu,heatpump),names_to="name",values_to="value")
df_t_out=df30_day%>%select(timestamp,T_out)%>%pivot_longer(cols=c(T_out),names_to="name",values_to="value")
df_operation=df30_day%>%select(timestamp,operation)%>%mutate(value=1)



# g1=ggplot(df_net,aes(timestamp,value,group=name,color=name))+geom_line()+
#   theme(axis.text.x = element_blank(), 
#         axis.text.y = element_text(size=10),
#         axis.title.y= element_text(size=12),
#         axis.title.x= element_blank(),#element_blank(),
#         legend.position=c(0.10,.7),
#         legend.direction="vertical",
#         legend.text=element_text(size=10),
#         panel.background = element_rect(fill  = "white", colour = "grey80"),
#         panel.grid.major.y  = element_line(colour = "grey80",size=.3),
#         panel.grid.minor.y= element_line(colour = "grey80",size=.2),
#         panel.grid.major.x= element_line(colour = "grey80",size=.3)
#   )


g2=ggplot(df_power,aes(timestamp,value,group=name,color=name))+geom_line()+
  theme(axis.text.x = element_blank(), 
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_blank(),#element_blank(),
        axis.ticks.x=element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_continuous(expand=c(0,0))

g3=ggplot(df_operation,aes(timestamp,value,group=operation,color=operation,fill=operation))+geom_col()+
  theme(axis.text.x = element_blank(), 
        axis.text.y = element_blank(),
        axis.title.y= element_blank(),
        axis.title.x= element_blank(),#element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_continuous(expand=c(0,0))

g4=ggplot(df_t_out,aes(timestamp,value))+geom_line(color="black")+
  theme(axis.text.x = element_text(angle = 0, size=10), # hjust = 1,vjust=.5,
        axis.text.y = element_text(size=10),
        axis.title.y= element_text(size=12),
        axis.title.x= element_text(size=10),#element_blank(),
        legend.position="none",
        legend.direction="vertical",
        legend.text=element_text(size=10),
        panel.background = element_rect(fill  = "white", colour = "grey80"),
        panel.grid.major.y  = element_line(colour = "grey80",size=.3),
        panel.grid.minor.y= element_line(colour = "grey80",size=.2),
        panel.grid.major.x= element_line(colour = "grey80",size=.3)
  )+scale_y_continuous(expand=c(0,0))+scale_x_datetime(expand=c(0,0))


g2+g3+g4+plot_layout(heights = c(0.4,0.1,0.2))

ggsave("outputs/figures/timeseries_15min.png",dpi=300,width=12,height=6)
