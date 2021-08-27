library(tidyverse)

softplus=function(x){
  return(log(1+exp(x)))
}

aa=softplus(rnorm(50000,-2,1))
quantile(aa,probs=c(0.025,0.5,0.975))
hist(aa,breaks=200,xlim=c(0,1))

# 2.5%         50%       97.5% 
# 0.001483015 0.048394356 1.006163756 

library(tidyverse)

softplus=function(x){
  return(log(1+exp(x)))
}


aa=softplus(rnorm(50000,-1,0.75))
quantile(aa,probs=c(0.025,0.5,0.975))
hist(aa,breaks=100)
# 2.5%        50%      97.5% 
# 0.08064485 0.31375553 0.95917994 

# E_misc
aa=softplus(rnorm(50000,-4,2.3))
quantile(aa,probs=c(0.025,0.5,0.975))
hist(aa,breaks=100)
# 2.5%          50%        97.5% 
# 0.0002109215 0.0182869140 0.9945600335 

# sigma_net
aa=softplus(rnorm(50000,-4,1.5))
quantile(aa,probs=c(0.025,0.5,0.975))
hist(aa,breaks=100)
# 2.5%          50%        97.5% 
# 0.0009852637 0.0181358328 0.2959371127 




aa=softplus(rnorm(50000,-.5,0.5))
quantile(aa,probs=c(0.025,0.5,0.975))
hist(aa,breaks=100)
# 2.5%       50%     97.5% 
# 0.2039341 0.4722836 0.9513658 


hist(exp((1-(rbeta(10000,2,8)))*3),breaks=100)
hist(exp(rbeta(10000,40,10)),breaks=100)


#     def calculate_concentration(self,mu,sigma):
#         concentration_alpha=((1-mu)/(sigma**2)-1/mu)*(mu**2)
#         concentration_beta=concentration_alpha*(1/mu-1)
#         return concentration_alpha, concentration_beta

# concentration1=10
# concentration2=5
# ab=rbeta(10000,concentration1,concentration2)
# mean(ab)
# sd(ab)
# calculate_concentration=function(mu,sigma){
#   concentration_alpha=((1-mu)/(sigma^2)-1/mu)*(mu^2)
#   concentration_beta=concentration_alpha*(1/mu-1)
#   print(concentration_alpha)
#   print(concentration_beta)
# }
# 
# calculate_concentration(mean(ab),sd(ab))
# calculate_concentration(1,0.5)
# calculate_concentration(4,0.5)




sigmoid=function(x,k=1){
  return(1/(1+exp(-k*x)))
}
# exponential right skewed to left skewed
hist(exp(rnorm(10000,-3,1.0)),breaks=1000,xlim=c(0,0.5))

par(mfrow=c(1,2))
#hist(softplus((rnorm(10000,1,0.9)),k=1)*0.6,breaks=1000,xlim=c(0,1))
hist(((rnorm(10000,1,1.))),breaks=100,xlim=c(-2,4))
hist(sigmoid((rnorm(10000,1,1)),k=1),breaks=100,xlim=c(0,1))
hist((softplus(rnorm(10000,-2,1))),breaks=100,xlim=c(0,1))

df=tibble(sigmoid=sigmoid(rnorm(10000,1,1)),
       softplus=softplus(rnorm(10000,-2,1)))

ggplot(df%>%pivot_longer(cols=c(sigmoid,softplus)),aes(value))+geom_density()+facet_grid(rows=vars(name),scales="free")+theme_bw()+
  scale_y_continuous(expand=c(0,0)) +scale_x_continuous(expand=c(0,0),limits=c(0,1)) 


abc=rnorm(10000,0.5,1.0)
hist(sigmoid(abc))
quantile(sigmoid(abc),probs=c(0.025,0.5,0.975))
sigmoid(-2:2)



library(tidyverse)



dd=tibble(x=c('1','1','2','2'),ymin_=c(1,2,3,4),ymax_=c(6,7,8,9),lower_=c(2,3,4,5),upper_=c(4,5,6,7),middle_=c(3,4,5,6),mark=c("c","d","c","d"))
ggplot(dd,aes(x=x,group=interaction(x,mark)))+geom_boxplot(aes(fill=mark,ymin=ymin_,ymax=ymax_,lower=lower_,middle=middle_,upper=upper_),width=0.5, stat = "identity")




min_max_recover=function(x,x_min=-20,x_max=40,lower=-1,upper=1.){
  return ((x-lower)*(x_max-x_min)/(upper-lower)+x_min)
}
aa=rnorm(n=5000,mean=-1/3,sd=0.2)
abc=min_max_recover(x=aa)

quantile(abc,probs=c(0.025,0.5,0.975))

