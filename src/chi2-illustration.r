pdf("../figures/chi2-illustration.pdf",width=9,height=3)
par(mfrow=c(1,3))
for(i in c(4,6,8)) {
  n = round(runif(1,20,100))
  y = rchisq(n,i)/i
  hist(y,main=sprintf("df = %d, n = %.0f\nmean = %.02f, var = %.02f",i,n,mean(y),var(y)))
}

dev.off()