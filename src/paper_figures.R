# Publication figures from saved results. Base R and Cairo; no model refitting.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); setwd(root)
out <- file.path(root,"img/img_used"); dat <- "data/processed/paper_plot_data"
set.seed(42)
read <- function(p) read.csv(p,check.names=FALSE,stringsAsFactors=FALSE)
blue <- "#376E91"; orange <- "#BD7546"; green <- "#45857A"; gray <- "#A9AFB5"
div <- colorRampPalette(c("#365D80","#F7F7F4","#BB754B"))(151)
shortlabel <- function(x,n) {x<-gsub("[()（）]","",x);ifelse(nchar(x)>n,paste0(substr(x,1,n-1),"…"),x)}
seqcol <- colorRampPalette(c("#F5F7F8","#A9C8D9","#325E82"))(151)
openfig <- function(name,w=7.6,h=8.5) {
  png(file.path(out,paste0(name,".png")),width=w,height=h,units="in",res=450,
      type="cairo",family="Noto Serif CJK SC",pointsize=11,bg="white")
  par(col.axis="#20272D",col.lab="#20272D",fg="#20272D",las=1,tcl=-.2,lwd=.65,
      mgp=c(1.8,.45,0),cex.axis=.8,cex.lab=.85,xaxs="i",yaxs="i")
}
closefig <- function() {dev.off();cat("[R figure] completed\n")}
mapcolors <- function(x,zlim,pal) {
  id <- 1+round((pmin(zlim[2],pmax(zlim[1],x))-zlim[1])/diff(zlim)*(length(pal)-1))
  ans <- pal[id];ans[!is.finite(x)] <- "#DADDDD";ans
}
heat <- function(M,rl=NULL,cl=NULL,zlim=range(M,finite=TRUE),pal=div,nums=FALSE,cex=.56,separators=NULL) {
  nr<-nrow(M);nc<-ncol(M)
  plot.new();plot.window(xlim=c(.5,nc+.5),ylim=c(nr+.5,.5))
  for(j in seq_len(nc))rect(j-.5,seq_len(nr)-.5,j+.5,seq_len(nr)+.5,
      col=mapcolors(M[,j],zlim,pal),border="white",lwd=.3)
  if(!is.null(rl))axis(2,at=seq_len(nr),labels=rl,las=2,tick=FALSE,cex.axis=.8)
  if(!is.null(cl))axis(1,at=seq_len(nc),labels=cl,las=2,tick=FALSE,cex.axis=.8)
  if(nums)for(i in seq_len(nr))for(j in seq_len(nc)) {
    v<-M[i,j];c<-mapcolors(v,zlim,pal);bright<-mean(col2rgb(c))
    text(j,i,ifelse(is.finite(v),formatC(ifelse(abs(v)<.005,0,v),format="f",digits=2),"NA"),cex=cex,
        col=ifelse(bright<145,"white","#192733"))
  }
  if(length(separators))abline(h=separators+.5,col="#525B62",lwd=.75)
}
key <- function(zlim,pal,label) {
  par(mar=c(1.5,.2,.2,.2));plot.new();plot.window(xlim=c(0,1),ylim=c(0,1))
  xx<-seq(.10,.90,length.out=length(pal)+1)
  rect(xx[-length(xx)],.52,xx[-1],.86,col=pal,border=NA)
  text(c(.10,.50,.90),.30,formatC(c(zlim[1],mean(zlim),zlim[2]),digits=2,format="fg"),cex=.75)
  text(.50,-.05,label,cex=.8,xpd=NA)
}
T<-read(file.path(dat,"targets.csv"));targets<-T$target;labs<-T$label
sep<-c(15,30)
matlong <- function(d,row,col,value,rows,cols) {
 M<-matrix(NA_real_,length(rows),length(cols),dimnames=list(rows,cols))
 for(i in seq_len(nrow(d)))if(d[[row]][i]%in%rows && d[[col]][i]%in%cols)
   M[as.character(d[[row]][i]),as.character(d[[col]][i])]<-d[[value]][i]
 M
}
# 02: target-level errors and rank agreement, not only an aggregate mean.
q<-read(file.path(dat,"q1_metrics.csv"))
mods<-c("style_mean","ridge","pls","extratrees","randomforest","lightgbm","blend_avg","selected_pipeline")
mn<-c("风格基线","Ridge","PLS2","ExtraTrees","RandomForest","LightGBM","五模型平均","训练内选型")
M<-matlong(q,"target","model","r2",targets,mods)
best<-q[q$model=="selected_pipeline",];best<-best[match(targets,best$target),]
openfig("02_Q1模型对照",7.7,8.7)
layout(matrix(c(1,2,3,4,4,4),2,3,byrow=TRUE),widths=c(5.4,1.15,1.15),heights=c(8,.55))
par(mar=c(4.7,8.4,.5,.5));heat(M,labs,mn,c(-.8,.8),div,separators=sep)
for(i in seq_along(targets))points(which.max(replace(M[i,],!is.finite(M[i,]),-Inf)),i,pch=3,cex=.25,lwd=.55)
for(k in c("mae","spearman")) {
 par(mar=c(4.7,.3,.5,.3));vals<-best[[k]]
 plot.new();plot.window(xlim=if(k=="mae")c(0,max(vals)*1.07) else c(-.1,.9),ylim=c(50.5,.5))
 abline(v=pretty(par("usr")[1:2]),col="#E3E5E6",lwd=.5)
 segments(0,1:50,vals,1:50,col=if(k=="mae")blue else green,lwd=2)
 points(vals,1:50,pch=16,cex=.3,col=if(k=="mae")blue else green)
 axis(1,cex.axis=.65);mtext(if(k=="mae")"主流程 MAE" else "主流程 Spearman",1,line=2.1,cex=.7)
 abline(h=sep+.5,col="#9DA2A6",lwd=.5)
}
key(c(-.8,.8),div,"逐目标 R² · 十字标记该目标最高折均值")
closefig()
# 04: trajectories of every fold and a paired degradation matrix.
m<-read("data/q1/missingness_folds.csv")
sc<-c("natural","MCAR 5%","MCAR 10%","MCAR 20%","MCAR 30%","MAR")
fold<-paste(m[["repeat"]]+1,m$fold+1,sep="-");m$id<-fold;ids<-unique(fold)
loss<-matrix(NA,15,5,dimnames=list(ids,sc[-1]))
for(i in seq_along(ids)) {
 a<-m[m$id==ids[i],];base<-a$r2[a$scenario=="natural"]
 loss[i,]<-100*(base-a$r2[match(sc[-1],a$scenario)])/abs(base)
}
openfig("04_Q1缺失压力",7.6,6.8)
layout(matrix(c(1,2,3,3,4,4),3,2,byrow=TRUE),heights=c(2.7,3.2,.5))
for(metric in c("r2","rmse")) {
 par(mar=c(3.6,3.7,.5,.7));ran<-range(m[[metric]])
 plot(0,0,type="n",xlim=c(0,32),ylim=ran,xlab="总缺失率 / %",ylab=if(metric=="r2")"宏平均 R²" else "宏 RMSE")
 abline(v=c(0,10,20,30),col="#ECEDEE",lwd=.5)
 for(id in ids) {
  a<-m[m$id==id,];b<-a[match(sc[1:5],a$scenario),]
  lines(b$total_missing_rate*100,b[[metric]],col=adjustcolor(gray,.6),lwd=.6)
  a<-a[a$scenario=="MAR",];points(a$total_missing_rate*100,a[[metric]],pch=2,cex=.5,col=adjustcolor(orange,.55))
 }
 meanx<-sapply(sc[1:5],function(s)mean(m$total_missing_rate[m$scenario==s])*100)
 meany<-sapply(sc[1:5],function(s)mean(m[[metric]][m$scenario==s]))
 lines(meanx,meany,col=blue,lwd=1.8,type="b",pch=16,cex=.55)
 points(mean(m$total_missing_rate[m$scenario=="MAR"])*100,mean(m[[metric]][m$scenario=="MAR"]),pch=17,col=orange,cex=1)
 legend("topright",legend=c("MCAR 折均值","MAR 折均值"),col=c(blue,orange),pch=c(16,17),bty="n",cex=.64)
}
par(mar=c(4.3,4.6,.5,.5));heat(loss,paste("重复-折",ids),c("MCAR 5%","MCAR 10%","MCAR 20%","MCAR 30%","MAR"),c(-35,35),div,TRUE,.75,c(5,10))
key(c(-35,35),div,"相对同折自然缺失的 R² 降幅 / %")
closefig()
# 05: the topology uses Gephi ForceAtlas2, with community attributes and hubs.
n<-read("data/q2/network_nodes.csv");e<-read("data/q2/network_edges.csv")
p<-read(file.path(dat,"gephi_layout.csv"));p<-p[match(n$feature,p$feature),];g<-sort(unique(n$community))
pal<-c("#497CA0","#BB8952","#60978A","#B66D7D","#8A81A9","#7F975B","#A17C60","#6E9FAC")
openfig("05_Q2关联网络",7.6,7.2)
layout(matrix(c(1,1,2,3),2,2,byrow=TRUE),heights=c(4.8,2.4))
pad <- function(x) range(x)+c(-1,1)*diff(range(x))*.09
par(mar=c(.8,.6,.4,.6));plot(p$x,p$y,type="n",axes=FALSE,xlab="",ylab="",asp=1,xlim=pad(p$x),ylim=pad(p$y))
a<-match(e$source,n$feature);b<-match(e$target,n$feature)
segments(p$x[a],p$y[a],p$x[b],p$y[b],col=adjustcolor("#71808A",.10),lwd=.25+.55*e$abs_r)
points(p$x,p$y,pch=21,bg=pal[match(n$community,g)],col="white",lwd=.3,cex=.28+.085*sqrt(n$degree))
keynames<-c("DMS","X2.3.butanedione","X4VG","ethanol..v.v.")
keylabels<-c("DMS","双乙酰","4VG","乙醇")
for(i in seq_along(keynames)) {
 j<-match(keynames[i],n$feature);if(!is.na(j))text(p$x[j],p$y[j],keylabels[i],pos=ifelse(p$x[j]>0,4,2),cex=.65,offset=.6)
}
legend("bottomleft",legend=paste(g,as.integer(table(factor(n$community,g))),sep=" · "),col=pal,pch=16,bty="n",ncol=4,cex=.62,xpd=NA)
par(mar=c(3.5,5,.7,.4));ct<-table(n$chemical_class,n$community)
sel<-order(rowSums(ct),decreasing=TRUE)[1:5];ct2<-rbind(ct[sel,],其他=colSums(ct[-sel,,drop=FALSE]))
cl<-c("#6C8CA2","#B49A71","#769987","#AE838C","#928AA7","#C4C5C3")
barplot(ct2[,g],col=cl,border=NA,las=1,cex.names=.65,ylab="社区节点数")
legend("topright",legend=c("酯类","萜类","醇类","醛类","萜烃类","其他"),fill=cl,bty="n",cex=.63,ncol=2)
par(mar=c(3.5,7.4,.7,.5));h<-n[order(n$hub_rank),][1:8,]
barplot(rev(h$degree),names.arg=rev(shortlabel(h$compound_2csv,21)),horiz=TRUE,col="#7796AA",border=NA,las=1,cex.names=.68,xlab="综合枢纽前八项的度")
closefig()
# 06: all target candidates, method agreement, and predictive utility.
top<-read("data/q2/top3.csv");J<-read("data/q2/jaccard.csv");cv3<-read("data/q2/top3_prediction_cv.csv")
freq<-matlong(top,"target","rank","stability_freq",targets,c("1","2","3"))
jmean<-tapply(J$jaccard,J$target,mean);rmean<-tapply(cv3$r2[cv3$method=="compound_only"],cv3$target[cv3$method=="compound_only"],mean)
openfig("06_Q2候选稳定性",7.3,8.7)
layout(matrix(c(1,2,3,4,4,4),2,3,byrow=TRUE),widths=c(4.1,1.3,1.9),heights=c(8,.5))
par(mar=c(4.3,8.4,.5,.5));heat(freq,labs,c("候选 1","候选 2","候选 3"),c(0,1),seqcol,TRUE,.75,sep)
par(mar=c(4.3,.3,.5,.4));v<-jmean[targets]
plot.new();plot.window(xlim=c(0,1),ylim=c(50.5,.5));abline(v=c(0,.5,1),col="#ECEDEE");segments(0,1:50,v,1:50,col=blue,lwd=2);axis(1);mtext("方法 Jaccard",1,line=2.2,cex=.68)
par(mar=c(4.3,.3,.5,.7));v<-rmean[targets];lim<-range(c(-.3,.5,v),finite=TRUE)
plot.new();plot.window(xlim=lim,ylim=c(50.5,.5));abline(v=0,col="#697780",lwd=.7);segments(0,1:50,v,1:50,col=ifelse(v>=0,green,orange),lwd=2);axis(1);mtext("化学 Top-3 的 CV R²",1,line=2.2,cex=.67)
key(c(0,1),seqcol,"500 次重采样入选频率 · 三列按平均秩顺序")
closefig()
# 07: style-specific means and SDs, each cell represents five holdouts.
z<-read("data/q3/metrics.csv");models<-c("empirical","mean_noise","global_cov","copula","covmatch","ot","ot_rank","ot_constrained")
modelnames<-c("经验重采样","独立噪声","共享协方差","条件 copula","协方差匹配","高斯 OT","OT 加目标边际","OT 加边际与上限")
styles<-c("Tripel","Kriek","Lambic");metric<-c("mmd2","coverage","precision","rf_hit","sensory_rmse")
metricnames<-c("MMD² ↓","覆盖度 ↑","精确度 ↑","分类命中率 ↑","画像 RMSE ↓")
openfig("07_Q3生成质量",8.0,6.9)
layout(rbind(1:5,6:10),widths=c(3.1,1.2,1.2,1.2,1.3),heights=c(6.2,.6))
lims<-pals<-list()
for(k in 1:5) {
 means<-sds<-numeric();rl<-character()
 for(s in styles)for(i in seq_along(models)) {
  v<-z[z$style==s & z$model==models[i],metric[k]];means<-c(means,mean(v));sds<-c(sds,sd(v));rl<-c(rl,paste(s,modelnames[i],sep=" · "))
 }
 lim<-range(means);if(diff(lim)<.01)lim<-lim+c(-.005,.005);lims[[k]]<-lim;pals[[k]]<-if(k%in%c(1,5))rev(seqcol) else seqcol
 par(mar=c(.6,if(k==1)13.4 else .3,.5,.3));heat(matrix(means,ncol=1),if(k==1)rl else NULL,NULL,lim,pals[[k]],separators=c(8,16))
 for(i in 1:24)text(1,i,paste0(formatC(means[i],digits=if(k==1)3 else 2,format="f")," ± ",formatC(sds[i],digits=if(k==1)3 else 2,format="f")),cex=.7,col=ifelse(mean(col2rgb(mapcolors(means[i],lim,pals[[k]])))<145,"white","#182733"))
}
for(k in 1:5)key(lims[[k]],pals[[k]],metricnames[k])
closefig()
# 08: all 70 calibrated indicators, paired real/generated medians in a shared geometry.
z<-read(file.path(dat,"chemical_medians.csv"));features<-unique(z$feature)
cols<-as.vector(rbind(paste(styles,"real",sep="_"),paste(styles,"generated",sep="_")))
z$key<-paste(z$style,z$kind,sep="_");M<-matlong(z,"feature","key","value",features,cols)
ord<-hclust(dist(M[,c(1,3,5)]),method="ward.D2")$order;M<-M[ord,,drop=FALSE]
label<-setNames(z$compound,z$feature);rl<-shortlabel(label[rownames(M)],26);lim<-ceiling(max(abs(M))*2)/2
openfig("08_Q3代表化学谱",8,7.7)
layout(matrix(c(1,2,3,3),2,2,byrow=TRUE),heights=c(7,.6))
for(ix in list(1:35,36:70)) {
 par(mar=c(4.3,10,.3,.4));heat(M[ix,,drop=FALSE],rl[ix],c("T 实测","T 生成","K 实测","K 生成","L 实测","L 生成"),c(-lim,lim),div)
 abline(v=c(2.5,4.5),col="#3F4C57",lwd=.65)
}
key(c(-lim,lim),div,"全部 70 个校准指标的标准化中位数 · T Tripel · K Kriek · L Lambic")
closefig()
# 09: all 50 sensory dimensions, paired portraits and proxy validation.
z<-read(file.path(dat,"sensory_profiles.csv"));z$key<-paste(z$style,z$kind,sep="_")
M<-matlong(z,"target","key","median",targets,cols);delta<-M[,c(2,4,6)]-M[,c(1,3,5)]
proxy<-read("data/q3/sensory_cv.csv");rr<-tapply(proxy$r2,proxy$target,mean)[targets]
openfig("09_Q3感官画像",8,8.8)
layout(matrix(c(1,2,3,4,5,6),2,3,byrow=TRUE),widths=c(4.6,2.1,1.3),heights=c(8,.6))
par(mar=c(4.3,8.4,.4,.4));lim<-ceiling(max(abs(M))*2)/2
heat(M,labs,c("T 实测","T 生成","K 实测","K 生成","L 实测","L 生成"),c(-lim,lim),div,separators=sep);abline(v=c(2.5,4.5),col="#5D666E",lwd=.6)
par(mar=c(4.3,.3,.4,.4));dlim<-max(.3,ceiling(max(abs(delta))*10)/10)
heat(delta,NULL,styles,c(-dlim,dlim),div,TRUE,.72,sep)
par(mar=c(4.3,.3,.4,.4));plot.new();plot.window(xlim=range(c(-.4,.8,rr)),ylim=c(50.5,.5));abline(v=0,col="#687680");segments(0,1:50,rr,1:50,col=ifelse(rr>=0,green,orange),lwd=2);axis(1);mtext("代理 CV R²",1,line=2.1,cex=.68)
key(c(-lim,lim),div,"标准化感官响应")
key(c(-dlim,dlim),div,"生成减实测")
par(mar=rep(0,4));plot.new();text(.5,.5,"T Tripel\nK Kriek\nL Lambic",cex=.62)
closefig()
# 10: individual source beers across constraints, not just zero violation counts.
tr<-read("data/q3/transfer_profiles.csv")
variants<-c("source","raw","excess","strict","style");vn<-c("源酒","原始OT","主约束","严格约束","风格上限")
cc<-c("DMS","X2.3.butanedione","X4VG","X.E..2.nonenal");cn<-c("DMS","双乙酰","4VG","T2N")
openfig("10_Q3迁移缺陷",7.6,6.1);par(mfrow=c(2,2),mar=c(3.2,4,.7,.5))
for(j in 1:4) {
 scenarios<-if(j==4)c("attachment_mg_L","alternative_ug_L") else "attachment_mg_L"
 upper<-0
 for(s in scenarios) {
  d<-tr[tr$unit_scenario==s & tr$variant%in%variants,];tau<-c(17.25,.09,.1,if(s=="attachment_mg_L").0001 else .1)[j]
  upper<-max(upper,log10(1+d[[cc[j]]]/tau))
 }
 plot(0,0,type="n",xlim=c(.7,5.3),ylim=c(0,upper*1.07),axes=FALSE,xlab="",ylab=paste(cn[j],"· OAV"))
 ticks<-c(0,.1,1,10,100,1000);ticks<-ticks[log10(1+ticks)<=upper*1.07];axis(2,at=log10(1+ticks),labels=ticks,las=1);axis(1,at=1:5,labels=vn,cex.axis=.65)
 abline(h=log10(2),lty=2,col="#2C3339",lwd=.75)
 for(k in seq_along(scenarios)) {
  s<-scenarios[k];d<-tr[tr$unit_scenario==s & tr$variant%in%variants,];tau<-c(17.25,.09,.1,if(s=="attachment_mg_L").0001 else .1)[j]
  M<-matlong(d,"sample_id","variant",cc[j],sort(unique(d$sample_id)),variants)/tau
  for(i in 1:nrow(M))lines(1:5+(k-1)*.04,log10(1+M[i,]),col=adjustcolor(if(k==1)blue else orange,.22),lwd=.6)
  lines(1:5+(k-1)*.04,log10(1+apply(M,2,median)),col=if(k==1)blue else orange,lwd=1.6,type="b",pch=if(k==1)16 else 17,cex=.65)
 }
 if(j==4)legend("topright",legend=c("按附件 mg/L","按替代 μg/L"),col=c(blue,orange),lty=1,pch=c(16,17),bty="n",cex=.57)
 box(bty="l")
}
closefig()
# 11: all sensory deviations plus individual-beer proxy errors in both scenarios.
z<-read(file.path(dat,"transfer_deltas.csv"));err<-read(file.path(dat,"transfer_beer_errors.csv"))
ks<-paste("attachment_mg_L",variants,sep="_");M<-matlong(z,"target","model","delta",targets,ks)
openfig("11_Q3迁移画像误差",7.8,8.5)
layout(matrix(c(1,2,1,3,4,5),3,2,byrow=TRUE),widths=c(4.5,3.3),heights=c(3.8,3.8,.6))
par(mar=c(4.3,8.4,.5,.4));lim<-max(abs(M));heat(M,labs,vn,c(-lim,lim),div,separators=sep)
ids<-sort(unique(err$beer))
for(s in c("attachment_mg_L","alternative_ug_L")) {
 par(mar=c(4.3,4,1.5,.4));E<-matlong(err,"beer","model","rmse",ids,paste(s,variants,sep="_"))
 heat(E,paste("Beer",ids),vn,c(0,.8),seqcol,TRUE,.75)
 mtext(if(s=="attachment_mg_L")"T2N 按附件 mg/L" else "T2N 按替代 μg/L",3,line=.05,adj=0,cex=.65)
}
key(c(-lim,lim),div,"相对 Tripel 实测画像的响应差 · 附件 mg/L 情景")
key(c(0,.8),seqcol,"逐酒代理画像 RMSE")
closefig()
cat("R version: ",R.version.string,"; graphics: Cairo\n")
