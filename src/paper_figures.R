# Publication figures from saved results. Base R and Cairo; no model refitting.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); setwd(root)
selected <- if(length(args)>1) strsplit(args[2],",",fixed=TRUE)[[1]] else sprintf("%02d",3:16)
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
variants<-c("source","raw","excess","strict","style");vn<-c("源酒","原始OT","主约束","严格约束","风格上限")
styles<-c("Tripel","Kriek","Lambic")
cols<-as.vector(rbind(paste(styles,"real",sep="_"),paste(styles,"generated",sep="_")))
T<-read(file.path(dat,"targets.csv"));targets<-T$target;labs<-T$label
sep<-c(15,30)
matlong <- function(d,row,col,value,rows,cols) {
 M<-matrix(NA_real_,length(rows),length(cols),dimnames=list(rows,cols))
 for(i in seq_len(nrow(d)))if(d[[row]][i]%in%rows && d[[col]][i]%in%cols)
   M[as.character(d[[row]][i]),as.character(d[[col]][i])]<-d[[value]][i]
 M
}
# 02: target-level errors and rank agreement, not only an aggregate mean.
if("02" %in% selected) {
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
}

# 03: the saved scoring cards, with a short grade legend only.
if("03" %in% selected) {
 cards<-read("data/q1/card_scores.csv");mapping<-read("data/q1/card_mapping.csv")
 gradecol<-c("#F3F7FA","#DCEAF3","#B6D3E4","#86B6CC","#5897B8","#337FA3","#195274")
 openfig("03_Q1评分示例",7.6,7.4);par(family="Noto Sans CJK SC")
 layout(matrix(c(1,2,3,3),2,2,byrow=TRUE),heights=c(6.8,.6))
 for(side in c("Aroma","Flavour")) {
  mp<-mapping[mapping$side==side,];d<-cards[cards$side==side & cards$beer%in%1:3,]
  M<-matlong(d,"item","beer","relative_grade",mp$item,as.character(1:3))
  labels<-labs[match(mp$target,targets)];labels[grepl("_all$",mp$item)]<-paste0(labels[grepl("_all$",mp$item)],"-总体")
  labels[is.na(labels)]<-"乙醛"
  par(mar=c(2.5,9.7,.4,.4));heat(M,labels,NULL,c(0,6),gradecol)
  axis(1,at=1:3,labels=paste("Beer",1:3),tick=FALSE,las=1,cex.axis=.85)
  mtext(paste(side,"卡片栏位"),2,line=8.6,cex=.9,las=0)
  for(i in seq_len(nrow(M)))for(j in 1:3)text(j,i,ifelse(is.finite(M[i,j]),as.character(as.integer(M[i,j])),"NA"),cex=.9,col=ifelse(is.finite(M[i,j]) && M[i,j]>=4,"white","#223442"))
 }
 par(mar=c(2,.2,.2,.2));plot.new();plot.window(xlim=c(-1.4,8.4),ylim=c(0,1))
 rect(0:6,.5,1:7,.9,col=gradecol,border="white",lwd=.4)
 axis(1,at=(0:6)+.5,labels=0:6,pos=.5,cex.axis=.8);mtext("相对等级",1,line=.6,cex=.9)
 closefig()
}
# 04: trajectories of every fold and a paired degradation matrix.
if("04" %in% selected) {
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
}

# 05: the topology uses Gephi ForceAtlas2, with community attributes and hubs.
if("05" %in% selected) {
n<-read("data/q2/network_nodes.csv");e<-read("data/q2/network_edges.csv")
p<-read(file.path(dat,"gephi_layout.csv"));p<-p[match(n$feature,p$feature),];g<-sort(unique(n$community))
pal<-c("#497CA0","#BB8952","#60978A","#B66D7D","#8A81A9","#7F975B","#A17C60","#6E9FAC")
openfig("05_Q2关联网络",8,8.2)
layout(matrix(c(1,1,2,3),2,2,byrow=TRUE),heights=c(6.5,1.7))
# A rigid rotation preserves layout distances while matching the available panel.
xy<-scale(cbind(p$x,p$y),center=TRUE,scale=FALSE)
angles<-seq(0,pi,length.out=181);aspect<-7.8/6.05
cost<-sapply(angles,function(a){z<-xy%*%matrix(c(cos(a),sin(a),-sin(a),cos(a)),2);r<-apply(z,2,function(x)diff(range(x)));abs(log((r[1]/r[2])/aspect))})
angle<-angles[which.min(cost)];xy<-xy%*%matrix(c(cos(angle),sin(angle),-sin(angle),cos(angle)),2);p$x<-xy[,1];p$y<-xy[,2]
pad <- function(x) range(x)+c(-1,1)*diff(range(x))*.035
par(mar=c(.9,.15,.15,.15));plot(p$x,p$y,type="n",axes=FALSE,xlab="",ylab="",asp=1,xlim=pad(p$x),ylim=pad(p$y))
a<-match(e$source,n$feature);b<-match(e$target,n$feature)
segments(p$x[a],p$y[a],p$x[b],p$y[b],col=adjustcolor("#71808A",.16),lwd=.25+.55*e$abs_r)
points(p$x,p$y,pch=21,bg=pal[match(n$community,g)],col="white",lwd=.3,cex=.42+.10*sqrt(n$degree))
keynames<-c("DMS","X2.3.butanedione","X4VG","ethanol..v.v.")
keylabels<-c("DMS","双乙酰","4VG","乙醇")
for(i in seq_along(keynames)) {
 j<-match(keynames[i],n$feature);if(!is.na(j))text(p$x[j],p$y[j],keylabels[i],pos=ifelse(p$x[j]>0,4,2),cex=.65,offset=.6)
}
legend("bottomleft",legend=paste(g,as.integer(table(factor(n$community,g))),sep=" · "),col=pal,pch=16,bty="n",ncol=4,cex=.62,xpd=NA)
par(mar=c(3.0,4.7,.25,.4));ct<-table(n$chemical_class,n$community)
sel<-order(rowSums(ct),decreasing=TRUE)[1:5];ct2<-rbind(ct[sel,],其他=colSums(ct[-sel,,drop=FALSE]))
cl<-c("#6C8CA2","#B49A71","#769987","#AE838C","#928AA7","#C4C5C3")
barplot(ct2[,g],col=cl,border=NA,las=1,cex.names=.65,ylab="社区节点数")
legend("topright",legend=c("酯类","萜类","醇类","醛类","萜烃类","其他"),fill=cl,bty="n",cex=.63,ncol=2)
par(mar=c(3.0,7.4,.25,.5));h<-n[order(n$hub_rank),][1:8,]
barplot(rev(h$degree),names.arg=rev(shortlabel(h$compound_2csv,21)),horiz=TRUE,col="#7796AA",border=NA,las=1,cex.names=.68,xlab="综合枢纽前八项的度")
closefig()
}

# 06: all target candidates, method agreement, and predictive utility.
if("06" %in% selected) {
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
key(c(0,1),seqcol,"500 次重采样入选频率")
closefig()
}

# 07: style-specific means and SDs, each cell represents five holdouts.
if("07" %in% selected) {
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
}

# 08: all 70 calibrated indicators, paired real/generated medians in a shared geometry.
if("08" %in% selected) {
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
}

# 09: the actual generated examples, shown as a compact core sensory portrait.
if("09" %in% selected) {
 examples<-read(file.path(dat,"generated_example_sensory.csv"));examples<-examples[match(styles,examples$style),]
 core<-c("A_malt_all","A_hops_all","A_esters_all","F_malt_all","F_hops_all","F_esters_all",
         "body","alcohol","co2","bitternes","sweetness","acidity","lactic","acetic")
 labels<-labs[match(core,targets)];values<-t(as.matrix(examples[,core]))
 order<-order(apply(values,1,function(x)diff(range(x))),decreasing=TRUE)
 labels<-labels[order];values<-values[order,,drop=FALSE]
 lim<-range(c(0,values));lim<-lim+c(-1,1)*diff(lim)*.22
 palette<-c(blue,orange,green)
 openfig("09_Q3感官画像",8.1,5.9);par(oma=c(1.6,0,.5,0))
 layout(matrix(1:3,1,3),widths=c(3.6,2.25,2.25))
 for(k in 1:3) {
  par(mar=c(2.3,if(k==1)7.6 else .4,2.2,.45))
  plot.new();plot.window(xlim=lim,ylim=c(length(core)+.7,.3))
  abline(v=0,col="#6D777F",lwd=.7)
  if(k==1)axis(2,at=seq_along(core),labels=labels,las=2,tick=FALSE,cex.axis=.85)
  for(i in seq_along(core)) {
   v<-values[i,k];rect(min(0,v),i-.27,max(0,v),i+.27,col=palette[k],border=NA)
   value<-if(abs(v)<.005)0 else v
   text(v+if(v>=0).035*diff(lim) else -.035*diff(lim),i,formatC(value,digits=2,format="f"),
        adj=if(v>=0)0 else 1,cex=.83,col="#26343F")
  }
  axis(1,cex.axis=.75)
  mtext(styles[k],3,line=1.05,cex=1)
  mtext(paste0("#",sub(".*_","",examples$sample_id[k])),3,line=.05,cex=.72)
 }
 mtext("标准化感官响应",1,outer=TRUE,line=.4,cex=.9)
 closefig()
}
# 09b: direct sensory portraits in three aligned interval panels.
if("09b" %in% selected) {
 z<-read(file.path(dat,"sensory_profiles.csv"))
 limits<-range(z[,c("q25","q75")],finite=TRUE);limits<-limits+c(-1,1)*diff(limits)*.04
 openfig("09b_Q3感官分布对照",8.2,8.3);par(oma=c(1.6,0,.6,0))
 layout(matrix(c(1,2,3,4,4,4),2,3,byrow=TRUE),widths=c(3.7,2.25,2.25),heights=c(7.9,.4))
 for(k in seq_along(styles)) {
  real<-z[z$style==styles[k] & z$kind=="real",];real<-real[match(targets,real$target),]
  generated<-z[z$style==styles[k] & z$kind=="generated",];generated<-generated[match(targets,generated$target),]
  par(mar=c(2.6,if(k==1)8.6 else .45,1.4,.3))
  plot.new();plot.window(xlim=limits,ylim=c(50.7,.3))
  abline(v=pretty(limits),col="#E9ECED",lwd=.5);abline(v=0,col="#A6ADB2",lwd=.65)
  abline(h=sep+.5,col="#727D85",lwd=.65)
  if(k==1)axis(2,at=1:50,labels=labs,las=2,tick=FALSE,cex.axis=.8)
  y<-1:50
  segments(real$median,y+.13,generated$median,y-.13,col="#BAC2C7",lwd=.6)
  segments(real$q25,y+.13,real$q75,y+.13,col="#B2BBC2",lwd=.9)
  segments(generated$q25,y-.13,generated$q75,y-.13,col=adjustcolor(blue,.65),lwd=1.2)
  points(real$median,y+.13,pch=21,bg="white",col="#89969F",cex=.6,lwd=.7)
  points(generated$median,y-.13,pch=16,col=blue,cex=.55)
  axis(1,cex.axis=.78);mtext(styles[k],3,line=.3,cex=.95)
 }
 par(mar=c(.1,0,.1,0));plot.new()
 legend("top",legend=c("实测","生成"),col=c("#89969F",blue),pt.bg=c("white",blue),pch=c(21,16),lty=1,lwd=c(.9,1.2),bty="n",horiz=TRUE,cex=.85)
 mtext("标准化感官响应",1,outer=TRUE,line=.5,cex=.85)
 closefig()
}
# 10: individual source beers across constraints, not just zero violation counts.
if("10" %in% selected) {
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
}

# 11: all sensory deviations plus individual-beer proxy errors in both scenarios.
if("11" %in% selected) {
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
}

# 12-14: the complete 50 by 3 compound identities, split by sensory domain.
if(any(c("12","13","14") %in% selected)) {
 top<-read("data/q2/top3.csv")
 groups<-list("12"=targets[grepl("^A_",targets)],"13"=targets[grepl("^F_",targets)],
              "14"=targets[!grepl("^[AF]_",targets)])
 filenames<-c("12"="12_Q2关键化合物_香气","13"="13_Q2关键化合物_风味","14"="14_Q2关键化合物_综合感官")
 for(id in intersect(names(groups),selected)) {
  tg<-groups[[id]];nr<-length(tg);label<-labs[match(tg,targets)]
  label[grepl("_all$",tg)]<-paste0(label[grepl("_all$",tg)],"-总体")
  openfig(filenames[[id]],7.8,if(nr>15)8.4 else 6.8)
  par(mar=c(.35,.1,.3,.1));plot.new();plot.window(xlim=c(0,10.05),ylim=c(nr+2.15,0))
  text(.12,.75,"感官维度",adj=0,cex=.9)
  for(j in 1:3) {
   left<-2.25+(j-1)*2.6
   text(left+1.3,.35,paste0("第",c("一","二","三")[j],"候选"),cex=.9)
   text(left+1.02,1.05,"化合物",cex=.77);text(left+2.34,1.05,"频率",cex=.77)
  }
  segments(0,1.6,10.05,1.6,col="#49555E",lwd=.8)
  for(i in seq_along(tg)) {
   y<-i+1.15
   if(i%%2==0)rect(0,y-.49,10.05,y+.49,col="#F3F5F6",border=NA)
   text(.12,y,label[i],adj=0,cex=.8)
   for(j in 1:3) {
    row<-top[top$target==tg[i]&top$rank==j,];stopifnot(nrow(row)==1)
    left<-2.25+(j-1)*2.6
    tokens<-strsplit(row$compound_2csv,"(?<=[ -])",perl=TRUE)[[1]]
    lines<-character();line<-""
    for(token in tokens) {
     candidate<-paste0(line,token)
     if(nchar(line)>0 && strwidth(trimws(candidate),cex=.8)>1.94) {
      lines<-c(lines,trimws(line));line<-token
     } else line<-candidate
    }
    name<-paste(c(lines,trimws(line)),collapse="\n")
    text(left+.1,y,name,adj=c(0,.5),cex=.8)
    rect(left+2.1,y-.36,left+2.57,y+.36,col=mapcolors(row$stability_freq,c(0,1),seqcol),border=NA)
    text(left+2.34,y,formatC(row$stability_freq,digits=2,format="f"),cex=.77,
         col=if(row$stability_freq>.75)"white" else "#20303D")
   }
  }
  for(x in c(2.25,4.85,7.45))segments(x,0,x,nr+1.7,col="#CAD1D6",lwd=.5)
  segments(0,nr+1.7,10.05,nr+1.7,col="#49555E",lwd=.8)
  closefig()
 }
}
# 15-16: original-unit outputs, with actual sample identities.
profile_features<-c("DMS","X2.3.butanedione","X4VG","X.E..2.nonenal","ethyl_acetate","isopentyl_acetate",
                    "ethyl_hexanoate","ethyl_octanoate","phenethyl_acetate","glycerol","acetic_acid.g.L.","lactic_acid.mg.L.")
fmt_number<-function(x) {
 if(!is.finite(x))return("NA")
 if(x!=0 && abs(x)<.001)return(formatC(x,digits=2,format="e"))
 trimws(formatC(x,digits=3,format="fg"))
}
concentration_table<-function(filename,compound,units,values,headings,subheadings=NULL,separate=NULL,value_colors=c("#566773",blue,orange)) {
 nr<-length(compound);openfig(filename,7.8,6.4)
 par(mar=c(.35,.1,.4,.1));plot.new();plot.window(xlim=c(0,10),ylim=c(nr+2,0))
 x<-c(.15,3.8,5.4,7.25,9.1)
 text(x[1],.65,"指标",adj=0,cex=.95);text(x[2],.65,"单位",cex=.9)
 for(j in 1:3) {
  text(x[j+2],if(is.null(subheadings)).65 else .43,headings[j],cex=.9)
  if(!is.null(subheadings))text(x[j+2],1.0,subheadings[j],cex=.71)
 }
 segments(0,1.45,10,1.45,col="#49555E",lwd=.8)
 for(i in seq_len(nr)) {
  y<-i+1
  if(i%%2==0)rect(0,y-.48,10,y+.48,col="#F3F5F6",border=NA)
  text(x[1],y,paste(strwrap(compound[i],width=31),collapse="\n"),adj=c(0,.5),cex=.83)
  text(x[2],y,units[i],cex=.8)
  for(j in 1:3)text(x[j+2],y,fmt_number(values[i,j]),cex=.9,col=value_colors[j])
 }
 if(length(separate))for(i in separate)segments(0,i+1.5,10,i+1.5,col="#A3ADB5",lwd=.65)
 segments(0,nr+1.5,10,nr+1.5,col="#49555E",lwd=.8)
 closefig()
}
if("15" %in% selected) {
 d<-read(file.path(dat,"transfer_concentrations.csv"));beer<-min(d$beer)
 main<-d[d$beer==beer & d$unit_scenario=="attachment_mg_L",];main<-main[match(profile_features,main$feature),]
 alt<-d[d$beer==beer & d$unit_scenario=="alternative_ug_L" & d$feature=="X.E..2.nonenal",]
 main$compound[main$feature=="X.E..2.nonenal"]<-"E-2-nonenal · 情景一";alt$compound<-"E-2-nonenal · 情景二"
 tab<-rbind(main[1:4,],alt,main[5:12,])
 concentration_table("15_Q3迁移前后化合物谱",tab$compound,tab$unit,
                     as.matrix(tab[,c("source","transferred","tripel_reference")]),
                     c(paste("源酒",beer),"主约束迁移后","Tripel 参考"),separate=5)
}
if("16" %in% selected) {
 examples<-read(file.path(dat,"generated_example_chemistry.csv"));examples<-examples[match(styles,examples$style),]
 meta<-read(file.path(dat,"chemical_medians.csv"));meta<-meta[!duplicated(meta$feature),];meta<-meta[match(profile_features,meta$feature),]
 units<-gsub("ug/L","μg/L",meta$unit,fixed=TRUE);units[is.na(units)|units==""]<-"—"
 concentration_table("16_Q3生成化合物谱示例",meta$compound,units,t(as.matrix(examples[,profile_features])),styles,
                     paste0("#",sub(".*_","",examples$sample_id)),separate=4,value_colors=c(blue,orange,green))
}
cat("R version: ",R.version.string,"; graphics: Cairo\n")
