# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 只保留03、05、17–21；模型计算与本文件分离。
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); setwd(root)
selected <- if(length(args)>1) strsplit(args[2],",",fixed=TRUE)[[1]] else c("03","05","17","18","19","20","21")
stopifnot(all(selected %in% c("03","05","17","18","19","20","21")))
out <- file.path(root,"img")
dat <- "data/processed/paper_plot_data"
set.seed(42)
read <- function(p) read.csv(p,check.names=FALSE,stringsAsFactors=FALSE)
blue <- "#376E91"; orange <- "#BD7546"; green <- "#45857A"; gray <- "#A9AFB5"
shortlabel <- function(x,n) {x<-gsub("[()（）]","",x);ifelse(nchar(x)>n,paste0(substr(x,1,n-1),"…"),x)}
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
styles<-c("Tripel","Kriek","Lambic")
T<-read(file.path(dat,"感官维度标签.csv"));targets<-T$target;labs<-T$label
matlong <- function(d,row,col,value,rows,cols) {
 M<-matrix(NA_real_,length(rows),length(cols),dimnames=list(rows,cols))
 for(i in seq_len(nrow(d)))if(d[[row]][i]%in%rows && d[[col]][i]%in%cols)
   M[as.character(d[[row]][i]),as.character(d[[col]][i])]<-d[[value]][i]
 M
}
if("03" %in% selected) {
 cards<-read("data/q1/完整评分卡.csv");mapping<-read("data/q1/评分卡栏位映射.csv")
 gradecol<-c("#F7FAFC","#CCE2F0","#91BCD8","#5594BD","#256C9F","#104571","#041A36")
 openfig("03_问题一评分示例",7.6,7.4)
 layout(matrix(c(1,2,3,3),2,2,byrow=TRUE),heights=c(6.8,.6))
 par(cex=1,oma=c(.15,0,.25,0))
 for(side in c("Aroma","Flavour")) {
  mp<-mapping[mapping$side==side,];d<-cards[cards$side==side & cards$beer%in%1:3,]
  M<-matlong(d,"item","beer","relative_grade",mp$item,as.character(1:3))
  labels<-labs[match(mp$target,targets)];labels[grepl("_all$",mp$item)]<-paste0(labels[grepl("_all$",mp$item)],"-总体")
  labels[is.na(labels)]<-"乙醛"
  par(mar=c(2.5,7.2,2.2,.5));heat(M,labels,NULL,c(0,6),gradecol)
  axis(1,at=1:3,labels=paste("Beer",1:3),tick=FALSE,las=1,cex.axis=.85)
  mtext(paste(side,"卡片栏位"),3,line=.7,cex=.95)
  for(i in seq_len(nrow(M)))for(j in 1:3)text(j,i,ifelse(is.finite(M[i,j]),as.character(as.integer(M[i,j])),"NA"),cex=.9,col=ifelse(is.finite(M[i,j]) && M[i,j]>=4,"white","#12202A"))
 }
 par(mar=c(2,.2,.2,.2));plot.new();plot.window(xlim=c(-1.4,8.4),ylim=c(0,1))
 rect(0:6+.035,.5,1:7-.035,.9,col=gradecol,border="white",lwd=.5)
 axis(1,at=(0:6)+.5,labels=0:6,pos=.5,cex.axis=.8);mtext("相对等级",1,line=.6,cex=.9)
 closefig()
}
if("05" %in% selected) {
n<-read("data/q2/网络节点.csv");e<-read("data/q2/网络连边.csv")
p<-read(file.path(dat,"关联网络布局坐标.csv"));p<-p[match(n$feature,p$feature),];g<-sort(unique(n$community))
pal<-c("#497CA0","#BB8952","#60978A","#B66D7D","#8A81A9","#7F975B","#A17C60","#6E9FAC")
openfig("05_问题二关联网络",8,8.2)
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
points(p$x,p$y,pch=21,bg=pal[match(n$community,g)],col=adjustcolor("#344B58",.7),lwd=.55,cex=1.4*(.42+.10*sqrt(n$degree)))
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
style_colors<-c(Tripel=blue,Kriek=orange,Lambic=green,"Pils/Lager"="#BA9844")
if("17" %in% selected) {
 observed<-read(file.path(dat,"校准浓度与风格.csv"));counts<-sort(table(observed$style),decreasing=TRUE)
 count_colors<-rep("#B8C4CB",length(counts));highlight<-names(counts)%in%names(style_colors)
 count_colors[highlight]<-style_colors[names(counts)[highlight]]
 openfig("17_风格样本分布",8.4,5.5)
 par(mar=c(3.4,10.7,.75,.75))
 yy<-barplot(rev(counts),horiz=TRUE,names.arg=rev(names(counts)),col=rev(count_colors),border=NA,
             xlim=c(0,max(counts)*1.12),las=1,cex.names=.78,space=.24,xlab="酒样数")
 text(as.numeric(rev(counts))+.45,yy,as.integer(rev(counts)),adj=0,cex=.8)
 closefig()
}
if("18" %in% selected) {
 top<-read("data/q2/前三关键化合物.csv")
 cases<-c("A_malt_all","A_hops_citrus","A_esters_fruity","dms","diacetyl","X4vg","t2n","stale_hops")
 case_labels<-c("麦芽香","酒花香 · 柑橘","酯香 · 果香","DMS","双乙酰","4VG","T2N","陈旧酒花")
 palette<-c("#376E91","#709CB5","#AAC4D3")
 openfig("18_问题二关键化合物柱状图",9.0,8.3);par(oma=c(1.6,0,.5,0))
 layout(matrix(1:8,4,2,byrow=TRUE))
 par(cex=1)
 for(k in seq_along(cases)) {
  z<-top[top$target==cases[k],];z<-z[order(z$rank),];stopifnot(nrow(z)==3)
  labels<-paste0(z$rank,"  ",vapply(z$compound_2csv,function(s)paste(strwrap(s,width=25),collapse="\n"),character(1)))
  par(mar=c(2.2,11.4,1.6,.95))
  plot.new();plot.window(xlim=c(0,1.12),ylim=c(.4,3.6))
  abline(v=c(.25,.5,.75,1),col="#E7EBED",lwd=.65)
  rect(0,3:1-.28,z$stability_freq,3:1+.28,col=palette,border=NA)
  axis(2,at=3:1,labels=labels,tick=FALSE,las=2,cex.axis=.77)
  axis(1,at=c(0,.25,.5,.75,1),labels=c(0,25,50,75,100),cex.axis=.72)
  inside<-z$stability_freq>.85
  for(i in 1:3)text(z$stability_freq[i]+if(inside[i])-.025 else .025,4-i,
       paste0(formatC(z$stability_freq[i]*100,format="f",digits=1),"%"),
       adj=if(inside[i])1 else 0,col=if(inside[i])"white" else "#20272D",cex=.76)
  mtext(case_labels[k],3,line=.35,adj=0,cex=.95)
 }
 mtext("Top-3 入选频率 / %",1,outer=TRUE,line=.1,cex=.88)
 closefig()
}
if("19" %in% selected) {
 points_data<-read(file.path(dat,"主成分投影坐标.csv"))
 variance<-read(file.path(dat,"主成分解释率.csv"))$explained_variance_ratio
 real<-points_data[points_data$dataset=="real",]
 ranges<-lapply(c("pc1","pc2"),function(col){r<-range(points_data[[col]]);r+c(-1,1)*diff(r)*.055})
 openfig("19_问题三生成与迁移主成分分析",8.8,7.1);par(oma=c(.9,0,.6,0))
 layout(rbind(matrix(1:4,2,2,byrow=TRUE),c(5,5)),heights=c(3.2,3.2,.65))
 par(cex=1)
 for(k in 1:4) {
  par(mar=c(3.3,3.9,1.6,.7))
  plot.new();plot.window(xlim=ranges[[1]],ylim=ranges[[2]])
  abline(h=0,v=0,col="#DFE4E6",lwd=.6)
  points(real$pc1,real$pc2,pch=16,col=adjustcolor("#BBC0C1",.5),cex=.45)
  for(st in styles) {
   r<-real[real$style==st,];hull<-chull(r$pc1,r$pc2)
   polygon(r$pc1[hull],r$pc2[hull],col=adjustcolor(style_colors[st],.055),
           border=adjustcolor(style_colors[st],.6),lwd=.75)
  }
  if(k==1)for(st in c(styles,"Pils/Lager")) {
   z<-real[real$style==st,];points(z$pc1,z$pc2,pch=21,bg=style_colors[st],col="white",cex=.8,lwd=.35)
  }
  if(k%in%c(2,3))for(st in styles) {
   z<-points_data[points_data$dataset==if(k==2)"copula" else "ot_rank",]
   z<-z[z$style==st,];points(z$pc1,z$pc2,pch=16,col=adjustcolor(style_colors[st],.35),cex=.45)
  }
  if(k==4) {
   a<-points_data[points_data$dataset=="transfer_source",]
   b<-points_data[points_data$dataset=="transfer_excess",];b<-b[match(a$sample_id,b$sample_id),]
   arrows(a$pc1,a$pc2,b$pc1,b$pc2,col=adjustcolor(blue,.5),length=.07,lwd=.8)
   points(a$pc1,a$pc2,pch=21,bg=style_colors["Pils/Lager"],col="white",cex=.85,lwd=.4)
   points(b$pc1,b$pc2,pch=21,bg=blue,col="white",cex=.9,lwd=.4)
  }
  axis(1,cex.axis=.77);axis(2,las=1,cex.axis=.77);box(bty="l")
  mtext(paste0("PC1 · ",formatC(100*variance[1],digits=1,format="f"),"%"),1,line=2.0,cex=.82)
  mtext(paste0("PC2 · ",formatC(100*variance[2],digits=1,format="f"),"%"),2,line=2.4,cex=.82,las=0)
  mtext(c("a  实测风格","b  Copula 生成","c  OT 秩校准生成","d  Pils/Lager → Tripel")[k],3,line=.35,adj=0,cex=.94)
 }
 par(mar=c(0,0,0,0));plot.new()
 legend("center",legend=c(styles,"Pils/Lager","实测背景"),col=c(style_colors[c(styles,"Pils/Lager")],"#BBC0C1"),
        pch=16,bty="n",horiz=TRUE,cex=.88)
 closefig()
}
radar_panel<-function(values,labels,colors,lty=c(1,2),fill=NULL) {
 n<-length(labels);theta<-pi/2-seq(0,2*pi,length.out=n+1)[-(n+1)]
 par(mar=c(.3,.3,1.4,.3))
 plot.new();plot.window(xlim=c(-1.56,1.56),ylim=c(-1.34,1.34),asp=1)
 if(length(fill))for(k in seq_along(fill))if(!is.na(fill[k])) {
  r<-values[k,]/100;polygon(r*cos(theta),r*sin(theta),col=fill[k],border=NA)
 }
 for(r in c(.25,.5,.75,1)) {
  t<-seq(0,2*pi,length.out=241)
  lines(r*cos(t),r*sin(t),col=if(r==1)"#9EAAB0" else "#D7DFE2",lwd=if(r==1).8 else .6)
 }
 segments(0,0,cos(theta),sin(theta),col="#D7DFE2",lwd=.6)
 for(k in seq_len(nrow(values))) {
  r<-values[k,]/100
  lines(c(r*cos(theta),r[1]*cos(theta[1])),c(r*sin(theta),r[1]*sin(theta[1])),
        col=colors[k],lty=lty[k],lwd=if(k==1)1.8 else 1.35)
  if(k==1)points(r*cos(theta),r*sin(theta),pch=16,col=colors[k],cex=.45)
 }
 label_x<-1.14*cos(theta);label_y<-1.14*sin(theta)
 for(i in seq_len(n))text(label_x[i],label_y[i],labels[i],cex=.88,
                         adj=c(if(abs(label_x[i])<.05).5 else if(label_x[i]>0)0 else 1,.5),xpd=NA)
 angle<-pi/2-pi/n
 for(v in c(25,50,75,100))text(v/100*cos(angle),v/100*sin(angle),v,col="#64747D",cex=.63,
                              adj=c(.5,.5))
}
if("20" %in% selected) {
 z<-read(file.path(dat,"雷达画像数据.csv"))
 dims<-c("A_esters_all","A_hops_all","A_malt_all","bitternes","body","alcohol","sweetness","acidity")
 labels<-c("酯香","酒花香","麦芽香","苦味","酒体","酒精","甜味","酸度")
 openfig("20_问题三生成感官雷达",8.8,7.4);par(oma=c(.15,.25,.65,.25))
 layout(rbind(matrix(1:4,2,2,byrow=TRUE),c(5,5)),heights=c(3.3,3.3,.65))
 par(cex=1)
 for(k in seq_along(styles)) {
  a<-z[z$kind=="generated"&z$style==styles[k]&z$model=="copula",]
  b<-z[z$kind=="real"&z$style==styles[k],]
  values<-rbind(a$percentile[match(dims,a$target)],b$percentile[match(dims,b$target)])
  radar_panel(values,labels,c(style_colors[styles[k]],"#5D6970"),fill=c(adjustcolor(style_colors[styles[k]],.12),NA))
  mtext(styles[k],3,line=.1,cex=1.05)
 }
 values<-t(sapply(styles,function(st) {
  a<-z[z$kind=="generated"&z$style==st&z$model=="copula",]
  a$percentile[match(dims,a$target)]
 }))
 radar_panel(values,labels,style_colors[styles],lty=c(1,1,1),fill=rep(NA,3))
 mtext("三风格生成对照",3,line=.1,cex=1.05)
 par(mar=c(0,0,0,0));plot.new()
 legend("center",legend=c(styles,"实测中位画像"),col=c(style_colors[styles],"#5D6970"),lty=c(1,1,1,2),lwd=c(1.8,1.35,1.35,1.35),
        bty="n",horiz=TRUE,cex=.85)
 closefig()
}
if("21" %in% selected) {
 z<-read(file.path(dat,"雷达画像数据.csv"));tr<-read("data/q3/迁移化合物谱.csv")
 dims<-list(c("A_malt_all","A_hops_all","A_esters_all","F_esters_all","F_hops_all","F_malt_all"),
            c("acidity","bitternes","sweetness","alcohol","body","overall"))
 labels<-list(c("麦芽香","酒花香","酯香","酯风味","酒花风味","麦芽风味"),
              c("酸度","苦味","甜味","酒精","酒体","整体"))
 profiles<-list(z[z$kind=="transfer"&z$model=="attachment_mg_L_excess",],
                z[z$kind=="transfer"&z$model=="attachment_mg_L_source",],
                z[z$kind=="real"&z$style=="Tripel",],
                z[z$kind=="transfer"&z$model=="attachment_mg_L_raw",])
 openfig("21_问题三迁移画像与缺陷",8.8,7.3);par(oma=c(.2,.25,.65,.25))
 layout(rbind(c(1,2),c(3,3),c(4,5),c(6,6)),heights=c(3.45,.55,2.65,.55))
 par(cex=1)
 for(k in 1:2) {
  values<-t(sapply(profiles,function(a)a$percentile[match(dims[[k]],a$target)]))
  radar_panel(values,labels[[k]],c(blue,orange,green,"#929BA0"),lty=c(1,1,2,3),
              fill=c(adjustcolor(blue,.12),adjustcolor(orange,.07),NA,NA))
  mtext(c("a  香气与风味","b  口感与整体")[k],3,line=.1,cex=.98)
 }
 par(mar=c(0,0,0,0));plot.new()
 legend("center",legend=c("主约束迁移","源酒","Tripel 参考","原始 OT"),col=c(blue,orange,green,"#929BA0"),
        lty=c(1,1,2,3),lwd=c(1.8,1.35,1.35,1.35),bty="n",horiz=TRUE,cex=.86)
 for(k in 1:2) {
  scenario<-c("attachment_mg_L","alternative_ug_L")[k]
  group<-tr[tr$unit_scenario==scenario,];ids<-sort(unique(group$sample_id))
  original<-group[group$variant=="source",];original<-original[match(ids,original$sample_id),]
  cc<-c("DMS","X2.3.butanedione","X4VG","X.E..2.nonenal");tau<-c(17.25,.09,.1,if(k==1).0001 else .1)
  counts<-sapply(c("raw","excess"),function(method) {
   moved<-group[group$variant==method,];moved<-moved[match(ids,moved$sample_id),]
   colSums(pmax(sweep(as.matrix(moved[,cc]),2,tau,"/")-1,0)>
           pmax(sweep(as.matrix(original[,cc]),2,tau,"/")-1,0)+1e-7)
  })
  par(mar=c(2.8,3.7,1.7,.65))
  mid<-barplot(t(counts),beside=TRUE,names.arg=c("DMS","双乙酰","4VG","T2N"),
               col=c("#AEB7BD",blue),border=NA,ylim=c(0,13),yaxt="n",cex.names=.85,space=c(.15,.9))
  axis(2,at=c(0,3,6,9,11),las=1,cex.axis=.78)
  text(mid,as.numeric(t(counts))+.55,as.integer(t(counts)),cex=.9,col=rep(c("#697780",blue),4))
  mtext("恶化酒样数",2,line=2.35,cex=.84,las=0)
  mtext(c("c  T2N 按附件 mg/L","d  T2N 按替代 μg/L")[k],3,line=.4,adj=0,cex=.94)
 }
 par(mar=c(0,0,0,0));plot.new()
 legend("center",legend=c("原始 OT","主约束迁移"),fill=c("#AEB7BD",blue),border=NA,bty="n",horiz=TRUE,cex=.88)
 closefig()
}
cat("R version: ",R.version.string,"; graphics: Cairo\n")
