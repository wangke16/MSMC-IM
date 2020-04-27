library(stringr)
library(ggplot2)
library(gridExtra)
library(cowplot)
library(tidyr)
library(RColorBrewer)
library(grid)
interpolation <- function(x, y, xval, yval){
  i = 1
  if ( xval != 0 ){
    while (x[i] < xval) { i = i+1 }
    intersectDistance = (xval - x[i-1]) / (x[i] - x[i-1])
    result = y[i-1] + intersectDistance * (y[i] - y[i-1])
  }
  else if ( yval != 0 ){
    while (y[i] < yval) { i = i+1 }
    intersectDistance = (yval - y[i-1]) / (y[i] - y[i-1])
    result = x[i-1] + intersectDistance * (x[i] - x[i-1])
  }
  return(result)
}
sub_df_m <- function(df_m, pirs, df_T) {
  plot_m <- data.frame()
  sub1_m <- data.frame()
  sub2_m <- data.frame()
  for (pir in pirs) {
    temp_m <- df_m[grepl(paste("^",pir,"$", sep=""), df_m$pair),]
    extra <- cbind.data.frame(unique(temp_m$pair),temp_m$tyears[2:32] - 0.01, temp_m$m[1:31], stringsAsFactors = F)
    names(extra) <- names(temp_m)
    temp_m2 <- rbind(temp_m, extra)[order(rbind(temp_m, extra)$tyears),]
    plot_m <- rbind(plot_m, temp_m2)
    
    quantile <- df_T[grepl(paste("^",pir,"$", sep=""), df_T$pair),]
    sub.df1 <- subset(temp_m2, tyears > quantile$q0.25 & tyears < quantile$q0.75)
    sub.df2 <- subset(temp_m2, tyears > quantile$q0.01 & tyears < quantile$q0.99)
    sub.df1 <- data.frame( mapply(c, data.frame(quantile$pair,quantile$q0.25,temp_m2$m[match(sub.df1$tyears[1], temp_m2$tyears)-1], stringsAsFactors = F), sub.df1,
                                  data.frame(quantile$pair,quantile$q0.75,temp_m2$m[match(tail(sub.df1$tyears, n=1), temp_m2$tyears)+1], stringsAsFactors = F)), stringsAsFactors = F)
    sub.df2 <- data.frame( mapply(c, data.frame(quantile$pair,quantile$q0.01,temp_m2$m[match(sub.df2$tyears[1], temp_m2$tyears)-1], stringsAsFactors = F), sub.df2,
                                  data.frame(quantile$pair,quantile$q0.99,temp_m2$m[match(tail(sub.df2$tyears, n=1), temp_m2$tyears)+1], stringsAsFactors = F)), stringsAsFactors = F)
    names(sub.df1) <- names(temp_m2)
    names(sub.df2) <- names(temp_m2)
    sub1_m <- rbind(sub1_m, sub.df1, stringsAsFactors = FALSE)
    sub2_m <- rbind(sub2_m, sub.df2, stringsAsFactors = FALSE)
  }
  list <- list(plot_m,sub1_m,sub2_m)
  return(list)
}
plot_func <- function(pop1_m, pop1_M, temp_T, sub1_m, sub2_m) {
  pop1_m$pair <- factor(pop1_m$pair, levels = unique(pop1_m$pair))
  sub1_m$pair <- factor(sub1_m$pair, levels = unique(sub1_m$pair))
  sub2_m$pair <- factor(sub2_m$pair, levels = unique(sub2_m$pair))
  temp_T$pair <- factor(temp_T$pair, levels = unique(pop1_m$pair))
  plot_m <- ggplot(pop1_m,aes(x = tyears, y = m)) + geom_line() + geom_ribbon(data=sub1_m, aes(ymin=0, ymax = m), alpha=1, fill = "#98A9EA") +
    geom_ribbon(data=sub2_m, aes(ymin=0, ymax = m), alpha=0.5, fill = "#98A9EA") + geom_vline(data=temp_T, aes(xintercept = q0.50),linetype=4) + 
    facet_grid(pair ~ ., scales = "free_y", switch = "y") +
    scale_x_continuous(limits = c(1e3,2e6), breaks=c(1e3,1e4,2e4,4e4,6e4,NA,1e5,2e5,4e5,6e5,NA,1e6,2e6), expand = c(0, 0), trans = 'log10') +
    theme_bw() +
    theme(plot.title = element_text(hjust=0.5,size=7), axis.text.x = element_text(angle=45, hjust=1,size=7), axis.title.y=element_blank(),
          axis.text.y = element_text(size=7), strip.text.y = element_text(angle = 180,size=8,face="bold"), strip.placement = "outside",
          strip.background = element_blank(), legend.position="none")+ labs(title="migration rate", x = "t(years)", tag=LETTERS[1])
  #axis.title.x = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black")) 
  
  pop1_M$pair <- factor(pop1_M$pair, levels = unique(pop1_M$pair))
  plot_M <- ggplot(pop1_M, aes(x = tyears, y = M)) + geom_ribbon(aes(ymin=0, ymax = M), alpha=1, color="black", fill = "#98A9EA") +
    geom_step(mapping = aes(x=tyears, y=MSMCrCCR), linetype="dashed") + facet_grid(pair ~ ., scales = "free_y") + 
    scale_y_continuous(limits = c(0,1.1), breaks=c(0,0.5,1.0)) +
    scale_x_continuous(limits = c(1e3,2e6), breaks=c(1e3,1e4,2e4,4e4,6e4,NA,1e5,2e5,4e5,6e5,NA,1e6,2e6), expand = c(0, 0), trans = 'log10') +
    theme_bw() + 
    theme(plot.title = element_text(hjust=0.5,size=7), axis.text.x = element_text(angle=45, hjust=1,size=7), axis.title.y=element_blank(),
          axis.text.y = element_text(size=7), strip.text.y = element_blank(), 
          strip.background = element_blank(), legend.position="none") + labs(title="M(t)", x = "t(years)", tag=LETTERS[2])
  #axis.title.x = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #grid.newpage()
  #return(list(grid.draw(cbind(ggplotGrob(plot_m), ggplotGrob(plot_M), size = "first"))))
  return(list(grid.arrange(plot_m, plot_M, ncol=2, widths = c(1.7,1))))
}

setwd("~/pathway/")

gen = 29
odir <- "~/output_pathway/"
fn = "Summary6.v190904.summary_mMrCCR.txt"
all_df = read.table(fn, sep = "\t", stringsAsFactors = F, header = T)
all_df$tyears = all_df$tgens * gen
all_df$pair = gsub('\\s+','',all_df$pair)

all_m <- all_df[,c("pair","tyears","m")]
all_m_M <- all_df[,c("pair","tyears","m","M")]
all_M_rCCR <- all_df[,c("pair","tyears","M","MSMCrCCR")] 

###### Summary on all pairs ######
AFR <- c("San", "Mbuti", "Mandenka", "Dinka", "Yoruba","Mende")
EUA <- c("French", "Dai","Sardinian", "Han")
OTH <- c("Papuan", "Karitiana", "Australian", "Mixe", "Quechua")
pops <- c(AFR, EUA, OTH)
  
yvals <- c(0.01,0.25,0.5,0.75,0.99)
quant.df <- data.frame()
cn=1
for (yval in yvals){
  summarydf <- data.frame()
  for (pair in as.character(unique(all_m_M$pair))) {
    temp_sub <- all_m_M[all_m_M$pair == pair,]
    tempdf <- data.frame(pair=NA, expectedYrs=NA, color=NA)
    pir <- gsub(" ","", pair)
    tempdf$pair <- pir
    if (length(interpolation(temp_sub$tyears, temp_sub$M, 0, yval)) != 0) {
      tempdf$expectedYrs <- interpolation(temp_sub$tyears, temp_sub$M, 0, yval)
    } else { tempdf$expectedYrs <- 0}
    if (str_split_fixed(pir,"_",2)[1]%in%AFR && str_split_fixed(pir,"_",2)[2]%in%AFR) {tempdf$color <- "red"
    } else if (str_split_fixed(pir,"_",2)[1]%in%AFR | str_split_fixed(pir,"_",2)[2]%in%AFR) {tempdf$color <- "blue"
    } else {tempdf$color <- "orange"} ##All other pairs Orange
    summarydf <- rbind(summarydf, tempdf)
  }
  if (cn == 1){ quant.df <- summarydf[,c(1,3,2)]
  } else {quant.df <- cbind(quant.df, summarydf[,2])}
  cn=cn+1
}
names(quant.df) <- c("pair","color","q0.01","q0.25","q0.50","q0.75","q0.99")
quant.df.sort <- data.frame()
for (pop in pops) {
  tempdf <- quant.df[grep(pop, quant.df$pair),]
  for (i in c(1:dim(tempdf)[1])){
    if (str_split_fixed(tempdf$pair[i],"_",2)[1]==pop) {tempdf$orpir[i] <- tempdf$pair[i]} 
    else {tempdf$orpir[i] <- paste(str_split_fixed(tempdf$pair[i],"_",2)[2], str_split_fixed(tempdf$pair[i],"_",2)[1], sep="_")}
  }
  quant.df.sort <- rbind(quant.df.sort, tempdf[order(-tempdf$q0.99),c("pair","color","q0.01","q0.25","q0.50","q0.75","q0.99","orpir")])
}
quant.df.sort <- quant.df.sort[!duplicated(quant.df.sort$pair),]
quant.df.sort <- rbind(quant.df.sort, cbind(quant.df[!quant.df$pair %in% quant.df.sort$pair,], data.frame(orpir="Quechua_Mixe")))
quant.df.sort <- quant.df.sort[c(which(quant.df.sort$color=="red"),which(quant.df.sort$color=="blue"),which(quant.df.sort$color=="orange")),]
quant.df.sort[quant.df.sort$orpir == "Dinka_Yoruba", ]$q0.01 <- 1000
quant.df.sort[quant.df.sort$orpir == "Dinka_Mende", ]$q0.01 <- 1000

###### Plot Fig7a ######
quant.df.sort$orpir <- factor(quant.df.sort$orpir, levels=quant.df.sort$orpir)
qplot <- ggplot(quant.df.sort, aes(orpir, fill=color, color=color)) + scale_fill_identity() +
  geom_boxplot(aes(ymin = q0.01, lower = q0.25, middle = q0.50, upper = q0.75, ymax = q0.99), stat = "identity",alpha=0.3) +
  geom_errorbar(aes(ymin = q0.01, ymax = q0.99), position = position_dodge(1)) + #geom_errorbar(aes(ymin = q0.05, ymax = q0.95)) +
  scale_color_manual(values = c("blue","orange","red")) + #geom_hline(yintercept=1e6, linetype="dashed") +
  scale_y_continuous(limits = c(1e3,1.2e6), breaks=c(1e3,1e4,2e4,4e4,6e4,NA,1e5,2e5,4e5,6e5,NA,1e6), expand = c(0, 0), trans="log10") +
  theme_bw() + theme(axis.text.x = element_blank(), ##text(angle = 90, hjust=1, size = 7),
                     axis.title.x = element_blank(),
                     axis.title.y = element_text(size = 7),
                     axis.text.y = element_text(size = 5),
                     legend.position="none") + labs(y="t(years)")

###### Plot Fig7 ######
xvals <- c(1e5,2e5,3e5,6e5,8e5,1e6)
cut.df <- data.frame()
cn=1
for (xval in xvals){
  summarydf <- data.frame()
  for (pair in as.character(unique(all_m_M$pair))) {
    temp_sub <- all_m_M[all_m_M$pair == pair,]
    tempdf <- data.frame(pair=NA, residueM=NA, color=NA)
    pir <- gsub(" ","", pair)
    tempdf$pair <- pir
    if (length(interpolation(temp_sub$tyears, temp_sub$M, 0, yval)) != 0) {
      tempdf$residueM <- 1-interpolation(temp_sub$tyears, temp_sub$M, xval, 0)
    } else { tempdf$residueM <- 0}
    if (str_split_fixed(pir,"_",2)[1]%in%AFR && str_split_fixed(pir,"_",2)[2]%in%AFR) {tempdf$color <- "red"
    } else if (str_split_fixed(pir,"_",2)[1]%in%AFR | str_split_fixed(pir,"_",2)[2]%in%AFR) {tempdf$color <- "blue"
    } else {tempdf$color <- "orange"} ##All other pairs Orange
    summarydf <- rbind(summarydf, tempdf)
  }
  if (cn == 1){ cut.df <- summarydf[,c(1,3,2)]
  } else {cut.df <- cbind(cut.df, summarydf[,2])}
  cn=cn+1
}

names(cut.df) <- c("pair","color","t3e5","t6e5","t8e5","t1e6")
cut.df.sort <- data.frame()
for (pop in pops) {
  tempdf <- cut.df[grep(pop, cut.df$pair),]
  for (i in c(1:dim(tempdf)[1])){
    if (str_split_fixed(tempdf$pair[i],"_",2)[1]==pop) {tempdf$orpir[i] <- tempdf$pair[i]} 
    else {tempdf$orpir[i] <- paste(str_split_fixed(tempdf$pair[i],"_",2)[2], str_split_fixed(tempdf$pair[i],"_",2)[1], sep="_")}
  }
  cut.df.sort <- rbind(cut.df.sort, tempdf[order(-tempdf$"t1e6"),c("pair","color","t3e5","t6e5","t8e5","t1e6","orpir")])
}
cut.df.sort <- cut.df.sort[!duplicated(cut.df.sort$pair),]
cut.df.sort <- rbind(cut.df.sort, cbind(cut.df[!cut.df$pair %in% cut.df.sort$pair,], data.frame(orpir="Quechua_Mixe")))
cut.df.sort$t3e5 <- cut.df.sort$t3e5 - cut.df.sort$t6e5
cut.df.sort$t6e5 <- cut.df.sort$t6e5 - cut.df.sort$t8e5
cut.df.sort$t8e5 <- cut.df.sort$t8e5 - cut.df.sort$t1e6

#cut.long.df <- cut.df.sort[,c(9,2:8)] %>% gather(cutoff, residuesM, c("t1e5","t2e5","t3e5","t6e5","t8e5","t1e6"))
cut.long.df <- cut.df.sort[,c(9,2:8)] %>% gather(cutoff, residuesM, c("t3e5","t6e5","t8e5","t1e6"))
cut.long.df <- cut.long.df[c(which(cut.long.df$color=="red"),which(cut.long.df$color=="blue"),which(cut.long.df$color=="orange")),]
cut.long.df$orpir <- factor(cut.long.df$orpir, levels = unique(cut.long.df$orpir))
cut.long.df$cutoff <- factor(cut.long.df$cutoff,unique(cut.long.df$cutoff))

cplot <- ggplot(cut.long.df) + geom_bar(aes(x = orpir, y = residuesM, fill = cutoff), stat="identity") +
         scale_y_continuous(breaks=c(0,0.01,0.02,0.03,0.04,0.05,0.10,0.15,0.20),expand = c(0,0)) + labs(y = "1-M(t)") + theme_bw() + 
         scale_fill_manual(name = "", values = c("grey70","grey50","grey30","black"),
                             labels = c("300 thousand years ago","600 thousand years ago",
                                        "800 thousand years ago","1 million years ago")) +
         theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5, size = 7), axis.title.x = element_blank(),
               axis.title.y = element_text(size = 7), axis.text.y = element_text(size = 5), 
               legend.position=c(0.9,0.7), legend.title = element_text(size = 7), legend.background = element_rect("transparent"),
               legend.key.size = unit(0.25, "cm"), legend.text = element_text(size = 7))

ggsave(paste(odir,"Figure7",sep=""),plot_grid(qplot,cplot,ncol=1,align="v",rel_heights = c(1,1.2)), width = 8, height = 5)


###### Plot three main figures - fig4,5,6 ######
AFRpir <-c("Yoruba_Mandenka", "Yoruba_Dinka", "Dinka_Mbuti", "Dinka_San", "Dinka_Mandenka", "Mbuti_Mandenka", "Yoruba_Mbuti", "Yoruba_San", "San_Mandenka", "San_Mbuti")
OOFApir <- c("Yoruba_Karitiana","Yoruba_Australian","Yoruba_Papuan","Yoruba_Dai","Yoruba_Han","Yoruba_French","Yoruba_Sardinian")
NAFRpir <-c("Karitiana_Quechua","Karitiana_Mixe","Quechua_Mixe","Papuan_Australian","Han_Karitiana","Papuan_Dai","Han_Papuan","French_Papuan","Papuan_Sardinian","French_Han")

pir_list <- list(AFRpir, OOFApir, NAFRpir)
ofn_list <- list("Fig4.within_AFRpairs", "Fig5.within_OofAFRpairs", "Fig6.within_nonAFRpairs")
for (i in c(1:3)) {
	sortpir <- pir_list[[i]]
	ofn <- ofn_list[[i]]
	
	pop1_m <- all_m[order(match(all_m$pair, sortpir),na.last = NA),]
	pop1_M <- all_M_rCCR[order(match(all_M_rCCR$pair, sortpir),na.last = NA),]
	pop1_T <- quant.df.sort[order(match(quant.df.sort$pair, sortpir),na.last = NA), ]
	list_df <- sub_df_m(pop1_m, sortpir, pop1_T)
	sub_plot_m <- list_df[[1]]
	sub1_m <- transform(list_df[[2]], tyears = as.numeric(tyears), m = as.numeric(m))
	sub2_m <- transform(list_df[[3]], tyears = as.numeric(tyears), m = as.numeric(m))
	plot.list <- plot_func(sub_plot_m, pop1_M, pop1_T, sub1_m, sub2_m) 
	ggsave(paste(odir,ofn,".pdf",sep=""), plot = plot.list[[1]], width = 17, height = 1.9 * length(sortpir), units = "cm")
}