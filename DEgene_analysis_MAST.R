freq_expressed <- 0.2
FCTHRESHOLD <- log2(1.5)

DEgene <- function(count_f,meta_f,save_f){
  countData <- read.csv(count_f, header = TRUE, row.names = 1, sep = ",")
  expressionData <- log2(countData+1)
  metaData <- read.csv(meta_f, header = TRUE, row.names = 1, sep = ",")
  fdat <- rownames(countData)
  fdat <- cbind(primerid = fdat,fdat)
  rownames(fdat) <- fdat[,'primerid']
  colnames(fdat) <- c('primerid','symbolid')
  fdat <- as.data.frame(fdat)
  metaData <- cbind(wellKey = rownames(metaData),metaData)
  sca <- FromMatrix(as.matrix(expressionData), metaData, fdat)
  cdr2 <-colSums(assay(sca)>0)
  colData(sca)$cngeneson <- scale(cdr2)
  cond<-factor(colData(sca)$subcluster)
  for (l in seq(length(levels(cond))))
  {
    cond<-factor(colData(sca)$subcluster)
    new_level = levels(cond)
    new_level[-l] = "res"
    levels(cond) <- new_level
    new_cond<-relevel(cond,"res")
    colData(sca)$condition<-new_cond
    zlmCond <- zlm(~condition + cngeneson, sca)
    current_sub <- paste('condition',levels(new_cond)[2],sep = '')
    summaryCond <- summary(zlmCond, doLRT=current_sub) 
    summaryDt <- summaryCond$datatable
    fcHurdle <- merge(summaryDt[contrast==current_sub & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
                      summaryDt[contrast==current_sub & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients
    fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    fcHurdleSig <- merge(fcHurdle[fdr<.05 & abs(coef)>FCTHRESHOLD], as.data.table(mcols(sca)), by='primerid')
    setorder(fcHurdleSig, fdr)
    save_file = paste(save_f,"/MASTresult",toString(l),".csv",sep='')
    file.create(save_file)
    write.csv(fcHurdleSig,file = save_file,row.names = T,col.names = T)
  }
  }
base <- c("/home/heavens/CMU/FISH_Clustering/FICT_Sample/excitatory_gene/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/excitatory_spatio/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/inhabitory_gene/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/inhabitory_spatio/")
for (b in base)
{
  DEgene(paste(b,'/count.csv',sep=''),paste(b,'/meta.csv',sep=''),b)
}

