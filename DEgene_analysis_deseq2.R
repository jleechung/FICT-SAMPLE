#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install("DESeq2")
library( "DESeq2" )
library(ggplot2)
DEgene <- function(count_f,meta_f,save_f){
countData <- read.csv(count_f, header = TRUE, sep = ",")
metaData <- read.csv(meta_f, header = TRUE, sep = ",")
dds <- DESeqDataSetFromMatrix(countData=countData, 
                              colData=metaData, 
                              design=~subcluster, tidy = TRUE)
dds <- DESeq(dds,betaPrior=TRUE)
clusters = unique(metaData$subcluster)
cluster_ids = c()
for (i in seq(length(clusters)))
{
  cluster_ids = c(cluster_ids,paste("subclusterc",toString(i-1),sep=''))
}

for (i in seq(length(clusters)))
{
res <- results(dds,contrast = list(cluster_ids[i],cluster_ids[-i]),listValues=c(1,-1/2))
head(res) #let's look at the results table
sort_gene = rownames(res)[order(res$pvalue)]
save_file = paste(save_f,"/result",toString(i),".csv",sep='')
file.create(save_file)
write.csv(res,file = save_file,row.names = T,col.names = T)
}
}
args = commandArgs(trailingOnly=TRUE)
if (length(args)<2) {
  stop("Two arguments: input output should be passed to the scripts.n", call.=FALSE)
}
DEgene(paste(args[1],'/count.csv',sep=''),paste(args[1],'/meta.csv',sep=''),args[2])

