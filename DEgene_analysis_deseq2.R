if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("DESeq2")
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

base <- c("/home/heavens/CMU/FISH_Clustering/FICT_Sample/excitatory_gene/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/excitatory_spatio/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/inhabitory_gene/",
          "/home/heavens/CMU/FISH_Clustering/FICT_Sample/inhabitory_spatio/")
for (b in base)
{
  DEgene(paste(b,'/count.csv',sep=''),paste(b,'/meta.csv',sep=''),b)
}
