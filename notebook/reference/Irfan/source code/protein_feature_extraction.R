library(protr)
library(comprehenr)


# FEATURE EXTRACTION
aac_list <- list()
dc_list <- list()
tc_list <- list()
qso_list <- list()

idx = as.numeric(1)
path = './data/lampiran/prot-fasta/'
  
for (i in list.files(path, recursive=FALSE)) {
  aac_list[[idx]] <- extractAAC(readFASTA(paste(path, i, sep=''))[[1]])
  dc_list[[idx]] <- extractDC(readFASTA(paste(path, i, sep=''))[[1]])
  tc_list[[idx]] <- extractTC(readFASTA(paste(path, i, sep=''))[[1]])
  qso_list[[idx]] <- extractQSO(readFASTA(paste(path, i, sep=''))[[1]])
  idx = idx + 1
}

# CREATE DATA FRAME
df_aac <- data.frame(matrix(unlist(aac_list), nrow=length(aac_list)))
head(df_aac)
colnames(df_aac) <- to_vec(for(i in 1:20) paste("aac", i, sep='_'))
head(df_aac)

df_dc <- data.frame(matrix(unlist(dc_list), nrow=length(dc_list)))
head(df_dc)
colnames(df_dc) <- to_vec(for(i in 1:400) paste("dc", i, sep='_'))
head(df_dc)

df_tc <- data.frame(matrix(unlist(tc_list), nrow=length(tc_list)))
head(df_tc)
colnames(df_tc) <- to_vec(for(i in 1:8000) paste("tc", i, sep='_'))
head(df_tc)

df_qso <- data.frame(matrix(unlist(qso_list), nrow=length(qso_list)))
head(df_qso)
colnames(df_qso) <- to_vec(for(i in 1:100) paste("qso", i, sep='_'))
head(df_qso)

list.files(path, recursive=FALSE)

uniprot_id <- c("B2MG_HUMAN", "EGFR_HUMAN", "HDAC3_HUMAN", "HMOX1_HUMAN", "IFNG_HUMAN", "IL1B_HUMAN", "IL6_HUMAN", "IL8_HUMAN", "LOX5_HUMAN", "MMP12_HUMAN", "PERM_HUMAN", "PGH2_HUMAN", "RASH_HUMAN", "SDF1_HUMAN", "TF65_HUMAN", "TNFA_HUMAN", "VEGFA_HUMAN")

df_aac$uniprot_id <- uniprot_id
df_dc$uniprot_id <- uniprot_id
df_tc$uniprot_id <- uniprot_id
df_qso$uniprot_id <- uniprot_id

# SAVE THE FILES
write.csv(df_aac, 'protein_aac.csv', row.names=FALSE)
write.csv(df_dc, 'protein_dc.csv', row.names=FALSE)
write.csv(df_tc, 'protein_tc.csv', row.names=FALSE)
write.csv(df_qso, 'protein_qso.csv', row.names=FALSE)
