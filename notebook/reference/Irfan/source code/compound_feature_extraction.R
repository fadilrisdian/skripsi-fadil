########### IMPORT LIBRARY ###########

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(version = "3.12")
BiocManager::install("ChemmineR")
BiocManager::install("ChemmineOB")
BiocManager::install("rcdk")

install.packages('rJava')

library(ChemmineR)
library(rcdk)

########### TEST FILE (HERBALDB) ###########

getwd()

# READ THE SDF FILES
sdfstr <- read.SDFstr('data/lampiran/com-test-herbaldb/first iteration/com-test-herbaldb-pubchem-id.sdf')
sdfstr <- read.SDFset('data/lampiran/com-test-herbaldb/first iteration/com-test-herbaldb-pubchem-id.sdf')
sdfstr[[1]]

# VALIDATE AND FILTER
valid <- validSDF(sdfstr)  
sdfstr <- sdfstr[valid]
sdfstr

# SET THE ID
cid(sdfstr) <- sdfid(sdfstr)

# EXTRACT THE FINGERPRINT
fpset <- fp2bit(sdfstr, type=2)
View(fpset)

sdfstr
smi <- sdf2smiles(sdfstr)
View(as.character(smi))

write.csv(fpset, 'data/com-test-herbaldb-fp.csv')

########### TRAIN FILE (SUPERTARGET) ###########

# READ THE SDF FILES
sdfstr <- read.SDFset('data/com-train-positive-interaction.sdf')
sdfstr

# VALIDATE AND FILTER
valid <- validSDF(sdfstr)  
sdfstr <- sdfstr[valid]
sdfstr

# SET THE ID
cid(sdfstr) <- sdfid(sdfstr)

# EXTRACT THE FINGERPRINT
fpset <- fp2bit(sdfstr, type=2)
View(fpset)

write.csv(fpset, 'data/com-train-pos-fp.csv')

########### TEST FILE (HERBALDB) THAT MISSING ###########

# READ THE SDF FILES
sdfstr <- read.SDFset('data/missing-test-fp.sdf')
sdfstr
sdfstr[[1]]

# VALIDATE AND FILTER
valid <- validSDF(sdfstr)  
sdfstr <- sdfstr[valid]
sdfstr

# SET THE ID
cid(sdfstr) <- sdfid(sdfstr)

# EXTRACT THE FINGERPRINT
fpset <- fp2bit(sdfstr, type=2)
View(fpset)

write.csv(fpset, 'data/missing-test-fp.csv')

########### TEST FILE (HERBALDB) THAT MISSING ###########

# READ THE SDF FILES
sdfstr <- read.SDFset('data/missing-train.sdf')
sdfstr
sdfstr[[9]]

# VALIDATE AND FILTER
valid <- validSDF(sdfstr)  
valid
sdfstr <- sdfstr[valid]
sdfstr

# SET THE ID
cid(sdfstr) <- sdfid(sdfstr)

# EXTRACT THE FINGERPRINT
fpset <- fp2bit(sdfstr, type=2)
View(fpset)

write.csv(fpset, 'data/missing-train-fp.csv')
