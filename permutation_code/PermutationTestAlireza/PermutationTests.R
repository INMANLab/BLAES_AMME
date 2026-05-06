########################### Initialization ##########################
rm(list=ls(all=TRUE))

library(pacman)
p_load(data.table,
       reshape2,
       ez,
       lme4,
       lmerTest,
       ggplot2,
       grid,
       tidyr,
       plyr,
       dplyr,
       effects,
       gridExtra,
       DescTools,
       Cairo, #alternate image writing package with superior performance.
       corrplot,
       knitr,
       PerformanceAnalytics,
       afex,
       ggpubr,
       readxl,
       officer,
       psych,
       rstatix,
       emmeans,
       ggformula,
       export,
       scales,
       correlation,
       sjPlot)

# To install HMLET package run the following lines
# install.packages("devtools")
# devtools::install_github("Alireza-Kazemi/HMLET/tree/main/RPackage")
library(HMLET)




RD = "C:\\Users\\alire\\Box\\InmanLab\\BLAES_data\\dissertation\\AMME_BLAES\\outputs\\csvs\\"
source("C:/Users/alire/Box/InmanLab/BLAES_data/dissertation/AMME_BLAES/permutation_code/PermutationTestAlireza/SourceFuncs.R")


# WD = "C:\\Users\\alire\\Box\\InmanLab\\BLAES_data\\dissertation\\AMME_BLAES\\outputs\\PermutationOutputsAlireza\\"
############# functions and scripts ============================================
# =============================== POWER analyses (encoding + retrieval) =======
cat("\n========== ENCODING POWER ==========\n")

enc_power_csv <- file.path("outputs", "csvs", "combined_encoding_power_all_mlmr_input.csv")

# MTL cortical (BLA, HPC, EC, PRC) - region_by_band
message("Encoding Power - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_mtl_rbband"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# MTL cortical - band_by_region
message("Encoding Power - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_mtl_bbreg"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# HPC subfields (BLA, CA, DG) - region_by_band
message("Encoding Power - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_hpc_rbband"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# HPC subfields - band_by_region
message("Encoding Power - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = enc_power_csv,
  output_dir = file.path("outputs", "stats", "enc_power_hpc_bbreg"),
  phase_name = "Encoding",
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

cat("\n========== RETRIEVAL POWER ==========\n")

ret_power_csv <- file.path("outputs", "csvs", "combined_retrieval_power_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Retrieval Power - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_mtl_rbband"),
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# MTL cortical - band_by_region
message("Retrieval Power - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_mtl_bbreg"),
  region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
)

# HPC subfields - region_by_band
message("Retrieval Power - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_hpc_rbband"),
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# HPC subfields - band_by_region
message("Retrieval Power - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Power", input_csv = ret_power_csv,
  output_dir = file.path("outputs", "stats", "ret_power_hpc_bbreg"),
  region_filter = function(d) filter_no_phg(filter_hpc_subfields(d))
)

# ====================== COHERENCE analyses (encoding + retrieval)==============
cat("\n========== ENCODING COHERENCE ==========\n")

enc_coh_csv <- file.path("outputs", "csvs", "combined_encoding_coherence_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Encoding Coherence - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_mtl_rbband"),
  phase_name = "Encoding",
  region_filter = filter_coherence_mtl_cortical
)

# MTL cortical - band_by_region
message("Encoding Coherence - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_mtl_bbreg"),
  phase_name = "Encoding",
  region_filter = filter_coherence_mtl_cortical
)

# HPC subfields - region_by_band
message("Encoding Coherence - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_hpc_rbband"),
  phase_name = "Encoding",
  region_filter = filter_coherence_hpc_subfields
)

# HPC subfields - band_by_region
message("Encoding Coherence - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = enc_coh_csv,
  output_dir = file.path("outputs", "stats", "enc_coh_hpc_bbreg"),
  phase_name = "Encoding",
  region_filter = filter_coherence_hpc_subfields
)

cat("\n========== RETRIEVAL COHERENCE ==========\n")

ret_coh_csv <- file.path("outputs", "csvs", "combined_retrieval_coherence_all_mlmr_input.csv")

# MTL cortical - region_by_band
message("Retrieval Coherence - MTL cortical - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_mtl_rbband"),
  region_filter = filter_coherence_mtl_cortical
)

# MTL cortical - band_by_region
message("Retrieval Coherence - MTL cortical - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_mtl_bbreg"),
  region_filter = filter_coherence_mtl_cortical
)

# HPC subfields - region_by_band
message("Retrieval Coherence - HPC subfields - region_by_band")
run_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_hpc_rbband"),
  region_filter = filter_coherence_hpc_subfields
)

# HPC subfields - band_by_region
message("Retrieval Coherence - HPC subfields - band_by_region")
run_band_region_mlmr_analysis(
  measure_name = "Coherence", input_csv = ret_coh_csv,
  output_dir = file.path("outputs", "stats", "ret_coh_hpc_bbreg"),
  region_filter = filter_coherence_hpc_subfields
)

# ====================== PAC analyses (encoding + retrieval)==============
cat("\n========== ENCODING PAC ==========\n")

enc_pac_csv <- file.path("outputs", "csvs", "combined_encoding_pac_all_mlmr_input.csv")

# Region-by-band (one model per region pair, PAC bands as predictors)
message("Encoding PAC - region_by_band")
run_pac_region_by_band(
  input_csv = enc_pac_csv,
  output_dir = file.path("outputs", "stats", "enc_pac_rbband"),
  phase = "encoding"
)

# Band-by-region (one model per band, region as predictor)
message("Encoding PAC - band_by_region")
run_pac_band_by_region(
  input_csv = enc_pac_csv,
  output_dir = file.path("outputs", "stats", "enc_pac_bbreg"),
  phase = "encoding"
)

cat("\n========== RETRIEVAL PAC ==========\n")

ret_pac_csv <- file.path("outputs", "csvs", "combined_retrieval_pac_all_mlmr_input.csv")

# Region-by-band
message("Retrieval PAC - region_by_band")
run_pac_region_by_band(
  input_csv = ret_pac_csv,
  output_dir = file.path("outputs", "stats", "ret_pac_rbband"),
  phase = "retrieval"
)

# Band-by-region
message("Retrieval PAC - band_by_region")
run_pac_band_by_region(
  input_csv = ret_pac_csv,
  output_dir = file.path("outputs", "stats", "ret_pac_bbreg"),
  phase = "retrieval"
)

cat("\n========== ALL ANALYSES COMPLETE ==========\n")

###################################################### Test MLM in Bands Not Complete ----
dataFileName = "combined_retrieval_power_all_mlmr_input.csv"
dat = read.csv(paste(RD,dataFileName,sep = ""))

ret_power_csv <- dat

# MTL cortical - region_by_band
message("Retrieval Power - MTL cortical - region_by_band")
measure_name = "Power"
input_csv = paste(RD,dataFileName,sep = "")
output_dir = RD
region_filter = function(d) filter_no_phg(filter_mtl_cortical(d))
ensure_dir(output_dir)

datmodel <- prepare_model_data(input_csv)



###################################################### PowerBand Bar Graphs ----
dataFileName = "combined_retrieval_power_all_mlmr_input.csv"
dat = read.csv(paste(RD,dataFileName,sep = ""))
dat$Memory = ifelse(dat$yes_or_no=="yes","Remembered","Forgotten")
dat$Accuracy = ifelse(dat$yes_or_no=="yes","1","0")
dat$StimCond = dat$trial_type

dat <- dat %>%  group_by(Measure, Patient, Region) %>%
  mutate(trial_idx = row_number()) %>%
  ungroup()



datL = pivot_longer(data = dat,
                    cols = names(dat)[grepl(pattern = "_Freq_",x =  names(dat))],
                    names_to = "Freqs",
                    values_to = "value")
datL$Freqs = gsub(pattern = "Freq_",replacement = "", x=datL$Freqs)
datL$ValueType = gsub(pattern = "_[0-9.]+",replacement = "", x = datL$Freqs)
datL$Freqs = as.numeric(gsub(pattern = "[^0-9.]",replacement = "", x=datL$Freqs))
datL = datL[order(datL$Freqs,datL$Patient,datL$Region,datL$trial_type), ]

valueType = "diff"
regionName = c("EC","BLA","PRC")
stimCondition = c("nostim","stim") 
datL = datL[datL$ValueType %in% valueType & 
              datL$Region %in% regionName &
              datL$StimCond %in%  stimCondition,]


theta_band_range <- c(4, 8)
slow_gamma_band_range <- c(30, 54.9999)
fast_gamma_band_range <- c(55, 100)

datBand <- datL %>%
  group_by(Measure, Patient, Region, StimCond, trial_idx, Accuracy, Memory) %>%
  summarise(
    theta = mean(value[Freqs >= theta_band_range[1] & Freqs <= theta_band_range[2]], na.rm = TRUE),
    slow_gamma = mean(value[Freqs >= slow_gamma_band_range[1] & Freqs <= slow_gamma_band_range[2]], na.rm = TRUE),
    fast_gamma = mean(value[Freqs >= fast_gamma_band_range[1] & Freqs <= fast_gamma_band_range[2]], na.rm = TRUE),
    N=n(),
    .groups = "drop") 
# %>%
#   mutate(
#     StimCond = factor(StimCond, levels = c("nostim", "stim")),
#     theta_c = theta - mean(theta, na.rm = TRUE),
#     slow_gamma_c = slow_gamma - mean(slow_gamma, na.rm = TRUE),
#     fast_gamma_c = fast_gamma - mean(fast_gamma, na.rm = TRUE)
#   )


datAvgBand = datBand%>% group_by(Measure, Patient, Region, StimCond, Accuracy, Memory) %>%
  summarise(theta = mean(theta,na.rm = T),
            slow_gamma = mean(slow_gamma,na.rm = T),
            fast_gamma = mean(fast_gamma,na.rm = T),
            N=n()) %>% as.data.frame()

datP = pivot_longer(data = datAvgBand,
                    cols = c("theta","slow_gamma","fast_gamma"),
                    names_to = "Band",
                    values_to = "value")

ggplot(datP, aes(x=Region, y=value, fill=StimCond)) +
  # geom_violin() +
  # geom_point(color="black", size=1, alpha=0.9)+
  geom_bar(stat="summary",fun="mean",position="dodge")+
  stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
  facet_wrap(Memory~Band)
ggplot(datP, aes(x=Region, y=value, fill=Region)) +
  geom_violin() +
  geom_point(color="black", size=1, alpha=0.9)+
  # geom_bar(stat="summary",fun="mean",position="dodge")+
  # stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
  facet_wrap(Memory~Band+StimCond)
###################################################### Power EC ----
dataFileName = "combined_retrieval_power_all_mlmr_input.csv"
dat = read.csv(paste(RD,dataFileName,sep = ""))
dat$Memory = ifelse(dat$yes_or_no=="yes","Remembered","Forgotten")
dat$Acc = ifelse(dat$yes_or_no=="yes","1","0")
dat$Stim = dat$trial_type

valueType = "diff"
regionName = c("EC","BLA","PRC")
stimCondition = c("nostim") 


datL = pivot_longer(data = dat,
                    cols = names(dat)[grepl(pattern = "_Freq_",x =  names(dat))],
                    names_to = "Freqs",
                    values_to = "value")
datL$Freqs = gsub(pattern = "Freq_",replacement = "", x=datL$Freqs)
datL$ValueType = gsub(pattern = "_[0-9.]+",replacement = "", x = datL$Freqs)
datL$Freqs = as.numeric(gsub(pattern = "[^0-9.]",replacement = "", x=datL$Freqs))
datL = datL[order(datL$Freqs,datL$Patient,datL$Region,datL$trial_type), ]


datL = datL[datL$ValueType %in% valueType & 
            datL$Region %in% regionName &
            datL$Stim %in%  stimCondition,]


names(datL)
datAvg = datL%>% group_by(Measure,Patient,Region,trial_type,Memory, Acc, Stim, Freqs,ValueType) %>%
  summarise(value = mean(value,na.rm = T),
            N=n()) %>% as.data.frame()

datAvg$Tests = paste(datAvg$Measure,datAvg$ValueType,datAvg$Region,datAvg$Stim,sep = "_")
datAvg$trial = 1
names(datAvg)
datP = PermutationTestDataPrep_HMLET(data = datAvg, ID = "Patient", trial = NULL, 
                                    timePoint = "Freqs",
                                    condition = "Memory", conditionLevels = c("Remembered","Forgotten"),
                                    gazeMeasure = "value", testName = "Tests")


Res = ClusterStats_HMLET(datP, paired = T, detailed = F, threshold_t = stats::qt(p=1-.05, df=length(unique(datP$ID))-1))
p = PlotTimeSeries_HMLET(datP,showDataPointProp = F, clusterData = Res, 
                         showOverallMean = "Point",
                         tickSizeOverallMean = 5,
                         shapeCodeOverallMean = 20)+
  facet_wrap(~testName,nrow = 2)
plot(p)

p = PlotTimeSeries_HMLET(datP,showDataPointProp = F, 
                         showOverallMean = "Line")+
  facet_wrap(~testName,nrow = 2)
plot(p)

set.seed(100)
samples = 1000
resPerm = PermutationTest_HMLET(datP, samples = samples, paired = T, threshold_t = stats::qt(p=1-.05, df=-1),
                                permuteTrialsWithinSubject = F)
resPerm[[1]]

A = AddClusterInfotoData_HMLET(resGP$filteredData,resGP$clusterStat)
A$timeBinName = NA
A$MeasureName = "Power"
A$value = A$AOI
A = A[,c("testName", "ID", "timePoint", "condition", "value", "timeBinName",
         "clusterDirection", "clusterIndex", "clusterTime",
         "clusterSig", "clusterpValue", "MeasureName")]
names(A) =c("testName", "ID", "frequency", "condition", "value", "freqName",
         "clusterDirection", "clusterIndex", "clusterRange",
         "clusterSig", "clusterpValue", "MeasureName")


PlotNullDistribution_HMLET(resPerm)
p = PlotTimeSeries_HMLET(resPerm, showDataPointProp =F,
                         showOverallMean = "Point",
                         tickSizeOverallMean = 5,
                         shapeCodeOverallMean = 20,
                         gazePropRibbonAlpha = 0.2,
                         clusterFillAlpha  = 0.7,
                         alphaOverallMean = .7,
                         yLabel = "Power (dB)",
                         pointSize = 1) 
p = p + 
  facet_wrap(~testName,nrow = 2)+
  scale_x_continuous(
    breaks = c(seq(0, 100, by = 5),105),
    limits = c(0, 110),
    expand = c(0, 0)
  )+
  scale_fill_manual(values=c("#7b2d8e","#daa520","#7CAE00","#66CC99","#CC6666", "#9999CC", "#66CC99"))+
  scale_color_manual(values=c("#7b2d8e","#daa520","#7CAE00","#66CC99","#CC6666", "#9999CC", "#66CC99"))
plot(p)





###################################################### Power Several Regions and conditions ----
dataFileName = "combined_retrieval_power_all_mlmr_input.csv"
dat = read.csv(paste(RD,dataFileName,sep = ""))
dat$Memory = ifelse(dat$yes_or_no=="yes","Remembered","Forgotten")
dat$Acc = ifelse(dat$yes_or_no=="yes","1","0")
dat$Stim = dat$trial_type



unique(dat$Region)
regionName = c("BLA","CA", "EC","HPC","PRC", "DG","PHG")
valueType = "diff"
stimCondition = c("stim","nostim") 


datL = pivot_longer(data = dat,
                    cols = names(dat)[grepl(pattern = "_Freq_",x =  names(dat))],
                    names_to = "Freqs",
                    values_to = "value")
datL$Freqs = gsub(pattern = "Freq_",replacement = "", x=datL$Freqs)
datL$ValueType = gsub(pattern = "_[0-9.]+",replacement = "", x = datL$Freqs)
datL$Freqs = as.numeric(gsub(pattern = "[^0-9.]",replacement = "", x=datL$Freqs))
datL = datL[order(datL$Freqs,datL$Patient,datL$Region,datL$trial_type), ]


datL = datL[datL$ValueType %in% valueType & 
              datL$Region %in% regionName &
              datL$Stim %in%  stimCondition,]


names(datL)
datAvg = datL%>% group_by(Measure,Patient,Region,trial_type,Memory, Acc, Stim, Freqs,ValueType) %>%
  summarise(value = mean(value,na.rm = T),
            N=n()) %>% as.data.frame()

datAvg$Tests = paste(datAvg$Measure,datAvg$ValueType,datAvg$Region,datAvg$Stim,sep = "_")
datAvg$trial = 1
names(datAvg)
datP = PermutationTestDataPrep_HMLET(data = datAvg, ID = "Patient", trial = NULL, 
                                     timePoint = "Freqs",
                                     condition = "Memory", conditionLevels = c("Remembered","Forgotten"),
                                     gazeMeasure = "value", testName = "Tests")


Res = ClusterStats_HMLET(datP, paired = T, detailed = F, threshold_t = stats::qt(p=1-.05, df=length(unique(datAvg$Patient))-1))
p = PlotTimeSeries_HMLET(datP,showDataPointProp = F, clusterData = Res, 
                         showOverallMean = "Point",
                         tickSizeOverallMean = 5,
                         shapeCodeOverallMean = 20)+
  facet_wrap(~testName,nrow = 4)
plot(p)

p = PlotTimeSeries_HMLET(datP,showDataPointProp = F, 
                         showOverallMean = "Line")+
  facet_wrap(~testName,nrow = 2)
plot(p)

set.seed(100)
samples = 1000
resPerm = PermutationTest_HMLET(datP, samples = samples, paired = T, threshold_t = stats::qt(p=1-.05, df=length(unique(datP$ID))-1),
                                permuteTrialsWithinSubject = F)
resPerm[[1]]

A = AddClusterInfotoData_HMLET(resGP$filteredData,resGP$clusterStat)
A$timeBinName = NA
A$MeasureName = "Power"
A$value = A$AOI
A = A[,c("testName", "ID", "timePoint", "condition", "value", "timeBinName",
         "clusterDirection", "clusterIndex", "clusterTime",
         "clusterSig", "clusterpValue", "MeasureName")]
names(A) =c("testName", "ID", "frequency", "condition", "value", "freqName",
            "clusterDirection", "clusterIndex", "clusterRange",
            "clusterSig", "clusterpValue", "MeasureName")


PlotNullDistribution_HMLET(resPerm)
p = PlotTimeSeries_HMLET(resPerm, showDataPointProp =F,
                         showOverallMean = "Point",
                         tickSizeOverallMean = 5,
                         shapeCodeOverallMean = 20,
                         gazePropRibbonAlpha = 0.2,
                         clusterFillAlpha  = 0.7,
                         alphaOverallMean = .7,
                         yLabel = "Power (dB)",
                         pointSize = 1) 
p = p + 
  facet_wrap(~testName,nrow = 2)+
  scale_x_continuous(
    breaks = c(seq(0, 100, by = 5),105),
    limits = c(0, 110),
    expand = c(0, 0)
  )+
  scale_fill_manual(values=c("#7b2d8e","#daa520","#7CAE00","#66CC99","#CC6666", "#9999CC", "#66CC99"))+
  scale_color_manual(values=c("#7b2d8e","#daa520","#7CAE00","#66CC99","#CC6666", "#9999CC", "#66CC99"))
plot(p)



