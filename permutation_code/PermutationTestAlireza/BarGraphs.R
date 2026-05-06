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