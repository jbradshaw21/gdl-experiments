Data <- Well

experiment <- 'compound_vs_dmso'
test <- 'calc_means'

dmso_well_list <- list(' C15', ' D15', ' E15', ' F15', ' G15', ' H15', ' I15',
                       ' J15', ' K15', ' L15', ' M15', ' N15')



om_well_list <- list(' C16', ' D16', ' E16', ' F16', ' G16', ' H16', ' I16',
                     ' J16', ' K16', ' L16', ' M16', ' N16')

high_concen_well_list <- list(' C03', ' D03', ' E03', ' F03', ' G03', ' H03',
                              ' I03', ' J03', ' K03', ' L03', ' M03', ' N03',
                              ' C04', ' D04', ' E04', ' F04', ' G04', ' H04',
                              ' I04', ' J04', ' K04', ' L04', ' M04', ' N04')

low_concen_well_list <- list(' C13', ' D13', ' E13', ' F13', ' G13', ' H13',
                             ' I13', ' J13', ' K13', ' L13', ' M13', ' N13', 
                             ' C14', ' D14', ' E14', ' F14', ' G14', ' H14',
                             ' I14', ' J14', ' K14', ' L14', ' M14', ' N14')

by_concen_1 <- list(' C14', ' D14', ' E14', ' F14', ' G14', ' H14', ' I14',
                    ' J14', ' K14', ' L14', ' M14', ' N14')
by_concen_2 <- list(' C13', ' D13', ' E13', ' F13', ' G13', ' H13', ' I13',
                    ' J13', ' K13', ' L13', ' M13', ' N13')
by_concen_3 <- list(' C12', ' D12', ' E12', ' F12', ' G12', ' H12', ' I12',
                    ' J12', ' K12', ' L12', ' M12', ' N12')
by_concen_4 <- list(' C11', ' D11', ' E11', ' F11', ' G11', ' H11', ' I11',
                    ' J11', ' K11', ' L11', ' M11', ' N11')
by_concen_5 <- list(' C10', ' D10', ' E10', ' F10', ' G10', ' H10', ' I10',
                    ' J10', ' K10', ' L10', ' M10', ' N10')
by_concen_6 <- list(' C09', ' D09', ' E09', ' F09', ' G09', ' H09', ' I09',
                    ' J09', ' K09', ' L09', ' M09', ' N09')
by_concen_7 <- list(' C08', ' D08', ' E08', ' F08', ' G08', ' H08', ' I08',
                    ' J08', ' K08', ' L08', ' M08', ' N08')
by_concen_8 <- list(' C07', ' D07', ' E07', ' F07', ' G07', ' H07', ' I07',
                    ' J07', ' K07', ' L07', ' M07', ' N07')
by_concen_9 <- list(' C06', ' D06', ' E06', ' F06', ' G06', ' H06', ' I06',
                    ' J06', ' K06', ' L06', ' M06', ' N06')
by_concen_10 <- list(' C05', ' D05', ' E05', ' F05', ' G05', ' H05', ' I05',
                     ' J05', ' K05', ' L05', ' M05', ' N05')
by_concen_11 <- list(' C04', ' D04', ' E04', ' F04', ' G04', ' H04', ' I04',
                     ' J04', ' K04', ' L04', ' M04', ' N04')
by_concen_12 <- list(' C03', ' D03', ' E03', ' F03', ' G03', ' H03', ' I03',
                     ' J03', ' K03', ' L03', ' M03', ' N03')

by_concen_1_4184 <- list(' C14', ' D14', ' E14', ' F14')
by_concen_2_4184 <- list(' C13', ' D13', ' E13', ' F13')
by_concen_3_4184 <- list(' C12', ' D12', ' E12', ' F12')
by_concen_4_4184 <- list(' C11', ' D11', ' E11', ' F11')
by_concen_5_4184 <- list(' C10', ' D10', ' E10', ' F10')
by_concen_6_4184 <- list(' C09', ' D09', ' E09', ' F09')
by_concen_7_4184 <- list(' C08', ' D08', ' E08', ' F08')
by_concen_8_4184 <- list(' C07', ' D07', ' E07', ' F07')
by_concen_9_4184 <- list(' C06', ' D06', ' E06', ' F06')
by_concen_10_4184 <- list(' C05', ' D05', ' E05', ' F05')
by_concen_11_4184 <- list(' C04', ' D04', ' E04', ' F04')
by_concen_12_4184 <- list(' C03', ' D03', ' E03', ' F03')

by_concen_1_4951 <- list(' G14', ' H14', ' I14', ' J14')
by_concen_2_4951 <- list(' G13', ' H13', ' I13', ' J13')
by_concen_3_4951 <- list(' G12', ' H12', ' I12', ' J12')
by_concen_4_4951 <- list(' G11', ' H11', ' I11', ' J11')
by_concen_5_4951 <- list(' G10', ' H10', ' I10', ' J10')
by_concen_6_4951 <- list(' G09', ' H09', ' I09', ' J09')
by_concen_7_4951 <- list(' G08', ' H08', ' I08', ' J08')
by_concen_8_4951 <- list(' G07', ' H07', ' I07', ' J07')
by_concen_9_4951 <- list(' G06', ' H06', ' I06', ' J06')
by_concen_10_4951 <- list(' G05', ' H05', ' I05', ' J05')
by_concen_11_4951 <- list(' G04', ' H04', ' I04', ' J04')
by_concen_12_4951 <- list(' G03', ' H03', ' I03', ' J03')

by_concen_1_1854 <- list(' K14', ' L14', ' M14', ' N14')
by_concen_2_1854 <- list(' K13', ' L13', ' M13', ' N13')
by_concen_3_1854 <- list(' K12', ' L12', ' M12', ' N12')
by_concen_4_1854 <- list(' K11', ' L11', ' M11', ' N11')
by_concen_5_1854 <- list(' K10', ' L10', ' M10', ' N10')
by_concen_6_1854 <- list(' K09', ' L09', ' M09', ' N09')
by_concen_7_1854 <- list(' K08', ' L08', ' M08', ' N08')
by_concen_8_1854 <- list(' K07', ' L07', ' M07', ' N07')
by_concen_9_1854 <- list(' K06', ' L06', ' M06', ' N06')
by_concen_10_1854 <- list(' K05', ' L05', ' M05', ' N05')
by_concen_11_1854 <- list(' K04', ' L04', ' M04', ' N04')
by_concen_12_1854 <- list(' K03', ' L03', ' M03', ' N03')

well_group_1 = list()
well_group_2_list = list()
if (experiment == 'om_vs_dmso'){
  well_group_1 <- dmso_well_list
  well_group_2_list <- list(om_well_list)
} else if (experiment == 'high_concen_vs_low_concen'){
  well_group_1 <- high_concen_well_list
  well_group_2_list <- list(low_concen_well_list)
} else if (experiment == 'compound_vs_dmso'){
  well_group_1 <- dmso_well_list
  if (test == 't_test'){
    well_group_2_list <- list(by_concen_1, by_concen_2,
                              by_concen_3, by_concen_4,
                              by_concen_5, by_concen_6,
                              by_concen_7, by_concen_8,
                              by_concen_9, by_concen_10,
                              by_concen_11, by_concen_12)
  } else if (test == 'calc_means'){
    well_group_2_list <- list(by_concen_1_4184, by_concen_2_4184,
                              by_concen_3_4184, by_concen_4_4184,
                              by_concen_5_4184, by_concen_6_4184,
                              by_concen_7_4184, by_concen_8_4184,
                              by_concen_9_4184, by_concen_10_4184,
                              by_concen_11_4184, by_concen_12_4184)

    # well_group_2_list <- list(by_concen_1_4951, by_concen_2_4951,
    #                           by_concen_3_4951, by_concen_4_4951,
    #                           by_concen_5_4951, by_concen_6_4951,
    #                           by_concen_7_4951, by_concen_8_4951,
    #                           by_concen_9_4951, by_concen_10_4951,
    #                           by_concen_11_4951, by_concen_12_4951)
    # 
    # well_group_2_list <- list(by_concen_1_1854, by_concen_2_1854,
    #                           by_concen_3_1854, by_concen_4_1854,
    #                           by_concen_5_1854, by_concen_6_1854,
    #                           by_concen_7_1854, by_concen_8_1854,
    #                           by_concen_9_1854, by_concen_10_1854,
    #                           by_concen_11_1854, by_concen_12_1854)
  }
}

vec_len_1 = 0
vec_len_2 = 0
for (well in Data$WellId){
  if (well %in% well_group_1){
    vec_len_1 <- vec_len_1 + 1
  } else if (well %in% well_group_2){
    vec_len_2 <- vec_len_2 + 1
  }
}

features <- c('SelectedObjectCount')


p_values_selected <- vector(, length(well_group_2_list))
mean_of_diffs_selected <- vector(, length(well_group_2_list))

p_values_spot <- vector(, length(well_group_2_list))
mean_of_diffs_spot <- vector(, length(well_group_2_list))

count <- 0
for (well_group_2 in well_group_2_list){
  count <- count + 1
  for (feature in features){
    idx_count_1 <- 1
    idx_count_2 <- 1
    feature_list_1 <- vector(, vec_len_1)
    feature_list_2 <- vector(, vec_len_2)
    
    for (well_idx in seq_along(Data$WellId)){
      well <- Data$WellId[well_idx]
      
      if (well %in% well_group_1){
        feature_list_1[idx_count_1] <- Data[well_idx, feature]
        idx_count_1 <- idx_count_1 + 1
        
      } else if (well %in% well_group_2){
        feature_list_2[idx_count_2] <- Data[well_idx, feature]
        idx_count_2 <- idx_count_2 + 1
      }
    }
    
    if (test == 't_test'){
      
      
      t_test <- t.test(feature_list_1, feature_list_2, paired = TRUE, alternative
                       = 'two.sided')
      
      if (feature == 'SelectedObjectCount'){
        p_values_selected[count] <- t_test$p.value
        mean_of_diffs_selected[count] <- t_test$estimate
      } else if (feature == 'SpotCountCh2'){
        p_values_spot[count] <- t_test$p.value
        mean_of_diffs_spot[count] <- t_test$estimate
      }
        
    } else if (test == 'calc_means'){
        if (feature == 'SelectedObjectCount'){
          mean_of_diffs_selected[count] <- mean(feature_list_2) - mean(feature_list_1)
        } else if (feature == 'SpotCountCh2'){
          mean_of_diffs_spot[count] <- mean(feature_list_2) - mean(feature_list_1)
        }
    }
  }
  }

concens <- c(10 / (3 ** 11), 10 / (3 ** 10), 10 / (3 ** 9), 10 / (3 ** 8),
             10 / (3 ** 7), 10 / (3 ** 6), 10 / (3 ** 5), 10 / (3 ** 4),
             10 / (3 ** 3), 10 / (3 ** 2), 10 / 3, 10)

plot(1, type='n', xlim=c(-10, 5), ylim=c(-700, 0), xlab='Log(concentration)', ylab='Mean diff of SelectedObjectCount')
lines(log(concens), mean_of_diffs_selected, type='b', col='blue')
legend("bottomright", legend = paste("Compound", c(4184, 4951, 1854)), col = c('blue', 'red', 'green'), pch = 19, bty = "n")
# note: we must assume normally distributed data

