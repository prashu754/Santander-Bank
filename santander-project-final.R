rm(list = ls())

#setting the directory
setwd("E:/Analytics/edWisor/Project/Santander/")

#importing the database
SB_db  = read.csv("train.csv")
# cardb_bkup  = read.csv("train_cab.csv")


colnames(SB_db)

##Analysis of Data
#Checking the data types
str(SB_db$target)

SB_db$target = as.factor(SB_db$target)
SB_uni = sapply(SB_db, function(x) length(unique(x)))

SB_uni = sort(SB_uni)

miss_val = apply(SB_db, 2, function(i){sum(is.na(i))})
miss_val = sort(miss_val, decreasing = TRUE)

desc = summary(SB_db[,c(2:202)])

rm(SB_uni, miss_val, desc)

library(ggplot2)
barplot(table(SB_db$target),width = 0.5, main = "Target Dist", xlab = "Target Var", ylab = "Freq.")

# PCA on Scaled data using simple random sampling

# SB_pca = prcomp(SB_train[,3:202], scale. = TRUE, center = TRUE)
# 
# summary(SB_pca)
# 
# sd = SB_pca$sdev
# pc_var = sd^2
# 
# #propotion of variance as seen in Summary of PCA
# prop_var = pc_var/sum(pc_var)
# 
# #Plotting propotion of variance against PC's
# plot(prop_var, xlab = "Principal Component", 
#      ylab = "Proportion of Variance", type = "b")
# 
# #Creating train & test from the data - Normal Sampling
# train_index = sample(1:nrow(SB_pca$x), 0.8 * nrow(SB_pca$x))
# train_pca = SB_pca$x[train_index,]
# test_pca = SB_pca$x[-train_index,]
# 
# #train_index = sample(1:nrow(SB_train), 0.8 * nrow(SB_train))
# train_target = SB_train[train_index, 2]
# test_target = SB_train[-train_index,2]
# 
# train_data_pca = data.frame(target = train_target, train_pca)
# test_data_pca = data.frame(target = test_target, test_pca)
# 
# library(utils)
# memory.limit(size = NA)
# 
# #Logistic Regression
# lr_model = glm(target ~., data = train_data_pca, family = "binomial")
# summary(lr_model)
# lr_pred = predict(lr_model, newdata = data.frame(test_pca), type = "response")
# lr_pred = ifelse(lr_pred > 0.5, 1, 0)
# 
# #Confussion Marrix
# library(caret)
# confusionMatrix(table(test_target, lr_pred))
# 
# #        0     1
# #   0 35480   478
# #   1  2979  1063
# 
# #TN = 35480, FP = 478, FN = 2979, TP = 1063
# 
# rec_SB_pca = recall(table(lr_pred, test_target))


# PCA on Unscaled data using simple random sampling

# Dividing dataset into train & test
# set.seed(123)
# train_index = sample(1:nrow(SB_db), 0.8 * nrow(SB_db))
# SB_train = SB_db[train_index,]
# SB_test = SB_db[-train_index,]
# 
# 
# SB_pca_train = prcomp(SB_train[,3:202], scale. = FALSE, center = TRUE)
# 
# summary(SB_pca_train)
# 
# sd = SB_pca_train$sdev
# pc_var = sd^2
# 
# #propotion of variance as seen in Summary of PCA
# prop_var = pc_var/sum(pc_var)
# 
# #Plotting propotion of variance against PC's
# grid()
# plot(cumsum(prop_var), xlab = "Principal Component", 
#      ylab = "Cummulative Proportion of Variance", type = "b")
# 
# # checking the size of data showing upto 98% variance
# pca_len = length(cumsum(prop_var)[cumsum(prop_var) < 0.98])
# 
# #As per the plot and data 135 variable shows 98.2% variance
# train_data = data.frame(target = SB_train$target, SB_pca_train$x)
# train_data = train_data[,1:(pca_len+1)]
# 
# pca_transform = function(db, n_cols){
#   test = predict(SB_pca_train, newdata = db)
#   test = test[,1:n_cols]
#   return(test)
# }
# 
# test_data = data.frame(pca_transform(SB_test[,3:202], pca_len))
# rm(SB_db, SB_train, a, pc_var, prop_var, sd, train_index)
# 
# set.seed(124)
# library(utils)
# memory.limit(size = NA)
# 
# #Logistic Regression
# lr_model = glm(target ~., data = train_data, family = "binomial")
# summary(lr_model)
# lr_pred = predict(lr_model, newdata = test_data, type = "response")
# lr_pred = ifelse(lr_pred > 0.5, 1, 0)
# 
# #Confussion Marrix
# library(caret)
# confusionMatrix(table(SB_test$target, lr_pred))
# 
# #        0     1
# #   0 35734   336
# #   1  3343   587
# 
# #TN = 35734, FP = 336, FN = 3343, TP = 587
# # Accuracy = (TP+TN)*100/(TN+FN) = 90.8%
# # Recall = TP*100/(TP+FN) = 14.93%
# # Precision = TP*100/(TP+FP) = 63.59%
# 
# # rec_SB_pca = recall(table(lr_pred, test_target))


# PCA on Unscaled data using Stratified sampling

SB_pca_train = prcomp(SB_db[,3:202], scale. = FALSE, center = TRUE)

summary(SB_pca_train)

sd = SB_pca_train$sdev
pc_var = sd^2

#propotion of variance as seen in Summary of PCA
prop_var = pc_var/sum(pc_var)

#Plotting propotion of variance against PC's
plot(cumsum(prop_var), xlab = "Principal Component", 
     ylab = "Cummulative Proportion of Variance", type = "b")
grid()

rm(pc_var, prop_var, sd)

#As per the plot and data 150 components shows > 99% variance
SB_train = data.frame(target = SB_db$target, SB_pca_train$x)
SB_train = SB_train[,1:151]

# Dividing dataset into train & test using Stratified sampling
library(sampling)
set.seed(123)
stratum = strata(SB_train, c("target"), size = c(35980, 4020), method = "srswor")
test_data = getdata(SB_train, stratum)
test_index = as.numeric(rownames(test_data))
train_data = SB_train[-test_index,]
test_data = test_data[,1:151]



pca_transform = function(db, n_cols){
  test = predict(SB_pca_train, newdata = db)
  test = test[,1:n_cols]
  return(test)
}

rm(SB_db, SB_train, stratum, test_index)

set.seed(124)
# library(utils)
# memory.limit(size = NA)
# 
# #Logistic Regression
# lr_model = glm(target ~., data = train_data, family = "binomial")
# summary(lr_model)
# lr_pred = predict(lr_model, newdata = test_data, type = "response")
# lr_pred = ifelse(lr_pred > 0.5, 1, 0)
# 
# #Confussion Marrix
# library(caret)
# confusionMatrix(table(SB_test$target, lr_pred))
# 

# Naive bayes
library(e1071)

NB_model = naiveBayes(target ~ ., data = train_data)

NB_pred = predict(NB_model, test_data[,1:150], type = 'class')

library(caret)
confusionMatrix(table(test_data$target, NB_pred))

# NB_pred
#       0     1
# 0 35668   312
# 1  3190   830

# Accuracy = 91%; Recall = 21%, Precision = 73%, Sensitiivity = 99%


# Applying Naive Bayes on Test Data
test_db = read.csv("test.csv")
test_db_pca = data.frame(pca_transform(test_db[,2:201], 150))
test_pred = predict(NB_model, newdata = test_db_pca, type = 'class')
test_db$target = test_pred
write.csv(test_db, file = "Test_output_R.csv")
