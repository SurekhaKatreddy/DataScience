#********************************************************************************
#********************************Load the  data**********************************
#********************************************************************************
library("readxl")
heart_failures_df  <- read.csv("heart_failure_clinical_records_dataset.csv")
heart_failures_df

#********************************************************************************
#**************************Handling the missing data*****************************
#********************************************************************************
#No missing data
sum(is.na(heart_failures_df))

#Identifying columns which have missing values
names(which(colSums(is.na(heart_failures_df)) > 0) )

#********************************************************************************
#**************************Identify redundant columns if any*********************
#********************************************************************************
library(caret)
#install.packages("corrplot")
library(corrplot)
heart_failures_cor <- cor(heart_failures_df,method = "spearman")
corrplot(heart_failures_cor)
findCorrelation(heart_failures_cor)
findCorrelation(heart_failures_cor, cutoff = 0.25)
findCorrelation(heart_failures_cor, cutoff = 0.5)
findCorrelation(heart_failures_cor, cutoff = 0.75)

#Correlated columns do exists
findLinearCombos(heart_failures_cor)
#Correlated columns are no exactly the linear combinations 
#So no columns have been removed

#Indicate the categorical variable by converting them to factors
names <- c('DEATH_EVENT' ,'anaemia','diabetes','high_blood_pressure','sex','smoking')
heart_failures_df[,names] <- lapply(heart_failures_df[,names] , factor)
str(heart_failures_df)

#********************************************************************************
#**********************************Visualization *******************************
#********************************************************************************


#*#Count of success stories
SuccessStories = heart_failures_df[heart_failures_df$DEATH_EVENT==0,]
nrow(SuccessStories)
FailureStories = heart_failures_df[heart_failures_df$DEATH_EVENT==1,]
nrow(FailureStories)                    
pie(table(heart_failures_df$DEATH_EVENT), lbls <- c(nrow(SuccessStories),nrow(FailureStories)))

#"age", "creatinine_phosphokinase","ejection_fraction","platelets","serum_sodium"
str(heart_failures_df)

pairs(heart_failures_df[,c(1,3,5,7,8,9)], main="Pair plot of heart failure Dataset", pch=21, bg = c("black", "red", "blue"))
pairs(SuccessStories[,c(1,3,5,7,8,9)], main="Pair plot of no heart failure cases", pch=21, bg = c("black", "red", "blue"))  
save.image("healthy_data_set.jpeg")
pairs(FailureStories[,c(1,3,5,7,8,9)], main="Pair plot of heart failure cases", pch=21, bg = c("black", "red", "blue"))  
save.image("failed_data_set.jpeg")
par(mfrow = c(2, 2))
hist(FailureStories$age, col="maroon", main = "age in failed cases")
hist(SuccessStories$age, col="light green", main = "age in no failure cases")
#failed cases has more proportion of people with anaemia
plot(FailureStories$anaemia,col="maroon",main = "anaemia in failed cases") 
plot(SuccessStories$anaemia,col="light green",main="anaemia in no failure case")  
hist(FailureStories$creatinine_phosphokinase,col="maroon",main = "cp in failure cases") 
hist(SuccessStories$creatinine_phosphokinase,col="light green",main = "cp in no failure cases") 
plot(FailureStories$diabetes,col="maroon",main = "diabetes in failure cases") 
plot(SuccessStories$diabetes,col="light green",main = "diabetes in no failure cases") 
plot(FailureStories$ejection_fraction,col="maroon",main = "ejection_fraction in failure cases") 
plot(SuccessStories$ejection_fraction,col="light green",main = "ejection_fraction in no failure cases") 
plot(FailureStories$high_blood_pressure,col="maroon",main = "High BP in failure cases") 
plot(SuccessStories$high_blood_pressure,col="light green",main = "High BP in no failure cases") 
plot(FailureStories$platelets,col="maroon",main = "platelet Count in failure cases") 
plot(SuccessStories$platelets,col="light green",main = "platelet Count in no failure cases") 
plot(FailureStories$serum_creatinine,col="maroon",main = "sc in failure cases") 
plot(SuccessStories$serum_creatinine,col="light green",main = "sc in no failure cases") 
plot(FailureStories$serum_sodium,col="maroon",main = "serum sodium in failure cases") 
plot(SuccessStories$serum_sodium,col="light green",main = "serum sodium in no failure cases") 
plot(FailureStories$sex,col="maroon",main = "sex in failure cases") 
plot(SuccessStories$sex,col="light green",main = "sex in no failure cases") 
plot(FailureStories$smoking,col="maroon",main = "smoking in failure cases") 
plot(SuccessStories$smoking,col="light green",main = "smoking in no failure cases") 
plot(FailureStories$time,col="maroon",main = "time in failure cases") 
plot(SuccessStories$time,col="light green",main = "time in no failure cases") 

summary(heart_failures_df$age)
summary(SuccessStories$age)
summary(FailureStories$age)

summary(heart_failures_df$anaemia)
summary(SuccessStories$anaemia)
summary(FailureStories$anaemia)

summary(heart_failures_df$creatinine_phosphokinase)
summary(SuccessStories$creatinine_phosphokinase)
summary(FailureStories$creatinine_phosphokinase)

summary(heart_failures_df$diabetes)
summary(SuccessStories$diabetes)
summary(FailureStories$diabetes)

summary(heart_failures_df$ejection_fraction)
summary(SuccessStories$ejection_fraction)
summary(FailureStories$ejection_fraction)

summary(heart_failures_df$high_blood_pressure)
summary(SuccessStories$high_blood_pressure)
summary(FailureStories$high_blood_pressure)

summary(heart_failures_df$platelets)
summary(SuccessStories$platelets)
summary(FailureStories$platelets)

summary(heart_failures_df$serum_creatinine)
summary(SuccessStories$serum_creatinine)
summary(FailureStories$serum_creatinine)

summary(heart_failures_df$serum_sodium)
summary(SuccessStories$serum_sodium)
summary(FailureStories$serum_sodium)

summary(heart_failures_df$sex)
summary(SuccessStories$sex)
summary(FailureStories$sex)

summary(heart_failures_df$smoking)
summary(SuccessStories$smoking)
summary(FailureStories$smoking)

summary(heart_failures_df$time)
summary(SuccessStories$time)
summary(FailureStories$time)
#********************************************************************************
#**********************************Pre-processing *******************************
#********************************************************************************
par(mfrow=c(1,1))
#platelet count values are too high when compared to other fields
boxplot(heart_failures_df[-13], main = 'Boxplot of heart_failure data before scaling',col = "light blue")
heart_failures_df[,c(1,3,5,7,8,9,12)] <- scale(heart_failures_df[,c(1,3,5,7,8,9,12)])
boxplot(heart_failures_df[-13], main = 'Boxplot of heart_failure data after scaling', col = "light blue")  
str(heart_failures_df)

#********************************************************************************
#******Split the data into Test and Train 
#********************************************************************************
# Prep Training and Test data.
library(caret)

#set the seed so the results are comparable in different runs
set.seed(100)
trainDataIndex <- createDataPartition(heart_failures_df$DEATH_EVENT, p=0.7, list = F)  # 70% training data
trainData <- heart_failures_df[trainDataIndex, ]
testData <- heart_failures_df[-trainDataIndex, ]

table(trainData$DEATH_EVENT)
table(testData$DEATH_EVENT)

#********************************************************************************
#*******************Logistic Model on train data set*****************************
#********************************************************************************
# Build Logistic Model
GLM_train_model <- glm(DEATH_EVENT ~ ., family = "binomial", data=trainData)
summary(GLM_train_model)
#AIC : 185.33 in 5 fisher Scoring iterations

max_likelihood_train<- logLik(GLM_train_model)
max_likelihood_train  #log Lik.' -79.66501 (df=13)
#Akaike's Information Criterion (AIC) is -2*log-likelihood+2*k 

#Odds Ratio
exp(coefficients(GLM_train_model))

#The logitmod is now built. You can now use it to predict the response on testData.
#********************************************************************************
#**********Train Accuracy
#********************************************************************************
glm_train_pred <- predict(GLM_train_model, trainData)
y_pred_num <- ifelse(glm_train_pred > 0.5, 1, 0)

y_pred_train <- factor(y_pred_num, levels=c(0, 1))
y_act_train <- trainData$DEATH_EVENT

#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
#train accuracy
mean(y_pred_train == y_act_train)
#Confusion matrix
addmargins(table(y_pred_train, y_act_train))
cm = as.matrix(table(y_pred_train, y_act_train))
cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of predictions per class
colsums = apply(cm, 2, sum) # number of actual per class
p = rowsums / n # distribution of instances over the predicted classes
q = colsums / n # distribution of instances over the actual classes


# alternately
accuracy = sum(diag) / n 
accuracy

#Precision and Recall
precision = diag / rowsums 
recall = diag / colsums 
f1 = 2 * precision * recall / (precision + recall)

data.frame(precision, recall, f1) 

#********************************************************************************
#**********Test Accuracy
#********************************************************************************
glm_test_pred <- predict(GLM_train_model, testData)
y_pred_num <- ifelse(glm_test_pred > 0.5, 1, 0)

y_pred_test <- factor(y_pred_num, levels=c(0, 1))
y_act_test <- testData$DEATH_EVENT

#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
######test accuracy
mean(y_pred_test == y_act_test)
addmargins(table(y_pred_test, y_act_test))
cm = as.matrix(table(y_pred_test, y_act_test))
cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of predictions per class
colsums = apply(cm, 2, sum) # number of actual per class
p = rowsums / n # distribution of instances over the predicted classes
q = colsums / n # distribution of instances over the actual classes


# alternately
accuracy = sum(diag) / n 
accuracy

#Precision and Recall
precision = diag / rowsums 
recall = diag / colsums 
f1 = 2 * precision * recall / (precision + recall)

data.frame(precision, recall, f1) 
#*********************************************************************************************
#*************************************Strong predictors************************
#*********************************************************************************************
###GLM strong predictors
GLM_spec_model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, 
                      family = "binomial", data=trainData)
summary(GLM_spec_model)
#AIC: 178.52 in 5 fisher Scoring iterations

max_likelihood_spec_train<- logLik(GLM_spec_model)
max_likelihood_spec_train

#Odds Ratio
exp(coefficients(GLM_spec_model))

#######train accuracy###########
glm_train_pred <- predict(GLM_spec_model, trainData)
y_pred_num <- ifelse(glm_train_pred > 0.5, 1, 0)
y_pred_test <- factor(y_pred_num, levels=c(0, 1))
y_act_test <- trainData$DEATH_EVENT
#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
#train accuracy
mean(y_pred_test == y_act_test)
#Confusion matrix
addmargins(table(y_pred_test, y_act_test))

###########test accuracy############
glm_test_pred <- predict(GLM_spec_model, testData)
y_pred_num <- ifelse(glm_test_pred > 0.5, 1, 0)
y_pred_test <- factor(y_pred_num, levels=c(0, 1))
y_act_test <- testData$DEATH_EVENT
#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
#test accuracy
mean(y_pred_test == y_act_test)
#Confusion matrix
addmargins(table(y_pred_test, y_act_test))
#*********************************************************************************************
#*********************************************************************************************
#To use sampling we may have to factor the response variable
#trainData$DEATH_EVENT <- factor(trainData$DEATH_EVENT, levels = c(0, 1))

#Now let me do the upsampling using the upSample function.

#It follows a similar syntax as downSample.
library(caret)
'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.
 table(trainData$DEATH_EVENT)

# Up Sample.
set.seed(100)
up_train <- upSample(x = trainData[, colnames(trainData) %ni% "Class"],
                     y = trainData$DEATH_EVENT)

table(up_train$Class)

#********************************************************************************
#**********Logistic Model on up train data set
#********************************************************************************
# Build Logistic Model
GLM_up_train <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time,  family = "binomial", data=up_train)
summary(GLM_up_train)
#AIC : 28 in Fisher Scoring iterations: 25 for all columns
#Odds Ratio
exp(coefficients(GLM_up_train))

max_likelihood_uptrain<- logLik(GLM_up_train)
max_likelihood_uptrain

#######train accuracy###########
glm_train_pred <- predict(GLM_up_train, trainData)
y_pred_num <- ifelse(glm_train_pred > 0.5, 1, 0)
y_pred_test <- factor(y_pred_num, levels=c(0, 1))
y_act_test <- trainData$DEATH_EVENT
#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
#train accuracy
mean(y_pred_test == y_act_test)
#Confusion matrix
addmargins(table(y_pred_test, y_act_test))

###########test accuracy############
glm_test_pred <- predict(GLM_up_train, testData)
y_pred_num <- ifelse(glm_test_pred > 0.5, 1, 0)
y_pred_test <- factor(y_pred_num, levels=c(0, 1))
y_act_test <- testData$DEATH_EVENT
#Let's compute the accuracy, which is nothing but the proportion of y_pred 
#that matches with y_act
#test accuracy
mean(y_pred_test == y_act_test)
#Confusion matrix
addmargins(table(y_pred_test, y_act_test))

#********************************************************************************
#***********************************c5.0 model***********************************
#********************************************************************************
#loading required libraries
library(C50)
library(caret)
set.seed(100)

heart_C5.0_model <- C5.0(DEATH_EVENT ~ ., data=trainData) # train the tree
summary(heart_C5.0_model)# view the model components 

#decision tree
plot(heart_C5.0_model, type = "simple" , cex =0.7 , main = 'heart failure diagnosis decision tree') # view the model graphically 

accuracy_C5.0_model <- predict(heart_C5.0_model, trainData, type = "class")
mean(accuracy_C5.0_model == trainData$DEATH_EVENT)


#making predictions and check the accuracy of train and test data
train_predictions_C5.0_model <- predict(heart_C5.0_model, trainData)#accuracy of train data
mean(train_predictions_C5.0_model==trainData$DEATH_EVENT)# accuracy of train data

test_predictions_C5.0_model <- predict(heart_C5.0_model, testData)#accuracy of test data
mean(test_predictions_C5.0_model == testData$DEATH_EVENT)#accuracy of test data

#Print the rules set
rules_C5.0_model <- C5.0(DEATH_EVENT ~ ., data=trainData, rules = TRUE)
summary(rules_C5.0_model)

#making predictions and check the accuracy of train and test data of rules set
train_predictions_C5.0_model_rules <- predict(rules_C5.0_model, trainData)#accuracy of train data
mean(train_predictions_C5.0_model_rules==trainData$DEATH_EVENT)# accuracy of train data

test_predictions_C5.0_model_rules <- predict(rules_C5.0_model, testData)#accuracy of test data
mean(test_predictions_C5.0_model_rules == testData$DEATH_EVENT)#accuracy of test data

#confusion Matrix of test data
confusionMatrix(table(test_predictions_C5.0_model, testData$DEATH_EVENT))

#train data
table_train_C5.0_model = table(predicted = train_predictions_C5.0_model, actual = trainData$DEATH_EVENT)
cm = as.matrix(table_train_C5.0_model)
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of predictions per class
colsums = apply(cm, 2, sum) # number of instances per class

cm
n
nc
diag
rowsums
colsums

p = rowsums / n # distribution of instances over the predicted classes
q = colsums / n # distribution of instances over the actual classes
p
q

# alternately
accuracy_train = sum(diag) / n 
accuracy_train

#test data
table_test_C5.0_model = table(predicted = test_predictions_C5.0_model, actual = testData$DEATH_EVENT)
cm1 = as.matrix(table_test_C5.0_model)

n1 = sum(cm1) # number of instances
nc1 = nrow(cm1) # number of classes
diag1 = diag(cm1) # number of correctly classified instances per class 
rowsums1 = apply(cm1, 1, sum) # number of predictions per class
colsums1 = apply(cm1, 2, sum) # number of instances per class

cm1
n1
nc1
diag1
rowsums1
colsums1

p1 = rowsums1 / n1 # distribution of instances over the predicted classes
q1 = colsums1 / n1 # distribution of instances over the actual classes
p1
q1

# alternately
accuracy1 = sum(diag1) / n1 
accuracy1



#********************************************************************************
#***********************************CART MODEL***********************************
#********************************************************************************
#CART model
library(rpart)
# grow tree 
heart_CART_model <- rpart(DEATH_EVENT ~ .,data=trainData, method="class")

# Exam the tree
printcp(heart_CART_model) # display the results 
plotcp(heart_CART_model) # visualize cross-validation results 
summary(heart_CART_model) # detailed summary of splits

# plot tree another way
plot(heart_CART_model, uniform=TRUE, main="Classification Tree for Heart failure data")
text(heart_CART_model, use.n=TRUE, all=TRUE, cex=.6)
summary(heart_CART_model)

#make plot
library(RColorBrewer)
library(rattle)
fancyRpartPlot(heart_CART_model, cex = 1.0, main = "Heart failure diagnosis")

library(tidyverse)
library(dplyr)
library(caret)
library(ggplot2)
library(rpart)
#library(pipeR)
library(magrittr)
library(mlbench)
#library(momocs)
library(e1071)
library(rpart.plot)

#create rules
rules_CART_model <- rpart.rules(heart_CART_model, cover = TRUE)
rules_CART_model


#accuracy of the train and test set
# Make predictions on the test data
predicted.classes_CART_model <- predict(heart_CART_model, testData, type = "class")
head(predicted.classes_CART_model)

#Find amount of overfitting
trained.classes_CART_model <- heart_CART_model %>% predict(trainData, type = "class")

# Compute model accuracy rate on test and train data
mean(predicted.classes_CART_model == testData$DEATH_EVENT)# test set
mean(trained.classes_CART_model == trainData$DEATH_EVENT)# train set

#Difference in accuracies = overfitting
mean(predicted.classes_CART_model == testData$DEATH_EVENT) - mean(trained.classes_CART_model == trainData$DEATH_EVENT)


#confusion matrix of test data

confusionMatrix(table(predicted.classes_CART_model, testData$DEATH_EVENT))


y_pred = predicted.classes_CART_model
y_act <- testData$DEATH_EVENT

table(y_pred, y_act)

cm = as.matrix(table(y_pred, y_act))

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of predictions per class
colsums = apply(cm, 2, sum) # number of instances per class

cm
n
nc
diag
rowsums
colsums

p = rowsums / n # distribution of instances over the predicted classes
q = colsums / n # distribution of instances over the actual classes
p
q

# alternately 
accuracy = sum(diag) / n 
accuracy

precision = diag / rowsums 
recall = diag / colsums 
f1 = 2 * precision * recall / (precision + recall)

data.frame(precision, recall, f1) 

macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(macroPrecision, macroRecall, macroF1)


#However, this full tree including all predictor appears to be very complex and 
#can be difficult to interpret in the situation where you have a large data sets with multiple predictors.

#Additionally, it is easy to see that, a fully grown tree will overfit the 
#training data and might lead to poor test set performance.

#confusion matrix of train data
y_pred_train = trained.classes_CART_model
y_act_train <- trainData$DEATH_EVENT

table(y_pred_train, y_act_train)

cm_train = as.matrix(table(y_pred_train, y_act_train))

n_train = sum(cm_train) # number of instances
nc_train = nrow(cm_train) # number of classes
diag_train = diag(cm_train) # number of correctly classified instances per class 
rowsums_train = apply(cm_train, 1, sum) # number of predictions per class
colsums_train = apply(cm_train, 2, sum) # number of instances per class

cm_train
n_train
nc_train
diag_train
rowsums_train
colsums_train

p_train = rowsums_train / n_train # distribution of instances over the predicted classes
q_train = colsums_train / n_train # distribution of instances over the actual classes
p_train
q_train

# alternately 
accuracy_train = sum(diag_train) / n_train 
accuracy_train

#################################################################################################################
#Random forest model

#Random forest
library(randomForest)
str(trainData)

# Create a Random Forest model with default parameters
heart_rf_model1 <- randomForest(DEATH_EVENT~., data = trainData, ntree = 20)
heart_rf_model1

#By default, number of trees is 500 and number of variables tried at each split is
#3 in this case. Error rate is 20.38%.

# Fine tuning parameters of Random Forest model
heart_rf_model2 <- randomForest(DEATH_EVENT ~ ., data = trainData, ntree = 1000, mtry = 6, importance = TRUE)
heart_rf_model2

#When we have increased the mtry to 6 from 3, error rate has reduced from 20.38% to 
#18.01%. We will now predict on the train dataset first and then predict on test dataset.

# Predicting on train set
pred_train <- predict(heart_rf_model2, trainData, type = "class")
# Checking classification accuracy
mean(pred_train == trainData$DEATH_EVENT)
table(pred_train, trainData$DEATH_EVENT)  

# Predicting on test set
pred_test <- predict(heart_rf_model2, testData, type = "class")
# Checking classification accuracy
mean(pred_test == testData$DEATH_EVENT)                    
table(pred_test,testData$DEATH_EVENT)

#confusion matrix of test data

confusionMatrix(table(pred_test, testData$DEATH_EVENT))


#********************************************************************************
#**********************************naive bayes model*****************************
#********************************************************************************
library(e1071)
#library('ElemStatLearn')
library("klaR")
library("caret")
library(magrittr)
library(dplyr)

#Fitting the Naive Bayes model
Naive_Bayes_Model=naiveBayes(DEATH_EVENT ~., data=trainData)
#What does the model say? Print the model summary
Naive_Bayes_Model

#Prediction on the train dataset
NB_Predictions=predict(Naive_Bayes_Model,trainData)
#Confusion matrix to check accuracy
# we can get a popular matrix called confusion matrix via function table to evaluate
#the performance of our prediction
# columns indicate the number of mushrooms in actual type; 
#likewise, rows indicate the number those in predicted type.

table(NB_Predictions,trainData$DEATH_EVENT)
mean(NB_Predictions == trainData$DEATH_EVENT)

#Prediction on the test dataset
NB_Predictions_test=predict(Naive_Bayes_Model,testData)
#Confusion matrix to check accuracy
table(NB_Predictions_test,testData$DEATH_EVENT)
mean(NB_Predictions_test == testData$DEATH_EVENT)

#confusion matrix of test data
confusionMatrix(table(NB_Predictions_test, testData$DEATH_EVENT))

#********************************************************************************
#******Artificial Neural Network Model
#********************************************************************************

library("nnet")
library(NeuralNetTools)
library(reshape)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

set.seed(100)

# Prepare outputs
heart_failures_df$DEATH_EVENT<- factor(heart_failures_df$DEATH_EVENT)
table(heart_failures_df$DEATH_EVENT)

# Pre Process Inputs
heart_failures_pp <- preProcess(trainData[1:12], method = c("range"))
trainData <- cbind(predict(heart_failures_pp, trainData[1:12]),DEATH_EVENT = trainData$DEATH_EVENT)
testData  <- cbind(predict(heart_failures_pp, testData[1:12]),DEATH_EVENT = testData$DEATH_EVENT)
table(trainData$DEATH_EVENT)
table(testData$DEATH_EVENT)

#Training the Neural network model


nnet_heart_failure_model <- nnet(DEATH_EVENT ~ ., data = trainData, size = 10, maxit = 1000)
plot.nnet(nnet_heart_failure_model)

#Checking the accuracy of the Neural network model
nnet_train_predictions <- predict(nnet_heart_failure_model, trainData[1:12], type = "class")
mean(nnet_train_predictions == trainData$DEATH_EVENT)

nnet_test_predictions <- predict(nnet_heart_failure_model, testData[1:12], type = "class")
mean(nnet_test_predictions == testData$DEATH_EVENT)

nnet_heart_failure_model2 <- nnet(DEATH_EVENT ~ ., data = trainData, size = 50, maxit = 10000)
plot.nnet(nnet_heart_failure_model2)
nnet_train_predictions2 <- predict(nnet_heart_failure_model2, trainData[1:12], type = "class")
mean(nnet_train_predictions2 == trainData$DEATH_EVENT)

nnet_test_predictions2 <- predict(nnet_heart_failure_model2, testData[1:12], type = "class")
mean(nnet_test_predictions2 == testData$DEATH_EVENT)

#Applying the weight decay

nnet_heart_failure_model3 <- nnet(DEATH_EVENT ~ ., data = trainData, size = 10, maxit = 1000, decay = 0.01)
plot.nnet(nnet_heart_failure_model3)
nnet_train_predictions3 <- predict(nnet_heart_failure_model3, trainData[1:12], type = "class")
mean(nnet_train_predictions3 == trainData$DEATH_EVENT)

nnet_test_predictions3 <- predict(nnet_heart_failure_model3, testData[1:12], type = "class")
mean(nnet_test_predictions3 == testData$DEATH_EVENT)
nnet_heart_failure_model3
#garson(nnet_heart_failure_model3)
#Confusion matrix to check the accuracy

confusionMatrix(table(predicted = nnet_test_predictions3, actual = testData$DEATH_EVENT))


