#############################################################################
# title: "HarvardX: PH125.9x Data Science  \n   Choose your own Project"
# author: "Subrahmanyam  Aryasomayajula"
# date: "October 9, 2022"

# Description :
# Credit Card Fraud detection  project is done as a part of Choose your Own (CYO) Data Science Project , HarvardX: PH125.9x.
# In this project we evaluate  K-Nearest Neighbours , Naive Bayes classifying algorithms, Random Forest 
# on the Credit Card Fraud Detection dataset and discover the highest accuracy model between the three models.
# This course also  refers to and draws in on the knowledge developed as a part of 
# "Data Science professional certificate program"  by Harvardx.


#############################################################################

if(!require(caret))  install.packages("caret")
if(!require(class))  install.packages("class")
if(!require(e1071))  install.packages("e1071")
if(!require(ROCR))  install.packages("ROCR")
if(!require(dplyr))  install.packages("dplyr")

if(!require(ggplot2))  install.packages("ggplot2")
if(!require(readr))  install.packages("readr")
if(!require(pROC))  install.packages("pROC")
if(!require(randomForest))  install.packages("randomForest")
if(!require(corrplot))  install.packages("corrplot")
if(!require(data.table))  install.packages("data.table")
if(!require(rpart))  install.packages("rpart")
if(!require(stringr))  install.packages("stringr")
if(!require(tidyverse))  install.packages("tidyverse")


library(caret)
library(class)
library(e1071)
library(ROCR)
library(dplyr)

library(ggplot2) 
library(readr) 
library(pROC) 
library(randomForest)
library(corrplot) 

library(data.table)
library(rpart)
library(stringr)
library(tidyverse)

dir = getwd()

download.file("https://raw.githubusercontent.com/SubraArya/creditcardfraud/main/creditcard.csv","./creditcard.csv" )

filename = paste(dir , "/creditcard.csv",sep ="")


filename  = str_replace_all(filename , "/","\\\\")
credit <- read.csv(filename)



## column names , types and sample records examined.
str(credit)



## Dimensions of the data frame i.e total number of rows and columns .

dim(credit)

credit$Class <- factor(credit$Class)

## Summary stats  on every columns on the dataframe .
summary(credit)

       


## Top 5 records
head(credit)


# Total records , fradulent records and non fraudulent records
rowsTotal <- nrow(credit)
fraudRowsTotal <- nrow(credit[credit$Class == 1,])
nonFraudRowsTotal <- rowsTotal - fraudRowsTotal
fraudRowsTotal
nonFraudRowsTotal
rowsTotal


## set seed for random sampling 
set.seed(2000)

## Take 20% sample data approximately.
samp <- sample(1:nrow(credit), round(0.2*nrow(credit)))

credit <- credit[samp, ]

nrow(credit)

index <- createDataPartition(credit$Class, p = 0.75, list = F)

## Training data Set 75% of sample
train <- credit[index, ]

nrow(train)

## Test data set 25% of sample
test <- credit[-index, ]
nrow(test)

knn1 <- knn(train = train[,-31], test = test[,-31], cl = train$Class, k = 5)

confusionMatrix(knn1, test$Class, positive = "1")



bayes <- naiveBayes(Class~., data = train, laplace = 1)

pred <- predict(bayes, test)
confusionMatrix(pred, test$Class, positive = "1")



# Build the Model with the Random Forest.


random_model_train <- randomForest(Class ~ ., data = train,ntree = 40)


# From the above train model is used in test data set.


random_pred_test <- predict(random_model_train,test)


confusionMatrix(random_pred_test,test$Class ,positive = "1") 


