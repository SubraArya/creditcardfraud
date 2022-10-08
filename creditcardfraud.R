#############################################################################
# title: "HarvardX: PH125.9x Data Science  \n   Choose your own Project"
# author: "Subrahmanyam  Aryasomayajula"
# date: "October 6, 2022"

# Description :
# Credit Card Fraud detection  project is done as a part of Choose your Own (CYO) Data Science Project , HarvardX: PH125.9x.
# In this project we evaluate  K-Nearest Neighbours , Naive Bayes classifying algorithms, Random Forest 
# on the Credit Card Fraud Detection dataset and discover the highest accuracy model between the three models.
# This course also  refers to and draws in on the knowledge developed as a part of 
# "Data Science professional certificate program"  by Harvardx.


#############################################################################


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

download.file("https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/raw/master/creditcard.csv","./creditcard.csv" )

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


## set seed for random sampling 
set.seed(2000)

## Take 20% sample data approximately.
samp <- sample(1:nrow(credit), round(0.2*nrow(credit)))

credit <- credit[samp, ]

nrow(credit)
## 56961 records in sample

index <- createDataPartition(credit$Class, p = 0.75, list = F)

## Training data Set 75% of sample
train <- credit[index, ]

nrow(train)
## 42722  rows

## Test data set 25% of sample
test <- credit[-index, ]
nrow(test)
## 14239  rows

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


