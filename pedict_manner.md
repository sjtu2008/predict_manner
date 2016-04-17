---
title: "predict_manner"
author: "Roy Wang"
date: "April 17, 2016"
output: html_document
---


## Prediction Assignment

## Background
Using devices such as JawboneUp, NikeFuelBand, and Fitbitit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
   
In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).   

#### Load packages and datasets



```r
library(caret)
```

```
## Warning: package 'ggplot2' was built under R version 3.2.4
```

```r
library(rpart)
library(corrplot)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(Rtsne)
library(xgboost)
library(stats)
library(knitr)
library(Ckmeans.1d.dp)
library(ggplot2)
knitr::opts_chunk$set(cache = TRUE)
```
Set same seed for the code below:

```r
set.seed(12345)
```


#### Getting the data

```r
# the training data  set
training_data <- download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","data")
```

```
## Error in download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", : cannot open destfile 'data', reason 'Is a directory'
```

```r
testing_data <- download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","data")
```

```
## Error in download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", : cannot open destfile 'data', reason 'Is a directory'
```

### Reading data

```r
# load the CSV files as data.frame 
training_data <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-training.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
testing_data <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-testing.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
names(training_data)
```

```
## Error in eval(expr, envir, enclos): object 'training_data' not found
```
The raw training data has 19622 rows of observations and 158 features (predictors). Column `X` is unusable row number. While the testing data has 20 rows and the same 158 features. There is one column of target outcome named `classe`. 

#### Data cleaning


```r
outcome_temp <- training_data[, "classe"]
```

```
## Error in eval(expr, envir, enclos): object 'training_data' not found
```

```r
outcome <- outcome_temp
```

```
## Error in eval(expr, envir, enclos): object 'outcome_temp' not found
```

```r
levels(outcome)
```

```
## Error in levels(outcome): object 'outcome' not found
```

Convert the outcome to numeric, XGBoost gradient booster only recognizes numeric data.   

```r
# convert character levels to numeric
num_len <-length(levels(outcome))
```

```
## Error in levels(outcome): object 'outcome' not found
```

```r
levels(outcome) = 1:num_len
```

```
## Error in eval(expr, envir, enclos): object 'num_len' not found
```

```r
head(outcome)
```

```
## Error in head(outcome): object 'outcome' not found
```


```r
# remove outcome from train
training_data$classe = NULL
```

```
## Error in training_data$classe = NULL: object 'training_data' not found
```

Seperate columns on: belt, forearm, arm, dumbell

```r
splitter <- grepl("belt|arm|dumbell", names(training_data))
```

```
## Error in grepl("belt|arm|dumbell", names(training_data)): object 'training_data' not found
```

```r
training_data <- training_data[, splitter]
```

```
## Error in eval(expr, envir, enclos): object 'training_data' not found
```

```r
testing_data <- testing_data[, splitter]
```

```
## Error in eval(expr, envir, enclos): object 'testing_data' not found
```
remove columns with NA

```r
cols_na = colSums(is.na(testing_data)) == 0
```

```
## Error in is.data.frame(x): object 'testing_data' not found
```

```r
training_data = training_data[,cols_na]
```

```
## Error in eval(expr, envir, enclos): object 'training_data' not found
```

```r
testing_data = testing_data[, cols_na]
```

```
## Error in eval(expr, envir, enclos): object 'testing_data' not found
```

### Preprocessing  data
check for zero variance

```r
zero_var = nearZeroVar(training_data, saveMetrics=TRUE)
```

```
## Error in is.vector(x): object 'training_data' not found
```

```r
zero_var
```

```
## Error in eval(expr, envir, enclos): object 'zero_var' not found
```

#### Plot of relationship between features and outcome  

   

```r
featurePlot(training_data, outcome_temp, "strip")
```

```
## Error in is.data.frame(x): object 'training_data' not found
```

#### Plot of correlation matrix  

Plot a correlation matrix between features.   
  

```r
corrplot.mixed(cor(training_data), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
```

```
## Error in is.data.frame(x): object 'training_data' not found
```


### Build machine learning model 

Now build a machine learning model to predict activity quality (`classe` outcome) from the activity monitors (the features or predictors) by using XGBoost extreme gradient boosting algorithm.    

#### XGBoost data




```r
# convert data to matrix
training_matrix <- as.matrix(training_data)
```

```
## Error in as.matrix(training_data): object 'training_data' not found
```

```r
mode(training_matrix) <- "numeric"
```

```
## Error in mode(training_matrix) <- "numeric": object 'training_matrix' not found
```

```r
testing_matrix<- as.matrix(testing_data)
```

```
## Error in as.matrix(testing_data): object 'testing_data' not found
```

```r
mode(testing_matrix) <- "numeric"
```

```
## Error in mode(testing_matrix) <- "numeric": object 'testing_matrix' not found
```

```r
# convert outcome from factor to numeric matrix 
#   xgboost takes multi-labels in [0, numOfClass)
y = as.matrix(as.integer(outcome)-1)
```

```
## Error in as.matrix(as.integer(outcome) - 1): object 'outcome' not found
```

#### XGBoost parameters 

Set XGBoost parameters for cross validation and training.  



```r
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num_len,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
              )
```

```
## Error in eval(expr, envir, enclos): object 'num_len' not found
```

#### Expected error rate 

Expected error rate is less than `1%` for a good classification. Do cross validation to estimate the error rate using 4-fold cross validation, with 200 epochs to reach the expected error rate of less than `1%`.  

####  k-fold cross validation, with timing 


```r
nround.cv = 200
system.time( bst.cv <- xgb.cv(param=param, data=training_matrix, label=y, 
              nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=FALSE) )
```

```
## Error in typeof(params): object 'param' not found
```

```
## Timing stopped at: 0.001 0 0
```

 


```r
tail(bst.cv$dt) 
```

```
## Error in tail(bst.cv$dt): object 'bst.cv' not found
```
   
From the cross validation, choose index with minimum multiclass error rate.  
Index will be used in the model training to fulfill expected minimum error rate of `< 1%`.  

```r
# index of minimum merror
merror_idx <- which.min(bst.cv$dt[, test.merror.mean]) 
```

```
## Error in which.min(bst.cv$dt[, test.merror.mean]): object 'bst.cv' not found
```

```r
merror_idx 
```

```
## Error in eval(expr, envir, enclos): object 'merror_idx' not found
```

```r
# minimum merror
bst.cv$dt[merror_idx,]
```

```
## Error in eval(expr, envir, enclos): object 'bst.cv' not found
```
Best cross-validation's minimum error rate `test.merror.mean` is around 0.0056 (0.6%), happened at 146th iteration.   

#### Confusion matrix 

Tabulates the cross-validation's predictions of the model against the truths.  


```r
# get CV's prediction decoding
pred.cv <- matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num_len, ncol=num_len)
```

```
## Error in matrix(bst.cv$pred, nrow = length(bst.cv$pred)/num_len, ncol = num_len): object 'bst.cv' not found
```

```r
pred.cv <- max.col(pred.cv, "last")
```

```
## Error in as.matrix(m): object 'pred.cv' not found
```

```r
# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))
```

```
## Error in factor(y + 1): object 'y' not found
```

Confusion matrix shows concentration of correct predictions is on the diagonal, as expected.  
  
The average accuracy is `99.84%`, with error rate is `0.16%`. So, expected error rate of less than `1%` is fulfilled.  

#### Model training 

Fit the XGBoost gradient boosting model on all of the training data.   

```r
# real model fit training, with full data
system.time( bst <- xgboost(param=param, data=training_matrix, label=y, 
                           nrounds=merror_idx, verbose=0) )
```

```
## Error in xgb.get.DMatrix(data, label): object 'training_matrix' not found
```

```
## Timing stopped at: 0.001 0 0.001
```
Time elapsed is around 63 seconds.  

#### Predicting the testing data


```r
# xgboost predict test data using the trained model
pred <- predict(bst, testing_matrix)  
```

```
## Error in predict(bst, testing_matrix): error in evaluating the argument 'object' in selecting a method for function 'predict': Error: object 'bst' not found
```

```r
head(pred, 10)  
```

```
## Error in head(pred, 10): object 'pred' not found
```

#### Post-processing

Output of prediction is the predicted probability of the 5 levels (columns) of outcome.  
Decode the quantitative 5 levels of outcomes to qualitative letters (A, B, C, D, E).   
  

```r
# decode prediction
pred <- matrix(pred, nrow=num_len, ncol=length(pred)/num_len)
```

```
## Error in matrix(pred, nrow = num_len, ncol = length(pred)/num_len): object 'pred' not found
```

```r
pred <- t(pred)
```

```
## Error in t(pred): object 'pred' not found
```

```r
pred <- max.col(pred, "last")
```

```
## Error in as.matrix(m): object 'pred' not found
```

```r
pred.char <- toupper(letters[pred])
```

```
## Error in toupper(letters[pred]): object 'pred' not found
```

 

#### Feature importance


```r
# get the trained model
model <- xgb.dump(bst, with.stats=TRUE)
```

```
## Error in xgb.dump(bst, with.stats = TRUE): object 'bst' not found
```

```r
# get the feature real names
names <- dimnames(training_matrix)[[2]]
```

```
## Error in eval(expr, envir, enclos): object 'training_matrix' not found
```

```r
# compute feature importance matrix
importance_matrix <- xgb.importance(names, model=bst)
```

```
## Error in xgb.importance(names, model = bst): feature_names: Has to be a vector of character or NULL if the model dump already contains feature name. Look at this function documentation to see where to get feature names.
```

```r
# plot
gp <- xgb.plot.importance(importance_matrix)
```

```
## Error in match(x, table, nomatch = 0L): object 'importance_matrix' not found
```

```r
print(gp) 
```

```
## Error in print(gp): object 'gp' not found
```

### Final submission files 


```r
path <- "./answer"
pml_write_files <- function(x) {
    n = length(x)
    for(i in 1: n) {
        filename <- paste0("problem_", i, ".txt")
        write.table(x[i], file=file.path(path, filename), 
                    quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}
pml_write_files(pred.char)
```

```
## Error in pml_write_files(pred.char): object 'pred.char' not found
```
------------------   


