Predicting Exercise Styles
==========================



The goal of this project is to use accelerometer data from 6 participants while
they exercies to predict the manner in which they are exercising.

First we load up the data and split it into a training set (75%) and a test set (25%).

```r
data <- read.csv('pml-training.csv')
inTraining <- createDataPartition(data$classe, p=0.75, list=FALSE)
training <- data[inTraining,]
testing <- data[-inTraining,]
```

We are interested in predicting based on acceleromater data, so next we limit our
testing and training predictors to just those items.  Once we've pulled out the
predictors and outcomes we care about, we can remove the original full data set to
save on a bit of memory.

```r
trainOutcome <- training[, names(training) == "classe"]
trainAccelerometers <- training[, grepl("^accel", names(training))]
testOutcome <- testing[, names(testing) == "classe"]
testAccelerometers <- testing[, grepl("^accel", names(testing))]
rm(data)
```

Now that our data is in shape, we train a Generalize Boosted Regression Model (GBM).
We use 4-fold cross validation repeating with 3 full sets of folds.  This reduces the
variance in our results, and validates that we aren't overfitting our model to the
test data.

```r
library(survival, quietly=TRUE)
library(gbm, quietly=TRUE)
library(splines, quietly=TRUE)
fitControl <- trainControl(method="repeatedcv", number=4, repeats=3)
modelFit <- train(trainOutcome ~ ., data=trainAccelerometers, method="gbm", trControl=fitControl, verbose=FALSE)
```

Finally we do some predictions on our test data and look at the confusion matrix
to see how it performed.

```r
predictions <- predict(modelFit, testAccelerometers)
cm <- confusionMatrix(predictions, testOutcome)
accuracyPercent <- cm$overall['Accuracy'] * 100
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1239   98   78   68   27
##          B   25  687   44   23   45
##          C   47   93  710   52   49
##          D   81   39   21  644   35
##          E    3   32    2   17  745
## 
## Overall Statistics
##                                        
##                Accuracy : 0.821        
##                  95% CI : (0.81, 0.831)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.773        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.888    0.724    0.830    0.801    0.827
## Specificity             0.923    0.965    0.940    0.957    0.987
## Pos Pred Value          0.821    0.834    0.747    0.785    0.932
## Neg Pred Value          0.954    0.936    0.963    0.961    0.962
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.253    0.140    0.145    0.131    0.152
## Detection Prevalence    0.308    0.168    0.194    0.167    0.163
## Balanced Accuracy       0.905    0.845    0.885    0.879    0.907
```

We can see that the our model has 82.0759% accuracy, and estimate
our out-of-sample error rate to be 17.9241% based on this.

