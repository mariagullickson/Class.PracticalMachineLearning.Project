Predicting Exercise Styles
==========================

The goal of this project is to use accelerometer data from 6 participants while
they exercise to predict the manner in which they are exercising.

First we load up the data and split it into a training set (75%) and a validation set (25%).

```r
library(caret, quietly=TRUE)
data <- read.csv('pml-training.csv')
inTraining <- createDataPartition(data$classe, p=0.75, list=FALSE)
training <- data[inTraining,]
validation <- data[-inTraining,]
```

We are interested in predicting based on accelerometer data, so next we limit our
training and validation predictors to just those items.  Once we've pulled out the
predictors and outcomes we care about, we can remove the original full data set to
save on a bit of memory.

```r
trainOutcome <- training[, names(training) == "classe"]
trainAccelerometers <- training[, grepl("^accel", names(training))]
validationOutcome <- validation[, names(validation) == "classe"]
validationAccelerometers <- validation[, grepl("^accel", names(validation))]
rm(data)
```

Now that our data is in shape, we can try out some modeling.  We will try out several
models, using 5-fold cross validation repeating with 3 full sets of folds.  This reduces the
variance in our results, and validates that we aren't overfitting our model to the
training data.

```r
library(survival, quietly=TRUE)
library(gbm, quietly=TRUE)
library(splines, quietly=TRUE)
fitControl <- trainControl(method="repeatedcv", number=4, repeats=3)
```

First we will try out a Linear Discriminant Analysis Model (LDA).

```r
ldaFit <- train(trainOutcome ~ ., data=trainAccelerometers, method="lda",
                trControl=fitControl, verbose=FALSE)
ldaPredictions <- predict(ldaFit, validationAccelerometers)
ldaCM <- confusionMatrix(ldaPredictions, validationOutcome)
ldaAccuracy <- ldaCM$overall['Accuracy'] * 100
ldaCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 977 244 420 111 146
##          B  75 458  89  89 182
##          C 106 130 247  61  54
##          D 203  82  93 476 113
##          E  34  35   6  67 406
## 
## Overall Statistics
##                                         
##                Accuracy : 0.523         
##                  95% CI : (0.509, 0.537)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.389         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.700   0.4826   0.2889   0.5920   0.4506
## Specificity             0.738   0.8900   0.9133   0.8802   0.9645
## Pos Pred Value          0.515   0.5129   0.4130   0.4922   0.7409
## Neg Pred Value          0.861   0.8776   0.8588   0.9167   0.8864
## Prevalence              0.284   0.1935   0.1743   0.1639   0.1837
## Detection Rate          0.199   0.0934   0.0504   0.0971   0.0828
## Detection Prevalence    0.387   0.1821   0.1219   0.1972   0.1117
## Balanced Accuracy       0.719   0.6863   0.6011   0.7361   0.7076
```

Next we will try out a Generalize Boosted Regression Model (GBM).

```r
gbmFit <- train(trainOutcome ~ ., data=trainAccelerometers, method="gbm",
                trControl=fitControl, verbose=FALSE)
gbmPredictions <- predict(gbmFit, validationAccelerometers)
gbmCM <- confusionMatrix(gbmPredictions, validationOutcome)
gbmAccuracy <- gbmCM$overall['Accuracy'] * 100
gbmCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1247  107   74   60   15
##          B   26  695   62   19   53
##          C   46   92  693   44   34
##          D   74   33   17  664   36
##          E    2   22    9   17  763
## 
## Overall Statistics
##                                         
##                Accuracy : 0.828         
##                  95% CI : (0.817, 0.839)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.782         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.894    0.732    0.811    0.826    0.847
## Specificity             0.927    0.960    0.947    0.961    0.988
## Pos Pred Value          0.830    0.813    0.762    0.806    0.938
## Neg Pred Value          0.956    0.937    0.959    0.966    0.966
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.254    0.142    0.141    0.135    0.156
## Detection Prevalence    0.306    0.174    0.185    0.168    0.166
## Balanced Accuracy       0.910    0.846    0.879    0.893    0.917
```

Then we will try out a Random Forest Model (RF).

```r
rfFit <- train(trainOutcome ~ ., data=trainAccelerometers, method="rf",
               trControl=fitControl, verbose=FALSE)
rfPredictions <- predict(rfFit, validationAccelerometers)
rfCM <- confusionMatrix(rfPredictions, validationOutcome)
rfAccuracy <- rfCM$overall['Accuracy'] * 100
rfCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1342   42   12   17    0
##          B    7  874   26    3   12
##          C   18   23  813   31    5
##          D   24    6    4  748   10
##          E    4    4    0    5  874
## 
## Overall Statistics
##                                         
##                Accuracy : 0.948         
##                  95% CI : (0.942, 0.954)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.935         
##  Mcnemar's Test P-Value : 6.09e-10      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.962    0.921    0.951    0.930    0.970
## Specificity             0.980    0.988    0.981    0.989    0.997
## Pos Pred Value          0.950    0.948    0.913    0.944    0.985
## Neg Pred Value          0.985    0.981    0.990    0.986    0.993
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.274    0.178    0.166    0.153    0.178
## Detection Prevalence    0.288    0.188    0.181    0.162    0.181
## Balanced Accuracy       0.971    0.954    0.966    0.960    0.983
```

Now that we've got 3 possible models, we'll compare the accuracy of each of them
to see which is the best fit for this problem.


```r
library(ggplot2, quietly=TRUE)
library(gridExtra, quietly=TRUE)
Model = c('LDA', 'GBM', 'RF')
Accuracy = c(ldaAccuracy, gbmAccuracy, rfAccuracy)
OutOfSampleErrorEstimate = 100-Accuracy
modelSummary = data.frame(Model, Accuracy, OutOfSampleErrorEstimate)
accuracyPlot <- ggplot(data=modelSummary, aes(x=Model, y=Accuracy, fill=Model)) + geom_bar(stat="identity") + xlab("Model") + ylab("Accuracy %") + ggtitle("Model Accuracy")
errorPlot <- ggplot(data=modelSummary, aes(x=Model, y=OutOfSampleErrorEstimate, fill=Model)) + geom_bar(stat="identity") + xlab("Model") + ylab("Out Of Sample Error %") + ggtitle("Model Error Estimate")
grid.arrange(accuracyPlot, errorPlot, ncol=2)
```

![plot of chunk modelSummary](figure/modelSummary.png) 

From this, we can see that the Random Forest model clearly has the highest accuracy (94.8409%),
and thus the lowest estimated out-of-sample error rate (5.1591%).  This is the model
we will choose to use.

Finally it's time to generate results for the test data.  We will run the prediction using
the Random Forest model we generated and spit the results out to files.
```
testing <- read.csv('pml-testing.csv')
testAccelerometers <- testing[, grepl("^accel", names(testing))]
answers <- predict(rfFit, testAccelerometers)

pml_write_files = function(x)
{
  n = length(x)
  for (i in 1:n)
  {
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```
