#resart the project once more
library(class)
library(caret)
library(caTools)
library(jsonlite)
library(tm)
library(rpart)
library(rpart.plot)
library(data.table)
library(Matrix)
library(rpart.control)
library(randomForest)
library(SnowballC)
library(party)
library(e1071)
library(ROCR)
library(xgboost)
#TRY
#preprocess to different columns of ingredients
train <- fromJSON('train.json', flatten = TRUE)
test <- fromJSON('test.json', flatten = TRUE)

table(train$cuisine)

train$ingredients <- lapply(train$ingredients, FUN=tolower)
test$ingredients <- lapply(test$ingredients, FUN=tolower)
train$ingredients <- lapply(train$ingredients, FUN=function(x) gsub("-", "_", x))  
test$ingredients <- lapply(test$ingredients, FUN=function(x) gsub("-", "_", x)) 
train$ingredients <- lapply(train$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x))
test$ingredients <- lapply(test$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) 

#use perhaps
#control = list(weighting =  function(x)  weightTfIdf(x, normalize =   FALSE),stopwords = TRUE)


comb_ingredients <- c(Corpus(VectorSource(train$ingredients)), Corpus(VectorSource(test$ingredients)))
comb_ingredients <- tm_map(comb_ingredients, stemDocument, language="english")
datasetMAIN <- DocumentTermMatrix(comb_ingredients)
#try

datasetMAIN<- removeSparseTerms(datasetMAIN, 0.99) 
datasetMAIN <- as.data.frame(as.matrix(datasetMAIN))

print("Done creating dataframe for decision trees")

temporaryData <- datasetMAIN  #just to be safe
datasetMAIN$cuisine <- as.factor(c(train$cuisine, rep("italian", nrow(test))))



trainDataset  <- datasetMAIN[1:nrow(train), ]
testDataset <- datasetMAIN[-(1:nrow(train)), ]



fit<-rpart(cuisine~., data=trainDataset, method="class",control = rpart.control(cp = 0.00001)) #method-> classification
prp(fit, type=1, extra=4)
prp(fit)

PredFit<-predict(fit, newdata=testDataset, type="class")
table(PredFit,testDataset$cuisine)
confusionMatrix(table(PredFit,testDataset$cuisine))

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits


Prediction <- predict(fit, newdata = testDataset, type = "class")
sample <- read.csv('sample_submission.csv')

FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'Submit22.csv', row.names=F, quote=F)



#alternative random forest

fit1 <- randomForest(cuisine~., data=trainDataset,importance=TRUE,ntree = 1500)
print(fit1) # view results 
importance(fit1) # importance of each predictor
PredFit<-predict(fit1, newdata=testDataset, type="class")
table(PredFit,testDataset$cuisine)
confusionMatrix(table(PredFit,testDataset$cuisine))

printcp(fit1) # display the results
plotcp(fit1) # visualize cross-validation results
summary(fit1) # detailed summary of splits


Prediction <- predict(fit1, newdata = testDataset, type = "class")
sample <- read.csv('sample_submission.csv')

FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'Submit33.csv', row.names=F, quote=F)


#Another random Forest
fit2 <- randomForest(cuisine~., data=trainDataset,importance=TRUE,ntree = 2000)
print(fit2) # view results 
importance(fit2) # importance of each predictor
PredFit<-predict(fit2, newdata=testDataset, type="class")
table(PredFit,testDataset$cuisine)
confusionMatrix(table(PredFit,testDataset$cuisine))

printcp(fit1) # display the results
plotcp(fit1) # visualize cross-validation results
summary(fit1) # detailed summary of splits


Prediction <- predict(fit1, newdata = testDataset, type = "class")
sample <- read.csv('sample_submission.csv')

FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitBIG.csv', row.names=F, quote=F)



rf.all<-combine(fit1,fit2)
print(rf.all) # view results 
importance(rf.all) # importance of each predictor
predFitAll<-predict(rf.all, newdata=testDataset, type="class")
table(predFitAll,testDataset$cuisine)
confusionMatrix(table(predFitAll,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(predFitAll))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'Submit35.csv', row.names=F, quote=F)

#al
fit <- ctree(cuisine ~., 
             data=trainDataset)
plot(fit, main="Conditional Inference Tree for data")



#knn

#Knn Try COMP STUCKKK
KNN<-knn(trainDataset[,-252],testDataset[,-252],cl=trainDataset$cuisine,k=7)#best k=7
table(testDataset$default,M)
summary(M)
confusionMatrix(M, testDataset$default,positive = "Defaulted" )

#k-fold cross validation

ctrl<-trainControl(method="repeatedcv", repeats=5, number=10)
knnFit<-train(cuisine~., data=trainDataset, method="knn",trControl=ctrl, tuneLength=20)
knnFit
plot(knnFit) # ay sents lav patkeracreci.

finalKNN<-knn(trainDataset[,-"cuisine"],testDataset[,-"cuisine"],cl=DataS$default,k=7)#best k=7
table(testDataset$default,finalKNN)

confusionMatrix(finalKNN, testDataset$default,positive = "Defaulted" )



#XGBOOST


xgbmat     <- xgb.DMatrix(Matrix(data.matrix(trainDataset[, !colnames(trainDataset) %in% c("cuisine")])), label=as.numeric(trainDataset$cuisine)-1)

#- train our multiclass classification model using softmax
xgb <- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)
PredXGB<- predict(xgb, newdata = data.matrix(testDataset[, !colnames(testDataset) %in% c("cuisine")]))
table(PredXGB,testDataset$cuisine)
PredXGB.text <- levels(testDataset$cuisine)[PredXGB+1]

sample <- read.csv('sample_submission.csv')

submit_match   <- cbind(as.data.frame(test$id), as.data.frame(PredXGB.text))
colnames(submit_match) <- c("id", "cuisine")
submit_match   <- data.table(submit_match, key="id")
submit_cuisine <- submit_match[id==sample$id, as.matrix(submit_match$cuisine)]

submission <- data.frame(id = sample$id, cuisine = submit_cuisine)
write.csv(submission, file = 'submi50.csv', row.names=F, quote=F)