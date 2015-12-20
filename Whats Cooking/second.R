#resart the project once more
library(nnet)
library(class)
library(caret)
library(caTools)
library(jsonlite)
library(tm)
library(rpart)
library(rpart.plot)
library(data.table)
library(Matrix)
library(e1071)
library(stringdist)
library(kernlab)
library(nnet)
library(RCurl)


#library(RWeka)
library(koRpus)
library(randomForest)
library(SnowballC)
library(party)
library(neuralnet)
library(randomForestSRC)
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

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tokenize_ngrams <- function(x, n=3) return(rownames(as.data.frame(unclass(textcnt(x,method="string",n=n)))))

comb_ingredients <- c(Corpus(VectorSource(train$ingredients)), Corpus(VectorSource(test$ingredients)))
comb_ingredients <- tm_map(comb_ingredients, stemDocument, language="english")
comb_ingredients <- tm_map(comb_ingredients, removeNumbers)
# remove punctuation
comb_ingredients <- tm_map(comb_ingredients, removePunctuation)

comb_ingredients  <- tm_map(comb_ingredients, content_transformer(tolower),lazy=TRUE)
comb_ingredients  <- tm_map(comb_ingredients,removeWords, stopwords(),lazy=TRUE)
comb_ingredients  <- tm_map(comb_ingredients,stripWhitespace,lazy=TRUE)
comb_ingredients <- tm_map(comb_ingredients, removeWords, stopwords("english"),lazy=TRUE)

datasetMAIN <- DocumentTermMatrix(comb_ingredients)

datasetMAIN<- removeSparseTerms(datasetMAIN, 1-3/nrow(datasetMAIN))

datasetMAIN <- as.data.frame(as.matrix(datasetMAIN))

datasetMAIN$ingredients_count  <- rowSums(datasetMAIN) # simple count of ingredients per receipe

#datasetMAIN$cuisine
print("Done creating dataframe for decision trees")

temporaryData <- datasetMAIN  #just to be safe
datasetMAIN<-temporaryData


datasetMAIN$cuisine <- as.factor(c(train$cuisine, rep("italian", nrow(test))))


str(datasetMAIN$cuisine)

#Cleanup. (BAD IDEA)
datasetMAIN$allpurpos<-NULL
datasetMAIN$and<-NULL
datasetMAIN$bake<-NULL
datasetMAIN$bell<-NULL
datasetMAIN$black<-NULL
datasetMAIN$boneless<-NULL
datasetMAIN$boil<-NULL
datasetMAIN$leaf<-NULL
datasetMAIN$brown<-NULL
datasetMAIN$cold<-NULL
datasetMAIN$cook<-NULL
datasetMAIN$crack<-NULL
datasetMAIN$dark<-NULL
datasetMAIN$free<-NULL
datasetMAIN$fat<-NULL
datasetMAIN$hot<-NULL
datasetMAIN$fine<-NULL
datasetMAIN$green<-NULL
datasetMAIN$bake<-NULL
datasetMAIN$breast<-NULL
datasetMAIN$chile<-NULL
datasetMAIN$extract<-NULL
datasetMAIN$ground<-NULL
datasetMAIN$golden<-NULL
datasetMAIN$flat<-NULL
datasetMAIN$frozen<-NULL
datasetMAIN$fresh<-NULL
datasetMAIN$larg<-NULL
datasetMAIN$firm<-NULL
datasetMAIN$ice<-NULL

trainDataset  <- datasetMAIN[1:nrow(train), ]
testDataset <- datasetMAIN[-(1:nrow(train)), ]


fit<-rpart(cuisine~., data=trainDataset, method="class",control = rpart.control(cp = 0.000002)) #method-> classification
prp(fit, type=1, extra=4)
prp(fit)

PredFit<-predict(fit, newdata=testDataset, type="class")
table(PredFit,testDataset$cuisine)
confusionMatrix(table(PredFit,testDataset$cuisine))

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits


Prediction <- predict(fit, newdata = testDataset, type = "class")

FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitRpart.csv', row.names=F, quote=F)



#alternative random forest
tuneER<-tuneRF(trainDataset,trainDataset$cuisine,ntreeTry = 100)
fit1 <- randomForest(cuisine~., data=trainDataset,ntree=1000,mtry=188) ##100tree mtry99 done 75%......1000tree mtry188 76%....mixup resamplin 79%
plot(fit1)
Prediction <- predict(fit1, newdata = testDataset, type = "class")
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'newVersion.csv', row.names=F, quote=F)
varImpPlot(fit1)

varImp(fit1)
importance(fit1)

plot(Prediction)
print(fit1) # view results 
importance(fit1) # importance of each predictor


#ctree try (bad stuff)
fit <- ctree(cuisine ~., 
             data=trainDataset)
plot(fit, main="Conditional Inference Tree for data")




#H2O Random Forest (didnt get it)
library(h2o)
install.packages("h2o")

trainH2O <- as.h2o(localH2O, trainDataset, key="trainDataset.hex")
prostate.hex <- as.h2o(localH2O, trainDataset, key="prostate.hex")

h2o.randomForest(x=trainDataset,y=cuisine,ntrees = 10)
help(h2o.randomForest)




#Cforest
library(party)
install.packages("party")
fit3 <- cforest(cuisine~., data=trainDataset,controls = cforest_unbiased( ntree = 10))
plot(fit3)
varimp(fit3)
Prediction <- predict(fit3, newdata = testDataset,type="prob")
Prediction
table(Prediction,testDataset$cuisine)
confusionMatrix(table(Prediction,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitAnhuisERO3.csv', row.names=F, quote=F)
varImpPlot(fit1)



# WHY NOT SVM

Model<-svm(cuisine~.,data = trainDataset)
Prediction1 <- predict(Model, newdata = testDataset)
table(Prediction1,testDataset$cuisine)
confusionMatrix(table(Prediction1,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction1))
plot(Prediction1)
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitAnhuisERO_SVM_Linear.csv', row.names=F, quote=F)
varImpPlot(Model)




#ksvm

svp <- ksvm(cuisine~.,data = trainDataset,type="C-svc",kernel="vanilladot",C=10)
Prediction2 <- predict(svp, newdata = testDataset)
plot(Prediction2)
confusionMatrix(table(Prediction2,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction2))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitAnhuisERO_KSVM_Linear.csv', row.names=F, quote=F)
varImpPlot(Model)





#tuning SVM


obj <-best.tune(svm, cuisine ~., data = trainDataset, kernel =
                  "polynomial")
obj

ModelTUNED<-svm(cuisine~.,data = trainDataset,kernel =
                  "polynomial",gamma=0.0004977601,cost=1,coef.0=0,degree=3)

Prediction3 <- predict(ModelTUNED, newdata = testDataset)
plot(Prediction3)
confusionMatrix(table(Prediction3,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction3))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'SubmitAnhuisERO_KSVM_Linear.csv', row.names=F, quote=F)





#trying Neural Networks
net<-nnet(cuisine~.,data = trainDataset, size=6, rang = 0.7, decay = 0, Hess = FALSE, trace = TRUE, MaxNWts = 25000,
          abstol = 1.0e-4, reltol = 1.0e-8,maxit=2000)

par(mar=numeric(4),mfrow=c(1,2),family='serif')

newOne<-predict(net, testDataset,type="class")
table(newOne,testDataset$cuisine)
confusionMatrix(table(newOne,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(newOne))
colnames(FINALLYY) <-c("id","cuisine")
str(FINALLYY)
write.csv(FINALLYY, file = 'Neural3.csv', row.names=F, quote=F)


#4 layer nnet with 1000 iteration gave 65% result (best)


#naive bayes


naive<-naiveBayes(cuisine~.,data = trainDataset,laplace = 0)
print(naive)
predictor1<-predict(naive, testDataset,type="class")
plot(predictor1)
predictor1
table(predictor1,testDataset$cuisine)
confusionMatrix(table(predictor1,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(predictor1))
colnames(FINALLYY) <-c("id","cuisine")
str(FINALLYY)
write.csv(FINALLYY, file = 'Naive.csv', row.names=F, quote=F)

#survive This Forest

#200tree mtry188

ModelOfSurvival<-rfsrc(cuisine~.,trainDataset,ntree=200,mtry=188)
plot(ModelOfSurvival)

predictorSurvival<-predict(ModelOfSurvival, testDataset,type="class")
plot(preder)
preder
table(preder$class,testDataset$cuisine)
confusionMatrix(table(preder$class,testDataset$cuisine))
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(preder$class))
colnames(FINALLYY) <-c("id","cuisine")
str(FINALLYY)
write.csv(FINALLYY, file = 'survival2.csv', row.names=F, quote=F)
preder<-predict.rfsrc(ModelOfSurvival, testDataset,type="class")


#pure Random forest (top 10-25 items)


ModelOfSurvival<-randomForest(cuisine~tortilla+ingredients_count+soy+oliv+ginger+cumin+parmesan+cilantro+chees+lime+sauc+chili+fish+sesam+basil+pepper+curri+sugar+salsa+corn+oil+garlic+masala+butter+buttermilk+rice+seed+milk+feta+egg,trainDataset,ntree=500)
plot(ModelOfSurvival)
varImpPlot(ModelOfSurvival)
Prediction <- predict(ModelOfSurvival, newdata = testDataset, type = "class")
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(Prediction))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'newVersion.csv', row.names=F, quote=F)
varImpPlot(fit1)

varImp(fit1)
importance(fit1)

plot(Prediction)
print(fit1) # view results 
importance(fit1) # importance of each predictor








