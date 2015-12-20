#resart the project once more
install.packages("gputools")

library(rpud)

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
#try


datasetMAIN<- removeSparseTerms(datasetMAIN, 0.9) #  1-3/nrow(datasetMAIN) use this for best results

datasetMAIN <- as.data.frame(as.matrix(datasetMAIN))

datasetMAIN$ingredients_count  <- rowSums(datasetMAIN) # simple count of ingredients per receipe

#datasetMAIN$cuisine
print("Done creating dataframe for decision trees")

temporaryData <- datasetMAIN  #just to be safe


datasetMAIN$cuisine <- as.factor(c(train$cuisine, rep("italian", nrow(test))))

trainDataset  <- datasetMAIN[1:nrow(train), ]
testDataset <- datasetMAIN[-(1:nrow(train)), ]

###############################################################################################################################################

# shuffle and split the data into three parts
set.seed(1234)
trainDataset <- trainDataset[sample(nrow(trainDataset)),]
split <- floor(nrow(trainDataset)/3)
ensembleData <- trainDataset[0:split,]
blenderData <- trainDataset[(split+1):(split*2),]
testingData <- trainDataset[(split*2+1):nrow(trainDataset),]

# set label name and predictors
labelName <- 'cuisine'
predictors <- names(ensembleData)[names(ensembleData) != labelName]

library(caret)
# create a caret control object to control the number of cross-validations performed
myControl <- trainControl(method='cv', number=2, returnResamp='none')

# quick benchmark model 
blenderData<-na.omit(blenderData)
testingData<-na.omit(testingData)
ensembleData<-na.omit(ensembleData)

test_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)


# train all the ensemble models with ensembleData
model_gbm <- train(ensembleData[,predictors], ensembleData[,labelName], method='gbm', trControl=myControl)
model_rpart <- train(ensembleData[,predictors], ensembleData[,labelName], method='rpart', trControl=myControl)
model_treebag <- train(ensembleData[,predictors], ensembleData[,labelName], method='treebag', trControl=myControl)

preder<-predict(testDataset,  object= model_treebag)
plot(preder)
FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(preder))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'ankap.csv', row.names=F, quote=F)

# get predictions for each ensemble model for two last data sets
# and add them back to themselves
blenderData$gbm_PROB <- predict(object=model_gbm, blenderData[,predictors])
blenderData$rf_PROB <- predict(object=model_rpart, blenderData[,predictors])
blenderData$treebag_PROB <- predict(object=model_treebag, blenderData[,predictors])
testingData$gbm_PROB <- predict(object=model_gbm, testingData[,predictors])
testingData$rf_PROB <- predict(object=model_rpart, testingData[,predictors])
testingData$treebag_PROB <- predict(object=model_treebag, testingData[,predictors])


testDataset$gbm_PROB <- predict(object=model_gbm, testDataset[,predictors])
testDataset$rf_PROB <- predict(object=model_rpart, testDataset[,predictors])
testDataset$treebag_PROB <- predict(object=model_treebag, testDataset[,predictors])


predictors <- names(blenderData)[names(blenderData) != labelName]
final_blender_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)

# run a final model to blend all the probabilities together
predictors <- names(blenderData)[names(blenderData) != labelName]
final_blender_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)

pedik<-predict(object=final_blender_model,testDataset[,predictors])
plot(pedik)

FINALLYY <- cbind(as.data.frame(test$id), as.data.frame(pedik))
colnames(FINALLYY) <-c("id","cuisine")
write.csv(FINALLYY, file = 'ensamble.csv', row.names=F, quote=F)

