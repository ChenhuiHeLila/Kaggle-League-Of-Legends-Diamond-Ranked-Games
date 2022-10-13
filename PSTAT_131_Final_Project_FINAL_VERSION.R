knitr::opts_chunk$set(echo = TRUE)
library(knitr)


# Introduction
knitr::include_graphics("League of Legends.png")

library(vembedr)
embed_youtube("BGtROJeMPeE")

# Loading Data and Packages

# Loading Data
# read in the data
RankData <- read.csv("high_diamond_ranked_10min.csv")

# Check the dimension
dimension <- dim(RankData)
dimension

#Loading Packages
library(tidyverse)
library(ForImp)
library(dplyr)
library(randomForest)
library(gbm)
library(ISLR)
library(tree)
library(class)
library(corrplot)
library(ROCR)
library(readr)
library(caret)
library(factoextra)

# Data Cleaning
# Exclude gameID column
RData <- RankData %>% 
  select(-gameId)
# Then, we move blueWins to the first column.
blueWins = RData[,1]

#Select important variables
blueData = RData %>% 
  select(-blueCSPerMin, -redCSPerMin, -blueGoldPerMin, -redGoldPerMin, -redDeaths, -redFirstBlood, -blueDeaths, -redGoldDiff, -redExperienceDiff)

#Check missing value#
sum(is.na(blueData))

# Exploratory Data Analysis
# Summary the data #
summary(blueData)

# Win Rate When Has a Gold Advantage #
red_wins <- nrow(blueData[blueData$blueWins == "0",])
blue_wins <- nrow(blueData[blueData$blueWins == "1",])
blue_wins_ga <- filter(blueData, blueGoldDiff > 0 & blueWins == "1")
blue_wins_gd <- filter(blueData, blueGoldDiff < 0 & blueWins == "1")
red_wins_ga <- filter(blueData, blueGoldDiff > 0 & blueWins == "0")
red_wins_gd <- filter(blueData, blueGoldDiff < 0 & blueWins == "0")
prob_blue_wins_ga <- round((length(blue_wins_ga[,1]))/blue_wins, digits = 2)
prob_blue_wins_gd <- round((length(blue_wins_gd[,1]))/blue_wins, digits = 2)
prob_red_wins_ga <- round((length(red_wins_ga[,1]))/red_wins, digits = 2)
prob_red_wins_gd <- round((length(red_wins_gd[,1]))/red_wins, digits = 2)
prob_blue_wins_gold <- c(prob_blue_wins_ga, prob_blue_wins_gd)
prob_red_wins_gold <- c(prob_red_wins_ga, prob_red_wins_gd)
barplot(prob_blue_wins_gold, main = "Blue win When Blue Has Gold Advantage",
        xlab = "Win", ylab = "Probability",
        col = c("blue", "red"))
barplot(prob_red_wins_gold, main = "Red win When Blue Has Gold Advantage",
        xlab = "Win", ylab = "Probability",
        col = c("blue", "red"))

# Fairness #
red_wins <- nrow(blueData[blueData$blueWins == "0",])
blue_wins <- nrow(blueData[blueData$blueWins == "1",])
prob_red_wins <- round((nrow(blueData[blueData$blueWins == "0",])) / length(blueData[,1]), digits = 2)
prob_blue_wins <-  round((nrow(blueData[blueData$blueWins == "1",])) / length(blueData[,1]), digits = 2)
prob_wins <- c(prob_red_wins, prob_blue_wins)
barplot(prob_wins, main = "The probability to win this ranked game of each team",
        xlab = "Team", ylab = "Probability",
        col = c("red", "blue"))

# Correlation 1#
CorrMatrix<-cor(blueData%>%select(-1))
corrplot(CorrMatrix,method = "color",type = "full", addCoef.col = "black", order = "hclust", tl.cex = 0.45,number.cex = 0.35)
# PCA #
#PVE Value#
pr.out = prcomp(blueData%>%select(-1), scale = TRUE)
pve = pr.out$sdev^2 / sum(pr.out$sdev^2)
pve

#PVE and Cumulative PVE Plot#
par(mfrow = c(1,2))
plot(pve,xlab="Principal Component",
     ylab="Proportion of Variance Explained ", ylim=c(0,1),type='l')
plot(cumsum(pve), xlab="Principal Component ",
     ylab=" Cumulative PVE", type = "l")

#PCA Plot#
fviz_pca_var(pr.out,
             col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

# Corrlation 2 #
BData = blueData %>% select(blueWins:blueExperienceDiff)
BData = BData %>% select(-blueEliteMonsters, -blueAvgLevel)

CorrMatrix2<-cor(BData%>%select(-1))
corrplot(CorrMatrix2,method = "color",type = "full", addCoef.col = "black", order = "hclust", tl.cex = 0.45,number.cex = 0.35)

# Data Splitting #
set.seed(1)

#We take 80% training data and 20% testing data.
train <- sample(1:nrow(BData), 0.8*nrow(BData))

blueData.train <- BData[train,]
blueData.test <- BData[-train,]

# dimensions of training dataset
dim(blueData.train)
# dimensions of test dataset
dim(blueData.test)

# Model fitting
# Logistic Regression
#Summary of the Logistic Regression
glm.fit <- glm(blueWins ~ ., 
               family = binomial("logit"),
               data = blueData.train)
summary(glm.fit)

#Find the probability of training dataset
prob.training = round(predict(glm.fit, type = "response"),digits = 2)

#Confusion matrix of training dataset
# Save the predicted labels using 0.5 as a threshold
blueData.train1 = blueData.train %>%
  mutate(predBLUEWINS=as.factor(ifelse(prob.training<=0.5, "No", "Yes")))
# Confusion matrix (training error/accuracy)
table(pred=blueData.train1$predBLUEWINS,
      true=blueData.train1$blueWins)
calc_error_rate <- function(predicted.value, true.value){
  return(mean(true.value!=predicted.value))
}

#Training Error
pred.train = predict(glm.fit, newdata = blueData.train1, type = "response")
pred.train <- ifelse(pred.train > .5, 1, 0)
calc_error_rate(pred.train, blueData.train1$blueWins)

# Accurate rate for training dataset
AR.train <- 1 - calc_error_rate(pred.train, blueData.train1$blueWins)
AR.train

# For lose game, the correct classified rate is
lostR <- 2977 / (2948 + 1093)
lostR

# For win game, the correct classified rate is
winR <- 2801 / (2801 + 1093)
winR

#Similar with training part
prob.test <- round(predict(glm.fit, blueData.test, type = "response"), digits = 2)

blueData.test1 <- blueData.test %>%
  mutate(winRate=prob.test)

blueData.test1 <- blueData.test1 %>%
  mutate(predBLUEWINS=as.factor(ifelse(prob.test<=0.5, "No", "Yes")))

# confusion matrix (test error)
table(pred=blueData.test1$predBLUEWINS, true=blueData.test1$blueWins)

#Test Error
pred.test = predict(glm.fit, newdata = blueData.test1, type = "response")
pred.test <- ifelse(pred.test > .5, 1, 0)
calc_error_rate(pred.test, blueData.test1$blueWins)

#Accurate rate for test dataset
AR.test <- 1 - calc_error_rate(pred.test, blueData.test$blueWins)
AR.test

#ROC plot#
pred = prediction(prob.training, blueData.train1$blueWins)

# We want TPR on the y axis and FPR on the x axis
perf = performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=2, lwd=3, main="ROC curve")
abline(0,1)


#AUC#
auc = performance(pred, "auc")@y.values
auc

# K-Nearest Neighbor Method
set.seed(1)
#We take 80% training data and 20% testing data.
train <- sample(1:nrow(blueData), 0.8*nrow(blueData))

blueData.train <- BData[train,]
blueData.test <- BData[-train,]

YTrain = blueData.train$blueWins
XTrain = blueData.train %>% select(-blueWins) %>% scale(center = TRUE, scale = TRUE)

YTest = blueData.test$blueWins
XTest = blueData.test %>% select(-blueWins) %>% scale(center = TRUE, scale = TRUE)

validation.error = NULL
allK = 1:50
set.seed(1)
for (i in allK){ 
  pred.Yval = knn.cv(train=XTrain, cl=YTrain, k=i) 
  # Loop through different number of neighbors
  # Predict on the left-out validation set
  validation.error = c(validation.error, mean(pred.Yval!=YTrain)) 
}
# Validation error for 1-NN, 2-NN, ..., 50-NN
plot(allK, validation.error, type = "l", xlab = "k")

numneighbor = max(allK[validation.error == min(validation.error)])
numneighbor


#Calculate training error rate#
set.seed(1)
pred.YTtrain = knn(train=XTrain, test=XTrain, cl=YTrain, k=44,
                   prob = T)

conf.train = table(predicted=pred.YTtrain, true=YTrain)
conf.train

1 - sum(diag(conf.train)/sum(conf.train))


#Calculate test error rate#
pred.YTest = knn(train=XTrain, test=XTest, cl=YTrain, k=44)

conf.test = table(predicted=pred.YTest, true=YTest)
conf.test

1 - sum(diag(conf.test)/sum(conf.test))


# Random Forest and Bagging #
set.seed(1)

#We take 80% training data and 20% testing data.
train <- sample(1:nrow(blueData), 0.8*nrow(blueData))

blueData.train <- BData[train,]
blueData.test <- BData[-train,]


#Random Forest#
set.seed(1)
fit.rf<-randomForest(as.factor(blueWins) ~ .,
                     data=blueData.train, 
                     mtry=4, 
                     ntree=500, 
                     nodesize=5,
                     importance=TRUE)
fit.rf

plot(fit.rf)
legend("top", 
       legend = colnames(fit.rf$err.rate),
       lty = 1:3,
       col = 1:3,
       horiz = T)

importance(fit.rf)

varImpPlot(fit.rf,main='feature important')

pred<-predict(fit.rf,blueData.test)
rf.err = table(pred = pred, truth = blueData.test$blueWins)
rf.err
test.rf.err = 1 - sum(diag(rf.err))/sum(rf.err)
test.rf.err

#Bagging#
set.seed(1)

train <- sample(1:nrow(blueData), 0.8*nrow(blueData))

blueData.train <- BData[train,]
blueData.test <- BData[-train,]

glimpse(blueData.train)
str(blueData.train)

blueData.train$blueWins <- as.character(blueData.train$blueWins)
blueData.train$blueWins <- as.factor(blueData.train$blueWins)

bag.rank <- randomForest(blueWins~., 
                         data = blueData.train, 
                         mtry=14,
                         importance = TRUE)
bag.rank

plot(bag.rank)
legend("top", 
       colnames(bag.rank$err.rate),
       col = 1:3,
       cex = 0.8, 
       fill = 1:3)

yhat.bag <- predict(bag.rank, newdata = blueData.test)
bag.err <- table(pred=yhat.bag, truth=blueData.test$blueWins)
bag.err
test.bag.err <- 1- sum(diag(bag.err))/sum(bag.err)
test.bag.err


# Boosted tree #
set.seed(1)

#We take 80% training data and 20% testing data.
train <- sample(1:nrow(blueData), 0.8*nrow(blueData))

blueData.train <- BData[train,]
blueData.test <- BData[-train,]

set.seed(1)
boost.blueData = gbm(blueWins ~., 
                     data=blueData.train,
                     distribution="bernoulli", 
                     n.trees=500, 
                     interaction.depth=4)


varimp_gbm <- summary(boost.blueData)
varimp_gbm
varimp_gbm %>%
  arrange(rel.inf) %>%
  mutate(var = as_factor(var)) %>%
  ggplot(aes(x = rel.inf, y = var)) +
  geom_col(fill = "blue") +
  labs(x = "", y = "") +
  theme_bw()

plot(boost.blueData, i="blueGoldDiff")

plot(boost.blueData, i="blueExperienceDiff")

yhat.boost = predict(boost.blueData, 
                     newdata = blueData.test, 
                     n.trees=500)
pred1=as.factor(ifelse(yhat.boost>=0.5,1,0))
boost.err = table(pred = pred1, truth = blueData.test$blueWins)
boost.err
test.boost.err = 1 - sum(diag(boost.err))/sum(boost.err)
test.boost.err

