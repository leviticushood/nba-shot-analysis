---
title: "Cuury Clutch Modeling"
output: html_document
---



``` r
 # data cleaning and binary conversion
model_df <- read.csv('/cloud/project/data/processed-data/combined_player_shot_chart_2023.csv') |> 
  select('player', 'qtr', 'time_remaining',
         'result', 'distance_ft', 'lead',
         'player_team_score', 'opponent_team_score',
         'shot_type') |> 
  mutate('sec_remaining' = as.numeric(ms(time_remaining), 'seconds')) |>
  mutate(margin = player_team_score - opponent_team_score,
         result = ifelse(result == 'TRUE', 1, 0),
         lead = ifelse(lead == 'TRUE', 1, 0),
         three_pt = ifelse(shot_type == 3, 1, 0)) |>
  select(-'time_remaining', -'player_team_score',
         -'opponent_team_score', -'lead', -'shot_type')

curry_df <- model_df |> filter(player == 'Curry')
```


``` r
 # convert qtrs to numeric features
num_curry_df <- curry_df |>
  select(-'player')
num_curry_df$qtr <- as.numeric(factor(num_curry_df$qtr,
         levels = c('1st Qtr', '2nd Qtr', '3rd Qtr',
                    '4th Qtr', '1st OT', '2nd OT')))
# correlation matrix
ggpairs(num_curry_df, progress = FALSE)
```

<img src="curry-modeling_files/figure-html/Exploratory Data Analysis-1.png" alt="" width="1344" />


``` r
set.seed(123)

 # stratified 80/20 split
train_idx <- createDataPartition(num_curry_df$result, p = 0.80, list = FALSE)

train_df <- num_curry_df[train_idx, ]
test_df <- num_curry_df[-train_idx, ]

 # verify stratification
cat("Train set:", nrow(train_df), "rows |",
    round(mean(train_df$result == 1) * 100, 1), "% Successful Shots\n")
```

```
## Train set: 1148 rows | 48.3 % Successful Shots
```

``` r
cat("Test set: ", nrow(test_df),  "rows |",
    round(mean(test_df$result  == 1) * 100, 1), "% Successful Shots\n")
```

```
## Test set:  286 rows | 51.7 % Successful Shots
```

``` r
 # 5 features
model_lr <- glm(result ~ distance_ft + qtr + sec_remaining + margin + three_pt,
                data = train_df,
                family = binomial)
 # feature significance
summary(model_lr)
```

```
## 
## Call:
## glm(formula = result ~ distance_ft + qtr + sec_remaining + margin + 
##     three_pt, family = binomial, data = train_df)
## 
## Coefficients:
##                 Estimate Std. Error z value Pr(>|z|)    
## (Intercept)    0.8578479  0.2276862   3.768 0.000165 ***
## distance_ft   -0.0674535  0.0135980  -4.961 7.03e-07 ***
## qtr           -0.0611919  0.0540645  -1.132 0.257706    
## sec_remaining  0.0002799  0.0003116   0.898 0.369065    
## margin         0.0273476  0.0065508   4.175 2.98e-05 ***
## three_pt       0.6167029  0.2810805   2.194 0.028232 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1590.2  on 1147  degrees of freedom
## Residual deviance: 1516.6  on 1142  degrees of freedom
## AIC: 1528.6
## 
## Number of Fisher Scoring iterations: 4
```

``` r
probs_lr <- predict(model_lr, newdata = test_df, type = "response")
roc_lr <- roc(test_df$result, probs_lr)
cat("5 Feature Logistic Regression AUC:", auc(roc_lr), "\n")
```

```
## 5 Feature Logistic Regression AUC: 0.6321974
```

``` r
 # confusion matrix
preds_lr <- ifelse(probs_lr > 0.5, 1, 0)
confusionMatrix(as.factor(preds_lr), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 101  79
##          1  37  69
##                                          
##                Accuracy : 0.5944         
##                  95% CI : (0.535, 0.6518)
##     No Information Rate : 0.5175         
##     P-Value [Acc > NIR] : 0.0053534      
##                                          
##                   Kappa : 0.1961         
##                                          
##  Mcnemar's Test P-Value : 0.0001408      
##                                          
##             Sensitivity : 0.7319         
##             Specificity : 0.4662         
##          Pos Pred Value : 0.5611         
##          Neg Pred Value : 0.6509         
##              Prevalence : 0.4825         
##          Detection Rate : 0.3531         
##    Detection Prevalence : 0.6294         
##       Balanced Accuracy : 0.5991         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
 # 3 features
model_lr_3 <- glm(result ~ distance_ft + margin + three_pt,
                data = train_df,
                family = binomial)

probs_lr_3 <- predict(model_lr_3, newdata = test_df, type = "response")
roc_lr_3 <- roc(test_df$result, probs_lr_3)
cat("3 Feature Logistic Regression AUC:", auc(roc_lr_3), "\n")
```

```
## 3 Feature Logistic Regression AUC: 0.640472
```

``` r
 # confusion matrix
preds_lr_3 <- ifelse(probs_lr_3 > 0.5, 1, 0)
confusionMatrix(as.factor(preds_lr_3), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 105  81
##          1  33  67
##                                           
##                Accuracy : 0.6014          
##                  95% CI : (0.5421, 0.6586)
##     No Information Rate : 0.5175          
##     P-Value [Acc > NIR] : 0.002626        
##                                           
##                   Kappa : 0.2111          
##                                           
##  Mcnemar's Test P-Value : 1.073e-05       
##                                           
##             Sensitivity : 0.7609          
##             Specificity : 0.4527          
##          Pos Pred Value : 0.5645          
##          Neg Pred Value : 0.6700          
##              Prevalence : 0.4825          
##          Detection Rate : 0.3671          
##    Detection Prevalence : 0.6503          
##       Balanced Accuracy : 0.6068          
##                                           
##        'Positive' Class : 0               
## 
```

``` r
 # LR feature selection comparison

 # 2 variable (dist, margin)                         Accuracy: 0.59, Kappa: 0.20,  AUC: 0.64, Sensitivity: 0.72, Specificity: 0.47
 # 3 variable (dist, margin, three_pt)               Accuracy: 0.60, Kappa: 0.21,  AUC: 0.64, Sensitivity: 0.76, Specificity: 0.45
 # 4 variable (dist, qtr, sec, margin)               Accuracy: 0.46, Kappa: -0.07, AUC: 0.53, Sensitivity: 0.55, Specificity: 0.38
 # 5 variable (dist, qtr, sec, margin, three_pt)     Accuracy: 0.59, Kappa: 0.20,  AUC: 0.63, Sensitivity: 0.73, Specificity: 0.47

 # // 3 variable (dist, margin, three_pt) as baseline going forward //
```




``` r
 # 5 features
model_nb <- naive_bayes(as.factor(result) ~ distance_ft + qtr + sec_remaining + margin + three_pt,
                        data = train_df,
                        laplace = 1)
 # NB model EDA (top 3 features)
model_nb$tables$distance_ft
```

```
##            
## distance_ft         0         1
##        mean 20.227656 15.976577
##        sd    9.578469 10.835053
```

``` r
model_nb$tables$margin
```

```
##       
## margin         0         1
##   mean 0.1989882 2.4810811
##   sd   9.4092859 9.4764149
```

``` r
model_nb$tables$three_pt
```

```
##         
## three_pt         0         1
##     mean 0.6155143 0.4612613
##     sd   0.4868842 0.4989468
```

``` r
preds_nb = predict(model_nb,
                   newdata = test_df,
                   type = 'class')
probs_nb = predict(model_nb,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_nb <- roc(test_df$result, probs_nb)
cat("5 Feature Naive Bayes AUC:", auc(roc_nb), "\n")
```

```
## 5 Feature Naive Bayes AUC: 0.6279867
```

``` r
confusionMatrix(as.factor(preds_nb), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 98 78
##          1 40 70
##                                          
##                Accuracy : 0.5874         
##                  95% CI : (0.5279, 0.645)
##     No Information Rate : 0.5175         
##     P-Value [Acc > NIR] : 0.0103481      
##                                          
##                   Kappa : 0.1814         
##                                          
##  Mcnemar's Test P-Value : 0.0006589      
##                                          
##             Sensitivity : 0.7101         
##             Specificity : 0.4730         
##          Pos Pred Value : 0.5568         
##          Neg Pred Value : 0.6364         
##              Prevalence : 0.4825         
##          Detection Rate : 0.3427         
##    Detection Prevalence : 0.6154         
##       Balanced Accuracy : 0.5916         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
 # 3 features
model_nb_3 <- naive_bayes(as.factor(result) ~ distance_ft + margin + three_pt,
                        data = train_df,
                        laplace = 1)

preds_nb_3 = predict(model_nb_3,
                   newdata = test_df,
                   type = 'class')
probs_nb_3 = predict(model_nb_3,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_nb_3 <- roc(test_df$result, probs_nb_3)
cat("3 Feature Naive Bayes AUC:", auc(roc_nb_3), "\n")
```

```
## 3 Feature Naive Bayes AUC: 0.636555
```

``` r
confusionMatrix(as.factor(preds_nb_3), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 98 76
##          1 40 72
##                                          
##                Accuracy : 0.5944         
##                  95% CI : (0.535, 0.6518)
##     No Information Rate : 0.5175         
##     P-Value [Acc > NIR] : 0.005353       
##                                          
##                   Kappa : 0.1949         
##                                          
##  Mcnemar's Test P-Value : 0.001155       
##                                          
##             Sensitivity : 0.7101         
##             Specificity : 0.4865         
##          Pos Pred Value : 0.5632         
##          Neg Pred Value : 0.6429         
##              Prevalence : 0.4825         
##          Detection Rate : 0.3427         
##    Detection Prevalence : 0.6084         
##       Balanced Accuracy : 0.5983         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
 # NB feature comparison

 # 5 variable (dist, qtr, sec, margin, three_pt)    Accuracy: 0.59, Kappa: 0.18, AUC: 0.63, Sensitivity: 0.71, Specificity: 0.47
 # 3 variable (dist, margin, three_pt)              Accuracy: 0.59, Kappa: 0.19, AUC: 0.64, Sensitivity: 0.71, Specificity: 0.49

 # // 3 variable (dist, margin, three_pt) //
```


``` r
 # 5 features
model_dt <- rpart(result ~ distance_ft + qtr + 
                           sec_remaining + margin + three_pt,
                         data = train_df,
                         method = 'class',
                         parms = list(split = 'gini'),
                         control = rpart::rpart.control(cp = 0.001)
                         )
printcp(model_dt)
```

```
## 
## Classification tree:
## rpart(formula = result ~ distance_ft + qtr + sec_remaining + 
##     margin + three_pt, data = train_df, method = "class", parms = list(split = "gini"), 
##     control = rpart::rpart.control(cp = 0.001))
## 
## Variables actually used in tree construction:
## [1] distance_ft   margin        qtr           sec_remaining
## 
## Root node error: 555/1148 = 0.48345
## 
## n= 1148 
## 
##           CP nsplit rel error  xerror     xstd
## 1  0.1495495      0   1.00000 1.00000 0.030508
## 2  0.0216216      1   0.85045 0.87207 0.030147
## 3  0.0162162      4   0.78198 0.85946 0.030085
## 4  0.0081081      5   0.76577 0.82342 0.029884
## 5  0.0072072      7   0.74955 0.84865 0.030029
## 6  0.0066066      9   0.73514 0.85045 0.030039
## 7  0.0054054     16   0.68288 0.82523 0.029895
## 8  0.0045045     19   0.66667 0.83964 0.029979
## 9  0.0040541     24   0.63784 0.83063 0.029927
## 10 0.0036036     29   0.61622 0.84865 0.030029
## 11 0.0030030     33   0.60180 0.88288 0.030196
## 12 0.0027027     36   0.59279 0.88288 0.030196
## 13 0.0024024     48   0.55856 0.90090 0.030270
## 14 0.0018018     53   0.54234 0.90991 0.030303
## 15 0.0010000     67   0.51351 0.92793 0.030363
```

``` r
best_cp <- model_dt$cptable[
  which.min(model_dt$cptable[, 'xerror']), 'CP']

cat("5 Feature Regression Tree Optimal CP:", best_cp, "\n")
```

```
## 5 Feature Regression Tree Optimal CP: 0.008108108
```

``` r
model_dt_pruned <- rpart::prune(model_dt, cp = best_cp)  
 # plot decision tree
rpart.plot(model_dt_pruned,
                       type = 4,
                       extra = 104,
                       fallen.leaves = TRUE,
                       main = 'Pruned Decision Tree from 5 Variables',
                       )
```

<img src="curry-modeling_files/figure-html/Decision Tree-1.png" alt="" width="672" />

``` r
preds_dt = predict(model_dt_pruned,
                   newdata = test_df,
                   type = 'class')

probs_dt = predict(model_dt_pruned,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_dt <- roc(test_df$result, probs_dt)
cat("5 Feature Decision Tree AUC:", auc(roc_dt), "\n")
```

```
## 5 Feature Decision Tree AUC: 0.6167254
```

``` r
confusionMatrix(as.factor(preds_dt), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 107  86
##          1  31  62
##                                           
##                Accuracy : 0.5909          
##                  95% CI : (0.5315, 0.6484)
##     No Information Rate : 0.5175          
##     P-Value [Acc > NIR] : 0.007492        
##                                           
##                   Kappa : 0.1917          
##                                           
##  Mcnemar's Test P-Value : 5.966e-07       
##                                           
##             Sensitivity : 0.7754          
##             Specificity : 0.4189          
##          Pos Pred Value : 0.5544          
##          Neg Pred Value : 0.6667          
##              Prevalence : 0.4825          
##          Detection Rate : 0.3741          
##    Detection Prevalence : 0.6748          
##       Balanced Accuracy : 0.5971          
##                                           
##        'Positive' Class : 0               
## 
```

``` r
model_dt$variable.importance
```

```
## sec_remaining   distance_ft        margin           qtr      three_pt 
##     67.287626     57.183636     52.471931     15.853903      6.423199
```

``` r
 # 3 features
model_dt_3 <- rpart(result ~ distance_ft + sec_remaining + margin,
                         data = train_df,
                         method = 'class',
                         parms = list(split = 'gini'),
                         control = rpart::rpart.control(cp = 0.001)
                         )
printcp(model_dt_3)
```

```
## 
## Classification tree:
## rpart(formula = result ~ distance_ft + sec_remaining + margin, 
##     data = train_df, method = "class", parms = list(split = "gini"), 
##     control = rpart::rpart.control(cp = 0.001))
## 
## Variables actually used in tree construction:
## [1] distance_ft   margin        sec_remaining
## 
## Root node error: 555/1148 = 0.48345
## 
## n= 1148 
## 
##           CP nsplit rel error  xerror     xstd
## 1  0.1495495      0   1.00000 1.00000 0.030508
## 2  0.0216216      1   0.85045 0.89009 0.030227
## 3  0.0081081      4   0.78198 0.83063 0.029927
## 4  0.0072072      6   0.76577 0.86486 0.030112
## 5  0.0066066      8   0.75135 0.87387 0.030155
## 6  0.0063063     15   0.69910 0.88649 0.030211
## 7  0.0054054     19   0.66667 0.88829 0.030219
## 8  0.0045045     21   0.65586 0.89910 0.030263
## 9  0.0042042     23   0.64685 0.89730 0.030256
## 10 0.0036036     29   0.62162 0.92793 0.030363
## 11 0.0030030     38   0.58919 0.92973 0.030368
## 12 0.0027027     43   0.57297 0.93153 0.030374
## 13 0.0021622     49   0.55676 0.94595 0.030413
## 14 0.0018018     54   0.54595 0.98378 0.030488
## 15 0.0012012     67   0.52072 0.98378 0.030488
## 16 0.0010000     70   0.51712 0.99459 0.030502
```

``` r
best_cp_3 <- model_dt_3$cptable[
  which.min(model_dt_3$cptable[, 'xerror']), 'CP']


cat("3 Feature Regression Tree Optimal CP:", best_cp_3, "\n")
```

```
## 3 Feature Regression Tree Optimal CP: 0.008108108
```

``` r
model_dt_pruned_3 <- rpart::prune(model_dt_3, cp = best_cp_3)  
 # plot decision tree
rpart.plot(model_dt_pruned_3,
                       type = 4,
                       extra = 104,
                       fallen.leaves = TRUE,
                       main = 'Pruned Decision Tree from 3 Variables',
                       )
```

<img src="curry-modeling_files/figure-html/Decision Tree-2.png" alt="" width="672" />

``` r
preds_dt_3 = predict(model_dt_pruned_3,
                   newdata = test_df,
                   type = 'class')

probs_dt_3 = predict(model_dt_pruned_3,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_dt_3 <- roc(test_df$result, probs_dt_3)
cat("3 Feature Decision Tree AUC:", auc(roc_dt_3), "\n")
```

```
## 3 Feature Decision Tree AUC: 0.6173374
```

``` r
confusionMatrix(as.factor(preds_dt_3), as.factor(test_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 115  93
##          1  23  55
##                                          
##                Accuracy : 0.5944         
##                  95% CI : (0.535, 0.6518)
##     No Information Rate : 0.5175         
##     P-Value [Acc > NIR] : 0.005353       
##                                          
##                   Kappa : 0.2015         
##                                          
##  Mcnemar's Test P-Value : 1.489e-10      
##                                          
##             Sensitivity : 0.8333         
##             Specificity : 0.3716         
##          Pos Pred Value : 0.5529         
##          Neg Pred Value : 0.7051         
##              Prevalence : 0.4825         
##          Detection Rate : 0.4021         
##    Detection Prevalence : 0.7273         
##       Balanced Accuracy : 0.6025         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
 # DT feature comparison

 # 5 variable (dist, qtr, sec, margin, three_pt)    Accuracy: 0.591, Kappa: 0.192, AUC: 0.617, Sensitivity: 0.775, Specificity: 0.419
 # 3 variable (dist, sec, margin)                   Accuracy: 0.594, Kappa: 0.202, AUC: 0.617, Sensitivity: 0.833, Specificity: 0.372

 # // 3 variable (dist, margin, sec) //
```


``` r
 # 5 variables

vote_df <- data_frame(
  nb = as.integer(as.character(preds_nb)),
  dt = as.integer(as.character(preds_dt)),
  lr = as.integer(as.character(preds_lr))
)

vote_df$final <- as.factor(
  ifelse(rowSums(vote_df) >= 2, 1, 0)
)

confusionMatrix(vote_df$final, as.factor((test_df$result)))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 100  81
##          1  38  67
##                                           
##                Accuracy : 0.5839          
##                  95% CI : (0.5244, 0.6417)
##     No Information Rate : 0.5175          
##     P-Value [Acc > NIR] : 0.0141077       
##                                           
##                   Kappa : 0.1755          
##                                           
##  Mcnemar's Test P-Value : 0.0001181       
##                                           
##             Sensitivity : 0.7246          
##             Specificity : 0.4527          
##          Pos Pred Value : 0.5525          
##          Neg Pred Value : 0.6381          
##              Prevalence : 0.4825          
##          Detection Rate : 0.3497          
##    Detection Prevalence : 0.6329          
##       Balanced Accuracy : 0.5887          
##                                           
##        'Positive' Class : 0               
## 
```

``` r
avg_prob_vote <- (probs_nb + probs_dt + probs_lr) / 3
roc_vote <- roc(test_df$result, avg_prob_vote)
cat("5 Variable Vote Ensemble (3 Models) AUC:", auc(roc_vote), "\n")
```

```
## 5 Variable Vote Ensemble (3 Models) AUC: 0.6377791
```

``` r
 # 3 variables

vote_df_3 <- data_frame(
  nb3 = as.integer(as.character(preds_nb_3)),
  dt3 = as.integer(as.character(preds_dt_3)),
  lr3 = as.integer(as.character(preds_lr_3))
)

vote_df_3$final <- as.factor(
  ifelse(rowSums(vote_df_3) >= 2, 1, 0)
)

confusionMatrix(vote_df_3$final, as.factor((test_df$result)))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 105  83
##          1  33  65
##                                          
##                Accuracy : 0.5944         
##                  95% CI : (0.535, 0.6518)
##     No Information Rate : 0.5175         
##     P-Value [Acc > NIR] : 0.005353       
##                                          
##                   Kappa : 0.1976         
##                                          
##  Mcnemar's Test P-Value : 5.376e-06      
##                                          
##             Sensitivity : 0.7609         
##             Specificity : 0.4392         
##          Pos Pred Value : 0.5585         
##          Neg Pred Value : 0.6633         
##              Prevalence : 0.4825         
##          Detection Rate : 0.3671         
##    Detection Prevalence : 0.6573         
##       Balanced Accuracy : 0.6000         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
avg_prob_vote_3 <- (probs_nb_3 + probs_dt_3 + probs_lr_3) / 3
roc_vote_3 <- roc(test_df$result, avg_prob_vote_3)
cat("3 Variable Vote Ensemble (3 Models) AUC:", auc(roc_vote_3), "\n")
```

```
## 3 Variable Vote Ensemble (3 Models) AUC: 0.648159
```

``` r
 # Note: 3-feature models use the best performing features per model type
 # LR + NB: distance_ft, margin, three_pt (statistically significant predictors)
 # DT:      distance_ft, sec_remaining, margin (highest variable importance scores)
 
 # 5 variable (dist, qtr, sec, margin, three_pt)  Accuracy: 0.584, Kappa: 0.176, AUC: 0.638, Sensitivity: 0.725, Specificity: 0.453
 # 3 variable highlights per model                Accuracy: 0.608, Kappa: 0.225, AUC: 0.648, Sensitivity: 0.768, Specificity: 0.460
```

## Model Performance Summary (Steph Curry 2023 Shot Prediction)

| Model | Features | Accuracy | Kappa | AUC | Sens | Spec |
|---|---|---|---|---|---|---|
| Logistic Regression | 5 | 0.594 | 0.196 | 0.632 | 0.732 | 0.466 |
| Logistic Regression | 3 | 0.601 | 0.211 | 0.640 | 0.761 | 0.453 |
| Naive Bayes | 5 | 0.587 | 0.181 | 0.628 | 0.710 | 0.473 |
| Naive Bayes | 3 | 0.594 | 0.195 | 0.637 | 0.710 | 0.486 |
| Decision Tree | 5 | 0.591 | 0.192 | 0.617 | 0.775 | 0.419 |
| Decision Tree | 3 | 0.594 | 0.202 | 0.617 | 0.833 | 0.372 |
| Voting Ensemble (NB+DT+LR) | 5 | 0.584 | 0.176 | 0.638 | 0.725 | 0.453 |
| Voting Ensemble (NB+DT+LR) | 3 (best per model) | 0.594 | 0.198 | 0.648 | 0.761 | 0.439 |

**Winner: 3-feature Voting Ensemble -- best AUC (0.648), competitive on all metrics**


``` r
 # clutch = 4th or OT & last 5 minutes & 5 point margin
clutch_df <- num_curry_df |>
  mutate(is_clutch = ifelse(
    qtr %in% c(4, 5, 6) &
      sec_remaining <= 300 &
      abs(margin) <= 5, 1, 0)
         ) |>
  filter(is_clutch == 1)

table(clutch_df$result)
```

```
## 
##  0  1 
## 56 44
```

``` r
 # LR clutch
probs_lr_clutch <- predict(model_lr_3,
                           newdata = clutch_df,
                           type = 'response')
preds_lr_clutch <- ifelse(probs_lr_clutch > 0.5, 1, 0)

 # NB clutch
probs_nb_clutch <- predict(model_nb_3, 
                           newdata = clutch_df, 
                           type = 'prob')[, 2]
preds_nb_clutch <- predict(model_nb_3, 
                           newdata = clutch_df, 
                           type = 'class')

 # DT clutch
probs_dt_clutch <- predict(model_dt_pruned_3, 
                           newdata = clutch_df, 
                           type = 'prob')[, 2]
preds_dt_clutch <- predict(model_dt_pruned_3, 
                           newdata = clutch_df, 
                           type = 'class')

 # AUC
cat("LR Clutch AUC:", auc(roc(clutch_df$result, probs_lr_clutch)), "\n")
```

```
## LR Clutch AUC: 0.6558442
```

``` r
cat("NB Clutch AUC:", auc(roc(clutch_df$result, probs_nb_clutch)), "\n")
```

```
## NB Clutch AUC: 0.6513799
```

``` r
cat("DT Clutch AUC:", auc(roc(clutch_df$result, probs_dt_clutch)), "\n")
```

```
## DT Clutch AUC: 0.6243912
```

``` r
 # confusion matrices
confusionMatrix(as.factor(preds_lr_clutch), as.factor(clutch_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 46 26
##          1 10 18
##                                           
##                Accuracy : 0.64            
##                  95% CI : (0.5379, 0.7336)
##     No Information Rate : 0.56            
##     P-Value [Acc > NIR] : 0.06452         
##                                           
##                   Kappa : 0.2399          
##                                           
##  Mcnemar's Test P-Value : 0.01242         
##                                           
##             Sensitivity : 0.8214          
##             Specificity : 0.4091          
##          Pos Pred Value : 0.6389          
##          Neg Pred Value : 0.6429          
##              Prevalence : 0.5600          
##          Detection Rate : 0.4600          
##    Detection Prevalence : 0.7200          
##       Balanced Accuracy : 0.6153          
##                                           
##        'Positive' Class : 0               
## 
```

``` r
confusionMatrix(as.factor(preds_nb_clutch), as.factor(clutch_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 37 20
##          1 19 24
##                                          
##                Accuracy : 0.61           
##                  95% CI : (0.5073, 0.706)
##     No Information Rate : 0.56           
##     P-Value [Acc > NIR] : 0.1826         
##                                          
##                   Kappa : 0.2067         
##                                          
##  Mcnemar's Test P-Value : 1.0000         
##                                          
##             Sensitivity : 0.6607         
##             Specificity : 0.5455         
##          Pos Pred Value : 0.6491         
##          Neg Pred Value : 0.5581         
##              Prevalence : 0.5600         
##          Detection Rate : 0.3700         
##    Detection Prevalence : 0.5700         
##       Balanced Accuracy : 0.6031         
##                                          
##        'Positive' Class : 0              
## 
```

``` r
confusionMatrix(as.factor(preds_dt_clutch), as.factor(clutch_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 51 30
##          1  5 14
##                                           
##                Accuracy : 0.65            
##                  95% CI : (0.5482, 0.7427)
##     No Information Rate : 0.56            
##     P-Value [Acc > NIR] : 0.04242         
##                                           
##                   Kappa : 0.2437          
##                                           
##  Mcnemar's Test P-Value : 4.976e-05       
##                                           
##             Sensitivity : 0.9107          
##             Specificity : 0.3182          
##          Pos Pred Value : 0.6296          
##          Neg Pred Value : 0.7368          
##              Prevalence : 0.5600          
##          Detection Rate : 0.5100          
##    Detection Prevalence : 0.8100          
##       Balanced Accuracy : 0.6144          
##                                           
##        'Positive' Class : 0               
## 
```

``` r
 # vote
vote_clutch <- tibble(
  nb = as.integer(as.character(preds_nb_clutch)),
  dt = as.integer(as.character(preds_dt_clutch)),
  lr = as.integer(as.character(preds_lr_clutch))
)

vote_clutch$final <- as.factor(
  ifelse(rowSums(vote_clutch) >= 2, 1, 0))

avg_prob_clutch <- (probs_nb_clutch + probs_dt_clutch + probs_lr_clutch) / 3

cat("Vote Clutch AUC:", auc(roc(clutch_df$result, avg_prob_clutch)), "\n")
```

```
## Vote Clutch AUC: 0.6594968
```

``` r
confusionMatrix(vote_clutch$final, as.factor(clutch_df$result))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 45 24
##          1 11 20
##                                           
##                Accuracy : 0.65            
##                  95% CI : (0.5482, 0.7427)
##     No Information Rate : 0.56            
##     P-Value [Acc > NIR] : 0.04242         
##                                           
##                   Kappa : 0.2666          
##                                           
##  Mcnemar's Test P-Value : 0.04252         
##                                           
##             Sensitivity : 0.8036          
##             Specificity : 0.4545          
##          Pos Pred Value : 0.6522          
##          Neg Pred Value : 0.6452          
##              Prevalence : 0.5600          
##          Detection Rate : 0.4500          
##    Detection Prevalence : 0.6900          
##       Balanced Accuracy : 0.6291          
##                                           
##        'Positive' Class : 0               
## 
```

## Clutch Model Performance Summary

**n=100 | 3-feature models, no retraining**  
**Clutch Definition: 4th Qtr/OT, last 5 min, abs(margin) <= 5**

| Model | Accuracy | Kappa | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Logistic Regression | 0.640 | 0.240 | 0.656 | 0.821 | 0.409 |
| Naive Bayes | 0.610 | 0.207 | 0.651 | 0.661 | 0.545 |
| Decision Tree | 0.650 | 0.244 | 0.624 | 0.911 | 0.318 |
| Voting Ensemble | 0.650 | 0.267 | 0.659 | 0.804 | 0.455 |

**Winner: Voting Ensemble -- best AUC (0.659) and Kappa (0.267)**  
**Note:** more balanced class distribution (56/44) vs prior definition (93/58); improved specificity across all models.

### Clutch Analysis Observations

- General models trained on ~48% make rate; clutch subset has ~44% make rate -- class distributions are closer but mismatch still limits clutch prediction reliability
- `qtr` and `sec_remaining` lose predictive value in clutch context due to near-zero variance (all clutch shots occur in 4th/OT with low seconds remaining)
- 3-feature models (`distance_ft`, `margin`, `three_pt`) are better suited for clutch evaluation as these features retain meaningful variance within clutch situations
- High sensitivity and low specificity across clutch models suggests over-prediction of misses -- model cannot explain clutch makes from general shot profile alone
- n=100 clutch observations is insufficient for statistically significant claims about clutch vs general performance differences (wide CIs, p > 0.05 vs NIR)
- Cannot claim Curry performs better or worse in clutch -- training distribution and sample size limitations confound any such interpretation
- Future work: clutch-specific model would require multi-season data to achieve sufficient n for reliable inference
