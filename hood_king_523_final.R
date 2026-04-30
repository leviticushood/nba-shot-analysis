##------------------------------------------------------------------------------
## Final Project - Code Submission
## INFO 523 - Data Mining and Discovery
## Levi Hood & Andrew King (Team 7)
## 30 April 2026
##------------------------------------------------------------------------------

##------------------- NBA Player Shot Analysis & Modeling ----------------------

# This project uses a public kaggle dataset comprised of three NBA shot 
# performance stats from the 2023 NBA season (LeBron, Curry, & Harden). 
# The project answers these three questions:

# 1. Do players have specific hot-zones where they have a significant advantage? 
#    Where are these hot-zones?
# 2. Who is the most consistent shooter?
# 3. Does the given player tend to perform better under pressure (or clutch up) 
#    compared to their overall average?

# Source: https://www.kaggle.com/datasets/dhavalrupapara/nba-2023-player-shot-dataset?select=2_james_harden_shot_chart_2023.csv

# Variables: 16
# Continuous: top, left, time_remaining, distance_ft, player_team_score, opponent_team_score
# Date: date
# Ordinal: qtr, season, shot_type
# Binary: result, lead
# Nominal: player, opponent, team, color

# Observations: 3,992

# The code will be sectioned by question proposed.

#--------------------------------- Question 1 ----------------------------------
# 1. Do players have specific hot-zones where they have a significant advantage? 
#    Where are these hot-zones?

# The following code section contains suggested visualizations to supplement 
# article and research questions:

# load in necessary libraries
library(ggplot2)
library(tidyverse)
library(sportyR)
library(ggridges)
library(lubridate)
library(dplyr)

# read in kaggle csv
shot_data <- read.csv('/cloud/project/data/processed-data/combined_player_shot_chart_2023.csv')

# box plots (3) of successful field goal distance by player w/ 3 point delineation
ggplot(shot_data |> 
         filter(result == 'TRUE'),            # filter for only successful shots
       aes(x = distance_ft, y = player,
           fill = player)) +
  annotate('rect',                            # add brown rect for 3 point range
           xmin = 22, xmax = Inf,
           ymin = -Inf, ymax = Inf,
           fill = '#AD9168',
           color = 'black',
           alpha = .5) +
  geom_boxplot() +
  scale_fill_manual(values = c('LeBron' = '#552583',
                               'Curry' = '#1D428A',
                               'Harden' = '#C8102E')) +      # team colors
  geom_text(aes(label = player), 
            color = 'white',
            x = 12,
            fontface = 'bold',
            size = 5,
            check_overlap = TRUE) +
  annotate('text',
           x = 29.5,y = 3,
           color = '#6A5639',
           vjust = -3,
           label = '(3pt Range)',
           fontface = 'bold',
           size = 5) +
  labs(x = 'Distance from Hoop (ft)', y = NULL,
       title = 'Successful Field Goal Distance by Player',
       subtitle = '2023 NBA Season',
       caption = 'Data Source: Kaggle (2024)') +
  theme_bw() +
  theme(axis.text.y = element_blank(),
        legend.position = 'none',
        axis.ticks.y = element_blank()) +
  scale_x_continuous(breaks = seq(0, 40, by = 5))

# create ridgeline df
ridgeline_data <- shot_data |> 
  select('player', 'qtr', 'time_remaining', 'result') |> 
  filter(qtr %in% c('1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr')) |> 
  filter(result == 'TRUE') |> 
  mutate('sec_remaining' = as.numeric(ms(time_remaining), 'seconds'))

# create ridgeline plot to display shots made by quarter per player
ggplot(ridgeline_data, aes(x = sec_remaining, y = player,
                           fill = player,
                           alpha = 0.6)) +
  geom_density_ridges(rel_min_height = 0.01,
                      position = 'identity') +
  scale_x_reverse(limits = c(720, 0), breaks = seq(720,0,-60)) +
  labs(title = 'Made Shots by Quarter',
       subtitle = 'When Each Player Scores as the Clock Winds Down',
       x = 'Seconds Remaining in Quarter',
       caption = 'Data Source: Kaggle (2024)') +
  scale_fill_manual(values = c('LeBron' = '#552583',
                               'Curry' = '#1D428A',
                               'Harden' = '#C8102E')) +
  facet_wrap(~qtr) +
  theme_ridges() +
  theme(axis.title.y = element_blank(),
        legend.position = 'none')

#create hotzone df
hotzone_data <- shot_data |> 
  select('player','top','left', 'result') |> 
  filter(result == 'TRUE')

# create the bball court background using sportyr for geospatial shot distribtuion
court <- geom_basketball(league = 'NBA',
                         court_units = 'feet',
                         display_range = 'offense'
)
#adjust the coord system for accuracy
hotzone_data$left_scaled <- (hotzone_data$left - 240) / 10
hotzone_data$top_scaled <- 47 - (hotzone_data$top / 10)
# create heat map style scatterplot over court layout
court + 
  geom_point(data = hotzone_data,
             aes(x = top_scaled, y = left_scaled, 
                 color = player,
                 alpha = 0.2)) +
  scale_color_manual(values = c('LeBron' = '#552583',
                                'Curry'  = '#1D428A',
                                'Harden' = '#C8102E')) +
  facet_wrap(~player) +
  labs(title = 'Geospatial Distribution of Successful Field Goals',
       subtitle = '2023 NBA Season',
       caption = 'Data Source: Kaggle (2024)') +
  theme(legend.position = 'none',
        strip.background = element_blank(),
        strip.text = element_text(face = "bold",
                                  size = 10,
                                  color = "black"))

#--------------------------------- Question 2 ----------------------------------
# 2. Who is the most consistent shooter?



################################################################################
################################################################################





#--------------------------------- Question 3 ----------------------------------
# 3. Does the given player tend to perform better under pressure (or clutch up) 
#    compared to their overall average?

# load in necessary libraries
library(tidyverse)
library(lubridate)
library(GGally)
library(rpart)
library(rpart.plot)
library(caret)
library(naivebayes)
library(pROC)

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


# Exploratory Data Analysis ----------------------------------------------------

# convert qtrs to numeric features
num_curry_df <- curry_df |>
  select(-'player')
num_curry_df$qtr <- as.numeric(factor(num_curry_df$qtr,
                                      levels = c('1st Qtr', '2nd Qtr', '3rd Qtr',
                                                 '4th Qtr', '1st OT', '2nd OT')))
# correlation matrix
ggpairs(num_curry_df, progress = FALSE)


# Train/Test Split -------------------------------------------------------------

set.seed(123)

# stratified 80/20 split
train_idx <- createDataPartition(num_curry_df$result, p = 0.80, list = FALSE)

train_df <- num_curry_df[train_idx, ]
test_df <- num_curry_df[-train_idx, ]

# verify stratification
cat("Train set:", nrow(train_df), "rows |",
    round(mean(train_df$result == 1) * 100, 1), "% Successful Shots\n")
cat("Test set: ", nrow(test_df),  "rows |",
    round(mean(test_df$result  == 1) * 100, 1), "% Successful Shots\n")


# Logistic Regression/Feature Selection --------------------------------------

# 5 features
model_lr <- glm(result ~ distance_ft + qtr + sec_remaining + margin + three_pt,
                data = train_df,
                family = binomial)
# feature significance
summary(model_lr)

probs_lr <- predict(model_lr, newdata = test_df, type = "response")
roc_lr <- roc(test_df$result, probs_lr)
cat("5 Feature Logistic Regression AUC:", auc(roc_lr), "\n")

# confusion matrix
preds_lr <- ifelse(probs_lr > 0.5, 1, 0)
confusionMatrix(as.factor(preds_lr), as.factor(test_df$result))

# 3 features
model_lr_3 <- glm(result ~ distance_ft + margin + three_pt,
                  data = train_df,
                  family = binomial)

probs_lr_3 <- predict(model_lr_3, newdata = test_df, type = "response")
roc_lr_3 <- roc(test_df$result, probs_lr_3)
cat("3 Feature Logistic Regression AUC:", auc(roc_lr_3), "\n")

# confusion matrix
preds_lr_3 <- ifelse(probs_lr_3 > 0.5, 1, 0)
confusionMatrix(as.factor(preds_lr_3), as.factor(test_df$result))

# LR feature selection comparison

# 2 variable (dist, margin)                         Accuracy: 0.59, Kappa: 0.20,  AUC: 0.64, Sensitivity: 0.72, Specificity: 0.47
# 3 variable (dist, margin, three_pt)               Accuracy: 0.60, Kappa: 0.21,  AUC: 0.64, Sensitivity: 0.76, Specificity: 0.45
# 4 variable (dist, qtr, sec, margin)               Accuracy: 0.46, Kappa: -0.07, AUC: 0.53, Sensitivity: 0.55, Specificity: 0.38
# 5 variable (dist, qtr, sec, margin, three_pt)     Accuracy: 0.59, Kappa: 0.20,  AUC: 0.63, Sensitivity: 0.73, Specificity: 0.47

# // 3 variable (dist, margin, three_pt) as baseline going forward //

# Naive Bayes ------------------------------------------------------------------

# 5 features
model_nb <- naive_bayes(as.factor(result) ~ distance_ft + qtr + sec_remaining + margin + three_pt,
                        data = train_df,
                        laplace = 1)
# NB model EDA (top 3 features)
model_nb$tables$distance_ft
model_nb$tables$margin
model_nb$tables$three_pt

preds_nb = predict(model_nb,
                   newdata = test_df,
                   type = 'class')
probs_nb = predict(model_nb,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_nb <- roc(test_df$result, probs_nb)
cat("5 Feature Naive Bayes AUC:", auc(roc_nb), "\n")

confusionMatrix(as.factor(preds_nb), as.factor(test_df$result))

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

confusionMatrix(as.factor(preds_nb_3), as.factor(test_df$result))

# NB feature comparison

# 5 variable (dist, qtr, sec, margin, three_pt)    Accuracy: 0.59, Kappa: 0.18, AUC: 0.63, Sensitivity: 0.71, Specificity: 0.47
# 3 variable (dist, margin, three_pt)              Accuracy: 0.59, Kappa: 0.19, AUC: 0.64, Sensitivity: 0.71, Specificity: 0.49

# // 3 variable (dist, margin, three_pt) //



# Decision Tree ----------------------------------------------------------------

# 5 features
model_dt <- rpart(result ~ distance_ft + qtr + 
                    sec_remaining + margin + three_pt,
                  data = train_df,
                  method = 'class',
                  parms = list(split = 'gini'),
                  control = rpart::rpart.control(cp = 0.001)
)
printcp(model_dt)

best_cp <- model_dt$cptable[
  which.min(model_dt$cptable[, 'xerror']), 'CP']

cat("5 Feature Regression Tree Optimal CP:", best_cp, "\n")

model_dt_pruned <- rpart::prune(model_dt, cp = best_cp)  
# plot decision tree
rpart.plot(model_dt_pruned,
           type = 4,
           extra = 104,
           fallen.leaves = TRUE,
           main = 'Pruned Decision Tree from 5 Variables',
)

preds_dt = predict(model_dt_pruned,
                   newdata = test_df,
                   type = 'class')

probs_dt = predict(model_dt_pruned,
                   newdata = test_df,
                   type = 'prob')[, 2]

roc_dt <- roc(test_df$result, probs_dt)
cat("5 Feature Decision Tree AUC:", auc(roc_dt), "\n")

confusionMatrix(as.factor(preds_dt), as.factor(test_df$result))

model_dt$variable.importance

# 3 features
model_dt_3 <- rpart(result ~ distance_ft + sec_remaining + margin,
                    data = train_df,
                    method = 'class',
                    parms = list(split = 'gini'),
                    control = rpart::rpart.control(cp = 0.001)
)
printcp(model_dt_3)

best_cp_3 <- model_dt_3$cptable[
  which.min(model_dt_3$cptable[, 'xerror']), 'CP']


cat("3 Feature Regression Tree Optimal CP:", best_cp_3, "\n")

model_dt_pruned_3 <- rpart::prune(model_dt_3, cp = best_cp_3)  
# plot decision tree
rpart.plot(model_dt_pruned_3,
           type = 4,
           extra = 104,
           fallen.leaves = TRUE,
           main = 'Pruned Decision Tree from 3 Variables',
)

preds_dt_3 = predict(model_dt_pruned_3,
                     newdata = test_df,
                     type = 'class')

probs_dt_3 = predict(model_dt_pruned_3,
                     newdata = test_df,
                     type = 'prob')[, 2]

roc_dt_3 <- roc(test_df$result, probs_dt_3)
cat("3 Feature Decision Tree AUC:", auc(roc_dt_3), "\n")

confusionMatrix(as.factor(preds_dt_3), as.factor(test_df$result))


# DT feature comparison

# 5 variable (dist, qtr, sec, margin, three_pt)    Accuracy: 0.591, Kappa: 0.192, AUC: 0.617, Sensitivity: 0.775, Specificity: 0.419
# 3 variable (dist, sec, margin)                   Accuracy: 0.594, Kappa: 0.202, AUC: 0.617, Sensitivity: 0.833, Specificity: 0.372

# // 3 variable (dist, margin, sec) //

# Vote -------------------------------------------------------------------------

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

avg_prob_vote <- (probs_nb + probs_dt + probs_lr) / 3
roc_vote <- roc(test_df$result, avg_prob_vote)
cat("5 Variable Vote Ensemble (3 Models) AUC:", auc(roc_vote), "\n")

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

avg_prob_vote_3 <- (probs_nb_3 + probs_dt_3 + probs_lr_3) / 3
roc_vote_3 <- roc(test_df$result, avg_prob_vote_3)
cat("3 Variable Vote Ensemble (3 Models) AUC:", auc(roc_vote_3), "\n")

# Note: 3-feature models use the best performing features per model type
# LR + NB: distance_ft, margin, three_pt (statistically significant predictors)
# DT:      distance_ft, sec_remaining, margin (highest variable importance scores)

# 5 variable (dist, qtr, sec, margin, three_pt)  Accuracy: 0.584, Kappa: 0.176, AUC: 0.638, Sensitivity: 0.725, Specificity: 0.453
# 3 variable highlights per model                Accuracy: 0.608, Kappa: 0.225, AUC: 0.648, Sensitivity: 0.768, Specificity: 0.460

## -----------------------------------------------------------------------------
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
  
# Define Clutch ----------------------------------------------------------------

# clutch = 4th or OT & last 5 minutes & 5 point margin
clutch_df <- num_curry_df |>
  mutate(is_clutch = ifelse(
    qtr %in% c(4, 5, 6) &
      sec_remaining <= 300 &
      abs(margin) <= 5, 1, 0)
  ) |>
  filter(is_clutch == 1)

table(clutch_df$result)

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
cat("NB Clutch AUC:", auc(roc(clutch_df$result, probs_nb_clutch)), "\n")
cat("DT Clutch AUC:", auc(roc(clutch_df$result, probs_dt_clutch)), "\n")

# confusion matrices
confusionMatrix(as.factor(preds_lr_clutch), as.factor(clutch_df$result))
confusionMatrix(as.factor(preds_nb_clutch), as.factor(clutch_df$result))
confusionMatrix(as.factor(preds_dt_clutch), as.factor(clutch_df$result))

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

confusionMatrix(vote_clutch$final, as.factor(clutch_df$result))

## -----------------------------------------------------------------------------
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
  
  ### Clutch Analysis Observations
  
  - General models trained on ~48% make rate; clutch subset has ~44% make rate -- class distributions are closer but mismatch still limits clutch prediction reliability
- `qtr` and `sec_remaining` lose predictive value in clutch context due to near-zero variance (all clutch shots occur in 4th/OT with low seconds remaining)
- 3-feature models (`distance_ft`, `margin`, `three_pt`) are better suited for clutch evaluation as these features retain meaningful variance within clutch situations
- High sensitivity and low specificity across clutch models suggests over-prediction of misses -- model cannot explain clutch makes from general shot profile alone
- n=100 clutch observations is insufficient for statistically significant claims about clutch vs general performance differences (wide CIs, p > 0.05 vs NIR)
- Cannot claim Curry performs better or worse in clutch -- training distribution and sample size limitations confound any such interpretation
- Future work: clutch-specific model would require multi-season data to achieve sufficient n for reliable inference

# AUC Comparison Plot ----------------------------------------------------------

results_df <- tibble(
  model = rep(c('LR', 'NB', 'DT', 'Vote'), 2),
  context = factor(rep(c('General', 'Clutch'), each = 4),
                   levels = c('General', 'Clutch')),
  AUC = c(0.640, 0.637, 0.617, 0.648,
          0.656, 0.651, 0.624, 0.659)
) |>
  mutate(highlight = ifelse(model == 'Vote', 'Winner', 'Other'))

ggplot(results_df,
       aes(x = model, y = AUC, fill = highlight)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label = round(AUC, 3)), 
            vjust = -0.5, 
            size = 3.5,
            fontface = 'bold') +
  geom_hline(yintercept = 0.5, linetype = 'dashed', color = 'red') +
  facet_wrap(~ context) +
  scale_fill_manual(values = c('Winner' = 'darkgreen', 'Other' = 'grey70')) +
  labs(title = 'AUC by Model: General vs Clutch',
       caption = 'Clutch situations are defined as shots taken in the final 5 minutes
       of the 4th quarter or overtime with a score margin of 5 points or fewer.',
       x = 'Model', y = 'AUC') +
  theme_minimal() +
  theme(legend.position = 'none') +
  coord_cartesian(ylim = c(0.5, 0.7))
