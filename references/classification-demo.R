
#' # Classification Demo - INFO523 Data mining
#' 
#' ## 🔧 Setup & Data
#' 
#' 
#' ## Installing & Loading Packages
#' 
## -----------------------------------------------------------------------------
##| label: pkg-install
##| eval: false

# # Run once — installs everything needed for this session
# pkgs <- c(
#   "tidyverse",   # data wrangling & plotting
#   "mlbench",     # Pima dataset
#   "caret",       # unified ML framework
#   "rpart",       # decision trees
#   "rpart.plot",  # tree visualization
#   "e1071",       # Naive Bayes & SVM
#   "C50",         # C5.0 decision tree / rules
#   "OneR",        # rule-based classification
#   "randomForest",# Random Forest
#   "gbm",         # Gradient Boosting
#   "ipred",       # Bagging
#   "pROC",        # ROC-AUC curves
#   "ROSE",        # Random Over-Sampling Examples
#   "themis",      # SMOTE via tidymodels
#   "recipes",     # preprocessing pipelines
#   "yardstick",   # model metrics
#   "kableExtra",   # pretty tables
#   "naivebayes"
# )
# install.packages(setdiff(pkgs, rownames(installed.packages())))

#' 
## -----------------------------------------------------------------------------
##| label: pkg-load
library(tidyverse); library(mlbench); library(caret)
library(rpart); library(rpart.plot); library(e1071)
library(C50); library(OneR); library(randomForest)
library(gbm); library(ipred); library(pROC)
library(ROSE); library(kableExtra); library(naivebayes)
set.seed(42)   # reproducibility throughout

#' 
#' ---
#' 
#' ## Loading the Pima Indians Diabetes Dataset
#' 
## -----------------------------------------------------------------------------
#| label: load-data
data("PimaIndiansDiabetes", package = "mlbench")
df <- PimaIndiansDiabetes

# Quick peek
glimpse(df)

#' 
#' ::: {.fragment}
#' **About the data**
#' 
#' | Feature | Description |
#' |---------|-------------|
#' | `pregnant` | Number of pregnancies |
#' | `glucose` | Plasma glucose (2-hr oral test) |
#' | `pressure` | Diastolic blood pressure (mm Hg) |
#' | `triceps` | Triceps skinfold thickness (mm) |
#' | `insulin` | 2-hr serum insulin (mu U/ml) |
#' | `mass` | Body mass index |
#' | `pedigree` | Diabetes pedigree function |
#' | `age` | Age in years |
#' | `diabetes` | **Target**: `neg` / `pos` |
#' :::
#' 
#' ---
#' 
#' ## Exploratory Data Analysis
#' 
## -----------------------------------------------------------------------------


# Class distribution
table(df$diabetes) |> prop.table() |> round(3)

#' ::: {.callout-tip}
#' **Note the class imbalance:** ~65% negative, ~35% positive.
#' :::
#' 
#' 
#' ---
#' 
#' 
## -----------------------------------------------------------------------------


df |>
  pivot_longer(-diabetes, names_to = "feature", values_to = "value") |>
  ggplot(aes(x = value, fill = diabetes)) +
  geom_density(alpha = 0.55) +
  facet_wrap(~feature, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("#4cc9f0", "#e94560")) +
  labs(title = "Feature Distributions by Diabetes Status",
       fill = "Diabetes") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

#' 
#' 
#' 
#'----------------------------------------------------------------------------
#' 
#' ## Exploratory Data Analysis 
#' 
## -----------------------------------------------------------------------------


# Correlation heatmap (numeric features)
num_df <- df |> mutate(y = as.integer(diabetes == "pos")) |>
  select(-diabetes)

cor_mat <- round(cor(num_df), 2)

# Simple heatmap with ggplot
cor_mat |>
  as.data.frame() |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "corr") |>
  ggplot(aes(var1, var2, fill = corr)) +
  geom_tile(color = "white") +
  geom_text(aes(label = corr), size = 3, color = "white") +
  scale_fill_gradient2(low = "#4cc9f0", mid = "#2d2d44",
                       high = "#e94560", midpoint = 0) +
  labs(title = "Feature Correlation Matrix") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#' 
#' 
#' 
#' ---
#' 
#' ## Train/Test Split
#' 
## -----------------------------------------------------------------------------
#| label: split

# Stratified 70/30 split to preserve class ratios
train_idx <- createDataPartition(df$diabetes, p = 0.70, list = FALSE)

train_df <- df[train_idx, ]
test_df  <- df[-train_idx, ]

# Verify stratification held
cat("Train set:", nrow(train_df), "rows |",
    round(mean(train_df$diabetes == "pos") * 100, 1), "% positive\n")
cat("Test set: ", nrow(test_df),  "rows |",
    round(mean(test_df$diabetes  == "pos") * 100, 1), "% positive\n")

#' 
#' ::: {.callout-tip}
#' `createDataPartition()` from `caret` does **stratified** sampling — it maintains the class ratio in both splits. Plain `sample()` does not!
#' :::
#' 
#' ---
#' 
#' ## 🌳 Decision Trees
#' 
#' ---
#' 
#' ## Fitting the Decision Tree
#' 
## -----------------------------------------------------------------------------
#| label: dt-fit

# Fit a full (unpruned) decision tree using Gini impurity
dt_model <- rpart(
  diabetes ~ .,
  data    = train_df,
  method  = "class",
  parms   = list(split = "gini"),
  control = rpart.control(minsplit = 5, cp = 0.001, maxdepth = 8)
)

# Summary of splits
printcp(dt_model)

#' 
#' ---
#' 
#' ## Visualizing the Decision Tree
#' 
## -----------------------------------------------------------------------------
#| label: dt-plot
#| fig-height: 5.5
#| out-width: "100%"
#| output-location: slide

rpart.plot(
  dt_model,
  type   = 4,     # label type: draw labels under nodes
  extra  = 104,   # show % and n in node
  fallen.leaves = TRUE,
  main   = "Decision Tree — Pima Diabetes (unpruned)",
  cex    = 0.5,
  box.palette = c("#4cc9f0", "#e94560")
)

#' 
#' ::: {.callout-note}
#' **Reading the tree:** Each internal node shows the split condition. The leaf nodes show the predicted class, the probability of the positive class, and the % of training samples in that leaf.
#' :::
#' 
#' ---
#' 
#' ## Pruning the Decision Tree
#' 
## -----------------------------------------------------------------------------
#| label: dt-prune
#| fig-height: 4.5
#| output-location: slide

# Cross-validated error vs complexity parameter
plotcp(dt_model)

#' 
#' 
#' ---
#' 
## -----------------------------------------------------------------------------
#| label: dt-prune2
#| fig-height: 4.2
#| output-location: slide

# Select cp that minimises cross-validated error (1-SE rule)
best_cp <- dt_model$cptable[
  which.min(dt_model$cptable[, "xerror"]), "CP"]

cat("Optimal cp:", round(best_cp, 5), "\n")

dt_pruned <- prune(dt_model, cp = best_cp)

rpart.plot(dt_pruned, type = 4, extra = 104, fallen.leaves = TRUE,
           main = "Pruned Decision Tree", cex = 0.70,
           box.palette = c("#4cc9f0", "#e94560"))

#' 
#' ---
#' 
#' ## Decision Tree — Predictions & Confusion Matrix
#' 
## -----------------------------------------------------------------------------
#| label: dt-predict

# Predictions on test set
dt_pred <- predict(dt_pruned, newdata = test_df, type = "class")

# Confusion matrix
cm_dt <- confusionMatrix(dt_pred, test_df$diabetes, positive = "pos")
print(cm_dt)

#' 
## -----------------------------------------------------------------------------
#| label: dt-metrics

# Store metrics for final comparison
results <- tibble(
  Model     = "Decision Tree",
  Accuracy  = cm_dt$overall["Accuracy"],
  Sensitivity = cm_dt$byClass["Sensitivity"],
  Specificity = cm_dt$byClass["Specificity"],
  Kappa     = cm_dt$overall["Kappa"]
)

#' 
#' ---
#' 
#' ## Variable Importance (Decision Tree)
#' 
## -----------------------------------------------------------------------------
#| label: dt-varimp
#| fig-height: 4.5
#| output-location: slide

vi_dt <- dt_pruned$variable.importance

tibble(Feature = names(vi_dt), Importance = vi_dt) |>
  mutate(Feature = fct_reorder(Feature, Importance)) |>
  ggplot(aes(Importance, Feature, fill = Importance)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#4cc9f0", high = "#e94560") +
  labs(title = "Decision Tree — Variable Importance",
       x = "Importance (Gini reduction)", y = NULL) +
  theme_minimal(base_size = 12)

#' 
#' ::: {.callout-tip}
#' `glucose` and `mass` dominate — consistent with clinical knowledge of diabetes risk factors.
#' :::
#' 
#' ---
#' 
#' ## 📊 Naive Bayes Classification
#' 
#' 
#' ---
#' 
#' ## Fitting Naive Bayes
#' 
## -----------------------------------------------------------------------------
#| label: nb-fit

# Naive Bayes (e1071)
nb_model <- naiveBayes(
  diabetes ~ .,
  data   = train_df,
  laplace = 1    # Laplace smoothing
)

# Inspect prior probabilities
nb_model$apriori

#' 
## -----------------------------------------------------------------------------
#| label: nb-tables
# Conditional probability tables for two key features
nb_model$tables$glucose

#' 
#' ::: {.fragment}
## -----------------------------------------------------------------------------
#| label: nb-tables2
nb_model$tables$mass

#' :::
#' 
#' ---
#' 
#' ## Naive Bayes — Predictions
#' 
## -----------------------------------------------------------------------------
#| label: nb-predict

# Hard class predictions
nb_pred  <- predict(nb_model, newdata = test_df, type = "class")

# Probability estimates
nb_probs <- predict(nb_model, newdata = test_df, type = "raw")

head(nb_probs, 8)

#' 
## -----------------------------------------------------------------------------
#| label: nb-cm
cm_nb <- confusionMatrix(nb_pred, test_df$diabetes, positive = "pos")
print(cm_nb)

#' 
#' ---
#' 
#' ## Tuning the Prior — Effect on Sensitivity
#' 
#' ::: {.callout-note}
#' In medical diagnosis, sensitivity (recall for the positive class) is often more important than accuracy. We can tune the classification threshold to trade off precision vs sensitivity.
#' :::
#' 
## -----------------------------------------------------------------------------
#| label: nb-threshold
#| fig-height: 4.0
#| output-location: slide

thresholds <- seq(0.1, 0.9, by = 0.05)

perf <- map_dfr(thresholds, function(thr) {
  pred_class <- factor(
    ifelse(nb_probs[, "pos"] >= thr, "pos", "neg"),
    levels = c("neg", "pos"))
  cm <- confusionMatrix(pred_class, test_df$diabetes, positive = "pos")
  tibble(Threshold = thr,
         Sensitivity = cm$byClass["Sensitivity"],
         Specificity = cm$byClass["Specificity"],
         Accuracy    = cm$overall["Accuracy"])
})

perf |>
  pivot_longer(-Threshold) |>
  ggplot(aes(Threshold, value, color = name)) +
  geom_line(size = 1.1) + geom_point(size = 2) +
  scale_color_manual(values = c("#4cc9f0","#e94560","#ffd60a")) +
  labs(title = "Naive Bayes: Threshold Tuning", y = "Metric", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

#' 
#' ---
#' 
#' ## Naive Bayes — Store Metrics
#' 
## -----------------------------------------------------------------------------
#| label: nb-metrics

results <- bind_rows(results, tibble(
  Model       = "Naive Bayes",
  Accuracy    = cm_nb$overall["Accuracy"],
  Sensitivity = cm_nb$byClass["Sensitivity"],
  Specificity = cm_nb$byClass["Specificity"],
  Kappa       = cm_nb$overall["Kappa"]
))

#' 
#' ---
#' 
#' ## 📋 Rule-Based Classification
#' 
#' ---
#' 
#' 
#' ## OneR — The Simplest Rule Classifier
#' 
## -----------------------------------------------------------------------------
#| label: oner-fit

# OneR: finds the single best attribute to build rules from
oner_model <- OneR(diabetes ~ ., data = train_df, verbose = TRUE)

#' 
## -----------------------------------------------------------------------------
#| label: oner-summary
summary(oner_model)

#' 
#' ::: {.callout-note}
#' OneR discretizes continuous features into bins automatically and selects the feature with the lowest training error. It serves as a useful baseline and sanity check.
#' :::
#' 
#' ---
#' 
#' ## OneR — Evaluation
#' 
## -----------------------------------------------------------------------------
#| label: oner-eval
oner_pred <- predict(oner_model, newdata = test_df)

cm_oner <- confusionMatrix(oner_pred, test_df$diabetes, positive = "pos")
cm_oner$overall[c("Accuracy","Kappa")]
cm_oner$byClass[c("Sensitivity","Specificity","Precision","F1")]

#' 
## -----------------------------------------------------------------------------
#| label: oner-plot
#| fig-height: 3.8
#| output-location: slide

plot(oner_model, main = "OneR — Decision Boundary on Best Feature")

#' 
#' ---
#' 
#' ## C5.0 Rule-Based Classifier
#' 
## -----------------------------------------------------------------------------
#| label: c50-rules-fit

# C5.0 with rules=TRUE extracts a propositional rule set
c50_rules <- C5.0(
  x       = train_df[, -9],   # features
  y       = train_df$diabetes, # target
  rules   = TRUE,              # generate rules, not a tree
  control = C5.0Control(
    noGlobalPruning = FALSE,
    winnow          = TRUE,    # auto feature selection; irrelevent predictors discarded
    CF              = 0.25     # confidence factor (pruning aggressiveness)
  )
)

summary(c50_rules)

#' 
#' ---
#' 
#' ## C5.0 Rules — Predictions & Evaluation
#' 
## -----------------------------------------------------------------------------
#| label: c50-rules-predict
c50r_pred <- predict(c50_rules, newdata = test_df[, -9])

cm_c50r <- confusionMatrix(c50r_pred, test_df$diabetes, positive = "pos")
print(cm_c50r)

#' 
## -----------------------------------------------------------------------------
#| label: c50-metrics
results <- bind_rows(results,
  tibble(
    Model       = "OneR",
    Accuracy    = cm_oner$overall["Accuracy"],
    Sensitivity = cm_oner$byClass["Sensitivity"],
    Specificity = cm_oner$byClass["Specificity"],
    Kappa       = cm_oner$overall["Kappa"]
  ),
  tibble(
    Model       = "C5.0 Rules",
    Accuracy    = cm_c50r$overall["Accuracy"],
    Sensitivity = cm_c50r$byClass["Sensitivity"],
    Specificity = cm_c50r$byClass["Specificity"],
    Kappa       = cm_c50r$overall["Kappa"]
  )
)

#' 
#' ---
#' 
#' ## C5.0 Boosted Rules (Brief Preview)
#' 
## -----------------------------------------------------------------------------
#| label: c50-boost

# C5.0 supports built-in AdaBoost via the trials argument
c50_boosted <- C5.0(
  x       = train_df[, -9],
  y       = train_df$diabetes,
  rules   = TRUE,
  trials  = 20   # 20 boosting iterations
)

c50b_pred <- predict(c50_boosted, newdata = test_df[, -9])
cm_c50b   <- confusionMatrix(c50b_pred, test_df$diabetes, positive = "pos")

cat("Boosted C5.0 Rules Accuracy:",
    round(cm_c50b$overall["Accuracy"], 4), "\n")
cat("vs. Single  C5.0 Rules Accuracy:",
    round(cm_c50r$overall["Accuracy"], 4), "\n")

#' 
#' ::: {.callout-tip}
#' Boosting C5.0 rules often provides a significant accuracy improvement at the cost of interpretability — the rule set becomes an *ensemble* of rule sets.
#' :::
#' 
#' ---
#' 
#' ## 📐 Model Evaluation
#' 
#' 
#' ---
#' 
#' ## Implementing 10-Fold Stratified CV with caret
#' 
## -----------------------------------------------------------------------------
#| label: cv-setup

# Define 10-fold stratified, repeated 3 times
train_ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 10,      # k = 10
  repeats         = 3,       # repeat 3 times = 30 total evaluations
  classProbs      = TRUE,    # compute class probabilities
  summaryFunction = twoClassSummary,  # ROC, Sens, Spec
  savePredictions = "final"
)

# Decision Tree via caret
cv_dt <- train(
  diabetes ~ .,
  data      = df,      # use full dataset — CV handles splitting
  method    = "rpart",
  metric    = "ROC",   # optimise by AUC
  trControl = train_ctrl,
  tuneLength = 10       # try 10 cp values
)

cv_dt

#' 
#' ---
#' 
#' ## CV Results for All Three Base Models
#' 
## -----------------------------------------------------------------------------
#| label: cv-all
#| cache: true

# Naive Bayes via caret
cv_nb <- train(
  diabetes ~ .,
  data      = df,
  method    = "naive_bayes",
  metric    = "ROC",
  trControl = train_ctrl
)

# C5.0 via caret
cv_c50 <- train(
  diabetes ~ .,
  data      = df,
  method    = "C5.0",
  metric    = "ROC",
  trControl = train_ctrl,
  tuneGrid  = expand.grid(trials = c(1, 5, 10, 20),
                           model  = "rules",
                           winnow = TRUE)
)

#' 
## -----------------------------------------------------------------------------
#| label: cv-compare
#| fig-height: 3.8
#| output-location: slide

cv_results <- resamples(list(
  `Decision Tree` = cv_dt,
  `Naive Bayes`   = cv_nb,
  `C5.0 Rules`    = cv_c50
))

bwplot(cv_results, metric = "ROC",
       main = "10-Fold CV (repeated 3x) — ROC-AUC Distribution")

#' 
#' ---
#' 
#' ## Bootstrap Resampling
#' 
## -----------------------------------------------------------------------------
#| label: boot-concept

# Bootstrap: sample n observations WITH replacement, evaluate on OOB
boot_ctrl <- trainControl(
  method          = "boot",
  number          = 200,     # 200 bootstrap samples
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)

cv_dt_boot <- train(
  diabetes ~ .,
  data      = df,
  method    = "rpart",
  metric    = "ROC",
  trControl = boot_ctrl,
  tuneLength = 5
)

cat("Bootstrap estimate — ROC AUC:",
    round(max(cv_dt_boot$results$ROC), 4), "\n")
cat("Bootstrap 95% CI approximation:\n")
quantile(cv_dt_boot$resample$ROC, probs = c(0.025, 0.975))

#' 
#' ::: {.callout-note}
#' **Bootstrap vs k-Fold:** Bootstrap tends to be *pessimistically biased* (~0.632 correction helps). Repeated k-Fold CV is generally preferred for model selection; bootstrap excels at confidence intervals.
#' :::
#' 
#' ---
#' 
#' ## Significance Testing — McNemar's Test
#' 
#' ::: {.callout-important}
#' **Goal:** Test whether two classifiers make *statistically significantly* different errors on the same test set — not just different accuracy values.
#' :::
#' 
## -----------------------------------------------------------------------------
#| label: mcnemar

# Predict on test set with both models
dt_test_pred  <- predict(cv_dt,  newdata = test_df)
nb_test_pred  <- predict(cv_nb,  newdata = test_df)
c50_test_pred <- predict(cv_c50, newdata = test_df)

true_labels <- test_df$diabetes

# Contingency table: DT correct? vs NB correct?
dt_correct  <- dt_test_pred  == true_labels
nb_correct  <- nb_test_pred  == true_labels

cont_table <- table(
  `DT Correct`  = dt_correct,
  `NB Correct`  = nb_correct
)

print(cont_table)

# McNemar's test
mc_test <- mcnemar.test(cont_table, correct = TRUE)
mc_test

#' 
#' ---
#' 
#' ## Interpreting McNemar's Test
#' 
## -----------------------------------------------------------------------------
#| label: mcnemar-interp

cat("McNemar statistic:", round(mc_test$statistic, 4), "\n")
cat("p-value:", round(mc_test$p.value, 4), "\n\n")

if (mc_test$p.value < 0.05) {
  cat("✓ Significant difference between DT and NB (p < 0.05)\n")
  cat("  The classifiers make DIFFERENT error patterns.\n")
} else {
  cat("✗ No significant difference between DT and NB (p ≥ 0.05)\n")
  cat("  The classifiers make SIMILAR error patterns.\n")
}

# Also compare DT vs C5.0
dt_c50_cont <- table(dt_correct, c50_test_pred == true_labels)
mc_dt_c50   <- mcnemar.test(dt_c50_cont, correct = TRUE)
cat("\nDT vs C5.0 p-value:", round(mc_dt_c50$p.value, 4), "\n")

#' 
#' ::: {.callout-tip}
#' McNemar's test uses the *discordant* cells — cases where one model is right and the other is wrong. It's specifically designed for paired binary classification results.
#' :::
#' 
#' ---
#' 
#' ## Paired t-Test on CV Folds
#' 
## -----------------------------------------------------------------------------
#| label: paired-ttest

# Extract per-fold ROC from resamples
cv_roc <- cv_results$values[, grepl("ROC", colnames(cv_results$values))]

dt_roc  <- cv_roc[, "Decision Tree~ROC"]
nb_roc  <- cv_roc[, "Naive Bayes~ROC"]
c50_roc <- cv_roc[, "C5.0 Rules~ROC"]

# Paired t-test: DT vs NB across same folds
t_test_dt_nb <- t.test(dt_roc, nb_roc, paired = TRUE, alternative = "two.sided")
cat("DT vs NB Paired t-test:\n")
print(t_test_dt_nb)

# DT vs C5.0
t_test_dt_c50 <- t.test(dt_roc, c50_roc, paired = TRUE)
cat("\nDT vs C5.0 Paired t-test p-value:", round(t_test_dt_c50$p.value, 4), "\n")

#' 
#' ---
#' 
#' ## ROC Curves — Building & Interpreting
#' 
## -----------------------------------------------------------------------------
#| label: roc-build

# Get probability scores from each model
dt_probs  <- predict(cv_dt,  newdata = test_df, type = "prob")[, "pos"]
nb_probs2 <- predict(cv_nb,  newdata = test_df, type = "prob")[, "pos"]
c50_probs <- predict(cv_c50, newdata = test_df, type = "prob")[, "pos"]

# Build ROC objects
roc_dt  <- roc(test_df$diabetes, dt_probs,  levels = c("neg","pos"), direction = "<")
roc_nb  <- roc(test_df$diabetes, nb_probs2, levels = c("neg","pos"), direction = "<")
roc_c50 <- roc(test_df$diabetes, c50_probs, levels = c("neg","pos"), direction = "<")

cat("Decision Tree  AUC:", round(auc(roc_dt),  4), "\n")
cat("Naive Bayes    AUC:", round(auc(roc_nb),  4), "\n")
cat("C5.0 Rules     AUC:", round(auc(roc_c50), 4), "\n")

#' 
#' ---
#' 
#' ## ROC Curves — Visualisation
#' 
## -----------------------------------------------------------------------------
#| label: roc-plot
#| fig-height: 5.2
#| out-width: "90%"
#| output-location: slide

par(bg = "#1e1e2e", col.axis = "white", col.lab = "white",
    col.main = "white", fg = "white")

plot(roc_dt,  col = "#4cc9f0", lwd = 2.5,
     main = "ROC Curves — Base Classifiers")
plot(roc_nb,  col = "#e94560", lwd = 2.5, add = TRUE)
plot(roc_c50, col = "#ffd60a", lwd = 2.5, add = TRUE)
abline(0, 1, lty = 2, col = "#ffffff66", lwd = 1.2)

legend("bottomright",
       legend = c(
         paste0("Decision Tree  AUC=", round(auc(roc_dt),  3)),
         paste0("Naive Bayes    AUC=", round(auc(roc_nb),  3)),
         paste0("C5.0 Rules     AUC=", round(auc(roc_c50), 3))
       ),
       col = c("#4cc9f0","#e94560","#ffd60a"), lwd = 2.5, bty = "n",
       text.col = "white", cex = 0.9)

#' 
#' ---
#' 
#' ## DeLong Test — Comparing AUCs Statistically
#' 
## -----------------------------------------------------------------------------
#| label: delong

# pROC implements DeLong's test for comparing paired AUC estimates
roc_test_dt_nb  <- roc.test(roc_dt, roc_nb,  method = "delong")
roc_test_dt_c50 <- roc.test(roc_dt, roc_c50, method = "delong")

cat("DeLong Test: DT vs NB\n")
print(roc_test_dt_nb)

cat("\nDeLong Test: DT vs C5.0\n")
print(roc_test_dt_c50)

#' 
#' ::: {.callout-note}
#' **DeLong's method** tests H₀: AUC₁ = AUC₂ using the covariance structure of the ROC curves. It is the gold-standard test for comparing diagnostic accuracy in medical research.
#' :::
#' 
#' ---
#' 
#' ## 🌲 Ensemble Methods
#' 
#' ### Bagging · Boosting · Random Forest
#' 
#' 
#' ---
#' 
#' ## Bagging with ipred
#' 
## -----------------------------------------------------------------------------
#| label: bagging-fit

bag_model <- bagging(
  diabetes ~ .,
  data   = train_df,
  nbagg  = 100,       # number of bootstrap replicates
  coob   = TRUE       # out-of-bag error estimate
)

cat("OOB Error Estimate:", round(bag_model$err, 4), "\n")
cat("OOB Accuracy:       ", round(1 - bag_model$err, 4), "\n")

#' 
## -----------------------------------------------------------------------------
#| label: bagging-eval
bag_pred <- predict(bag_model, newdata = test_df)

cm_bag <- confusionMatrix(bag_pred, test_df$diabetes, positive = "pos")
cat("Test Accuracy:", round(cm_bag$overall["Accuracy"], 4), "\n")
cat("Sensitivity:  ", round(cm_bag$byClass["Sensitivity"], 4), "\n")
cat("Specificity:  ", round(cm_bag$byClass["Specificity"], 4), "\n")

#' 
#' ---
#' 
#' ## Random Forest
#' 
## -----------------------------------------------------------------------------
#| label: rf-fit1
#| fig-height: 3.8

rf_model <- randomForest(
  diabetes ~ .,
  data     = train_df,
  ntree    = 500,     # number of trees
  mtry     = 3,       # features per split (default √p ≈ 3)
  importance = TRUE   # compute variable importance
)

print(rf_model)

#' 
## -----------------------------------------------------------------------------
#| label: rf-fit2
#| fig-height: 3.8
#| output-location: slide

# OOB error vs number of trees
plot(rf_model, main = "Random Forest — OOB Error vs Number of Trees")
legend("topright", colnames(rf_model$err.rate),
       col = 1:3, lty = 1:3, cex = 0.85)

#' 
#' ---
#' 
#' ## Random Forest — Variable Importance
#' 
## -----------------------------------------------------------------------------
#| label: rf-varimp
#| fig-height: 4.5
#| output-location: slide

varImpPlot(rf_model,
           main = "Random Forest — Variable Importance",
           col  = c("#4cc9f0","#e94560"),
           pch  = 16)

#' 
#' ---
#' 
#' 
## -----------------------------------------------------------------------------
#| label: rf-eval

rf_pred <- predict(rf_model, newdata = test_df)
cm_rf   <- confusionMatrix(rf_pred, test_df$diabetes, positive = "pos")

cat("RF Test Accuracy:", round(cm_rf$overall["Accuracy"], 4), "\n")
cat("Sensitivity:     ", round(cm_rf$byClass["Sensitivity"], 4), "\n")
cat("Specificity:     ", round(cm_rf$byClass["Specificity"], 4), "\n")

#' 
#' ---
#' 
#' ## Tuning Random Forest with caret
#' 
## -----------------------------------------------------------------------------
#| label: rf-tune1
#| cache: true

rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6))

cv_rf <- train(
  diabetes ~ .,
  data      = df,
  method    = "rf",
  metric    = "ROC",
  trControl = train_ctrl,
  tuneGrid  = rf_grid,
  ntree     = 300
)

#' 
## -----------------------------------------------------------------------------
#| label: rf-tune2
#| fig-height: 3.8
#| output-location: slide

plot(cv_rf, main = "Random Forest: mtry Tuning")
cat("Best mtry:", cv_rf$bestTune$mtry, "\n")
cat("Best CV AUC:", round(max(cv_rf$results$ROC), 4), "\n")

#' 
#' ---
#' 
#' ## Gradient Boosting (GBM)
#' 
## -----------------------------------------------------------------------------
#| label: gbm-fit
#| cache: true

# Create numeric target (gbm needs 0/1 for binary)
train_gbm <- train_df |> mutate(target = as.integer(diabetes == "pos"))
test_gbm  <- test_df  |> mutate(target = as.integer(diabetes == "pos"))

gbm_model <- gbm(
  target ~ pregnant + glucose + pressure + triceps +
             insulin + mass + pedigree + age,
  data            = train_gbm,
  distribution    = "bernoulli",  # binary outcome
  n.trees         = 500,
  interaction.depth = 4,          # max depth of each tree
  shrinkage       = 0.05,         # learning rate
  bag.fraction    = 0.7,
  cv.folds        = 5,
  verbose         = FALSE
)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = TRUE)
cat("Best number of trees (CV):", best_iter, "\n")

#' 
#' ---
#' 
#' ## GBM — Evaluation & Variable Importance
#' 
## -----------------------------------------------------------------------------
#| label: gbm-eval
#| fig-height: 4.0

gbm_probs <- predict(gbm_model, newdata = test_gbm,
                     n.trees = best_iter, type = "response")
gbm_pred  <- factor(ifelse(gbm_probs > 0.5, "pos", "neg"),
                    levels = c("neg","pos"))

cm_gbm <- confusionMatrix(gbm_pred, test_df$diabetes, positive = "pos")
cat("GBM Test Accuracy:", round(cm_gbm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:      ", round(cm_gbm$byClass["Sensitivity"], 4), "\n")

#' 
#' ---
#' ## GBM — Evaluation & Variable Importance
#' 
## -----------------------------------------------------------------------------
#| label: gbm-varimp
#| fig-height: 3.6
#| output-location: slide

summary(gbm_model, n.trees = best_iter, plotit = TRUE,
        main = "GBM — Relative Influence")

#' 
#' ---
#' 
#' ## Ensemble Comparison — ROC Curves
#' 
## -----------------------------------------------------------------------------
#| label: ensemble-roc
#| fig-height: 5.0
#| out-width: "90%"
#| output-location: slide

# ROC for ensemble models
rf_probs_test  <- predict(rf_model, newdata = test_df, type = "prob")[,"pos"]
bag_probs_test <- predict(bag_model, newdata = test_df, type = "prob")[,"pos"]
gbm_probs_test <- gbm_probs  # already computed above

roc_rf  <- roc(test_df$diabetes, rf_probs_test,  levels=c("neg","pos"), direction="<")
roc_bag <- roc(test_df$diabetes, bag_probs_test, levels=c("neg","pos"), direction="<")
roc_gbm <- roc(test_df$diabetes, gbm_probs_test, levels=c("neg","pos"), direction="<")

par(bg = "#1e1e2e", col.axis="white", col.lab="white", col.main="white", fg="white")
plot(roc_rf,  col="#4cc9f0", lwd=2.5, main="ROC Curves — Ensemble Models")
plot(roc_bag, col="#e94560", lwd=2.5, add=TRUE)
plot(roc_gbm, col="#ffd60a", lwd=2.5, add=TRUE)
abline(0, 1, lty=2, col="#ffffff55")
legend("bottomright",
  legend = c(paste0("Random Forest AUC=",round(auc(roc_rf), 3)),
             paste0("Bagging       AUC=",round(auc(roc_bag),3)),
             paste0("GBM           AUC=",round(auc(roc_gbm),3))),
  col=c("#4cc9f0","#e94560","#ffd60a"), lwd=2.5, bty="n", text.col="white")

#' 
#' ---
#' 
#' ## Add Ensemble Metrics to Leaderboard
#' 
## -----------------------------------------------------------------------------
#| label: ensemble-metrics

roc_bag2 <- roc(test_df$diabetes, bag_probs_test, levels=c("neg","pos"), direction="<")

results <- bind_rows(results,
  tibble(Model="Bagging",
         Accuracy=cm_bag$overall["Accuracy"],
         Sensitivity=cm_bag$byClass["Sensitivity"],
         Specificity=cm_bag$byClass["Specificity"],
         Kappa=cm_bag$overall["Kappa"]),
  tibble(Model="Random Forest",
         Accuracy=cm_rf$overall["Accuracy"],
         Sensitivity=cm_rf$byClass["Sensitivity"],
         Specificity=cm_rf$byClass["Specificity"],
         Kappa=cm_rf$overall["Kappa"]),
  tibble(Model="GBM",
         Accuracy=cm_gbm$overall["Accuracy"],
         Sensitivity=cm_gbm$byClass["Sensitivity"],
         Specificity=cm_gbm$byClass["Specificity"],
         Kappa=cm_gbm$overall["Kappa"])
)

#' 
#' ---
#' 
#' ## ⚖️ Handling Class Imbalance
#' 
#' ---
#' 
#' ## The Class Imbalance Problem
#' 
## -----------------------------------------------------------------------------
#| label: imbalance-demo
#| fig-height: 4.2

# What does our imbalance look like?
class_dist <- table(df$diabetes)
cat("Class distribution:\n")
print(class_dist)
cat("\nImbalance ratio:", round(class_dist["neg"] / class_dist["pos"], 2), ":1\n")

# Simulate what happens without handling imbalance
baseline_acc <- max(prop.table(class_dist))
cat("\nZero-Rule (majority class) accuracy:", round(baseline_acc, 4), "\n")
cat("A model can achieve 65% accuracy by ALWAYS predicting 'neg'!\n")

#' 
#' 
#' ---
#' 
## -----------------------------------------------------------------------------
#| label: imbalance-plot
#| fig-height: 3.0

ggplot(df, aes(x = diabetes, fill = diabetes)) +
  geom_bar(width = 0.5) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, color = "white") +
  scale_fill_manual(values = c("#4cc9f0","#e94560")) +
  labs(title = "Class Imbalance in Pima Diabetes Dataset") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

#' 
#' 
#' ## ROSE — Random Over-Sampling Examples
#' 
## -----------------------------------------------------------------------------
#| label: rose-oversample

cat("Before ROSE:\n")
table(train_df$diabetes)

# ROSE generates synthetic samples using a smoothed bootstrap
rose_train <- ROSE(diabetes ~ ., data = train_df, seed = 42)$data

cat("\nAfter ROSE:\n")
table(rose_train$diabetes)

#' 
## -----------------------------------------------------------------------------
#| label: rose-rf

# Train Random Forest on ROSE-balanced data
rf_rose <- randomForest(
  diabetes ~ .,
  data     = rose_train,
  ntree    = 500,
  mtry     = 3,
  importance = TRUE
)

rf_rose_pred <- predict(rf_rose, newdata = test_df)
cm_rose <- confusionMatrix(rf_rose_pred, test_df$diabetes, positive = "pos")

cat("ROSE-balanced RF:\n")
cat("  Accuracy:   ", round(cm_rose$overall["Accuracy"], 4), "\n")
cat("  Sensitivity:", round(cm_rose$byClass["Sensitivity"], 4), "\n")
cat("  Specificity:", round(cm_rose$byClass["Specificity"], 4), "\n")

#' 
#' ---
#' 
#' ## SMOTE — Synthetic Minority Oversampling
#' 
## -----------------------------------------------------------------------------
#| label: smote

# SMOTE creates synthetic minority samples by interpolating
# between existing minority class k-nearest neighbors
library(themis)
library(recipes)

smote_recipe <- recipe(diabetes ~ ., data = train_df) |>
  step_smote(diabetes, over_ratio = 0.8, neighbors = 5)

smote_prep   <- prep(smote_recipe, training = train_df)
smote_train  <- bake(smote_prep, new_data = NULL)

cat("After SMOTE:\n")
table(smote_train$diabetes)

#' 
## -----------------------------------------------------------------------------
#| label: smote-rf

rf_smote <- randomForest(diabetes ~ ., data = smote_train,
                          ntree = 500, mtry = 3, importance = TRUE)

rf_smote_pred <- predict(rf_smote, newdata = test_df)
cm_smote <- confusionMatrix(rf_smote_pred, test_df$diabetes, positive = "pos")

cat("SMOTE-balanced RF:\n")
cat("  Accuracy:   ", round(cm_smote$overall["Accuracy"],    4), "\n")
cat("  Sensitivity:", round(cm_smote$byClass["Sensitivity"],  4), "\n")
cat("  Specificity:", round(cm_smote$byClass["Specificity"],  4), "\n")

#' 
#' ---
#' 
#' ## Cost-Sensitive Random Forest
#' 
## -----------------------------------------------------------------------------
#| label: cost-sensitive

# Assign higher weight to the minority (positive) class
imbalance_ratio <- sum(train_df$diabetes == "neg") /
                   sum(train_df$diabetes == "pos")

cat("Class weight for 'pos':", round(imbalance_ratio, 2), "x 'neg'\n\n")

rf_weighted <- randomForest(
  diabetes ~ .,
  data      = train_df,
  ntree     = 500,
  mtry      = 3,
  classwt   = c("neg" = 1, "pos" = imbalance_ratio)  # cost-sensitive
)

rf_wt_pred <- predict(rf_weighted, newdata = test_df)
cm_wt <- confusionMatrix(rf_wt_pred, test_df$diabetes, positive = "pos")

cat("Weighted RF:\n")
cat("  Accuracy:   ", round(cm_wt$overall["Accuracy"],    4), "\n")
cat("  Sensitivity:", round(cm_wt$byClass["Sensitivity"],  4), "\n")
cat("  Specificity:", round(cm_wt$byClass["Specificity"],  4), "\n")

#' 
#' ---
#' 
#' ## Precision-Recall Curve — Better Metric for Imbalance
#' 
## -----------------------------------------------------------------------------
#| label: roc-curve
#| fig-height: 4.8
#| out-width: "90%"

rf_probs_rose  <- predict(rf_rose,     newdata=test_df, type="prob")[,"pos"]
rf_probs_smote <- predict(rf_smote,    newdata=test_df, type="prob")[,"pos"]
rf_probs_wt    <- predict(rf_weighted, newdata=test_df, type="prob")[,"pos"]

roc_rose  <- roc(test_df$diabetes, rf_probs_rose,  levels=c("neg","pos"), direction="<")
roc_smote <- roc(test_df$diabetes, rf_probs_smote, levels=c("neg","pos"), direction="<")
roc_wt    <- roc(test_df$diabetes, rf_probs_wt,    levels=c("neg","pos"), direction="<")

par(bg="#1e1e2e", col.axis="white", col.lab="white", col.main="white", fg="white")
plot(roc_rf,    col="#aaaaaa", lwd=1.5, lty=2, main="ROC — Imbalance Strategies vs Baseline")
plot(roc_rose,  col="#4cc9f0", lwd=2.5, add=TRUE)
plot(roc_smote, col="#e94560", lwd=2.5, add=TRUE)
plot(roc_wt,    col="#ffd60a", lwd=2.5, add=TRUE)
abline(0, 1, lty=2, col="#ffffff44")
legend("bottomright",
  legend=c(paste0("Baseline RF  AUC=",round(auc(roc_rf),3)),
           paste0("ROSE RF      AUC=",round(auc(roc_rose),3)),
           paste0("SMOTE RF     AUC=",round(auc(roc_smote),3)),
           paste0("Weighted RF  AUC=",round(auc(roc_wt),3))),
  col=c("#aaaaaa","#4cc9f0","#e94560","#ffd60a"), lwd=2.5, bty="n", text.col="white")

#' 
#' ---
#' 
## -----------------------------------------------------------------------------
#| label: pr-curve
#| fig-height: 4.8
#| out-width: "90%"
#| output-location: slide

# PR-AUC is more informative than ROC-AUC under class imbalance

# Install if needed
# install.packages("PRROC")
library(PRROC)

# Build PR curve objects — needs raw probability scores + binary labels
true_binary <- as.integer(test_df$diabetes == "pos")

pr_rf    <- pr.curve(scores.class0 = rf_probs_test,  weights.class0 = true_binary, curve = TRUE)
pr_rose  <- pr.curve(scores.class0 = rf_probs_rose,  weights.class0 = true_binary, curve = TRUE)
pr_smote <- pr.curve(scores.class0 = rf_probs_smote, weights.class0 = true_binary, curve = TRUE)
pr_wt    <- pr.curve(scores.class0 = rf_probs_wt,    weights.class0 = true_binary, curve = TRUE)

# Plot
par(bg = "#1e1e2e", col.axis = "white", col.lab = "white",
    col.main = "white", fg = "white", mar = c(4, 4, 3, 2))

plot(pr_rf,
     col       = "#aaaaaa",
     lwd       = 1.5,
     lty       = 2,
     main      = "Precision-Recall Curves — Imbalance Strategies vs Baseline",
     auc.main  = FALSE,
     xlab      = "Recall (Sensitivity)",
     ylab      = "Precision")

plot(pr_rose,  col = "#4cc9f0", lwd = 2.5, add = TRUE, auc.main = FALSE)
plot(pr_smote, col = "#e94560", lwd = 2.5, add = TRUE, auc.main = FALSE)
plot(pr_wt,    col = "#ffd60a", lwd = 2.5, add = TRUE, auc.main = FALSE)

# No-skill baseline = fraction of positives in test set
no_skill <- mean(true_binary)
abline(h = no_skill, lty = 2, col = "#ffffff55", lwd = 1.2)
text(0.05, no_skill + 0.02, "No-skill baseline",
     col = "#ffffff88", cex = 0.75)

legend("topright",
  legend = c(
    paste0("Baseline RF   PR-AUC = ", round(pr_rf$auc.integral,    3)),
    paste0("ROSE RF       PR-AUC = ", round(pr_rose$auc.integral,  3)),
    paste0("SMOTE RF      PR-AUC = ", round(pr_smote$auc.integral, 3)),
    paste0("Weighted RF   PR-AUC = ", round(pr_wt$auc.integral,    3))
  ),
  col     = c("#aaaaaa", "#4cc9f0", "#e94560", "#ffd60a"),
  lwd     = 2.5,
  bty     = "n",
  text.col = "white",
  cex     = 0.88
)


#' 
#' ---
#' 
#' ## Imbalance: Side-by-Side Comparison
#' 
## -----------------------------------------------------------------------------
#| label: imbalance-compare
#| fig-height: 4.5
#| output-location: slide

compare_imb <- tibble(
  Strategy    = c("Baseline RF","ROSE RF","SMOTE RF","Weighted RF"),
  Sensitivity = c(cm_rf$byClass["Sensitivity"],
                  cm_rose$byClass["Sensitivity"],
                  cm_smote$byClass["Sensitivity"],
                  cm_wt$byClass["Sensitivity"]),
  Specificity = c(cm_rf$byClass["Specificity"],
                  cm_rose$byClass["Specificity"],
                  cm_smote$byClass["Specificity"],
                  cm_wt$byClass["Specificity"])
) |>
  pivot_longer(-Strategy, names_to="Metric", values_to="Value")

ggplot(compare_imb, aes(Strategy, Value, fill=Metric)) +
  geom_col(position="dodge") +
  geom_text(aes(label=round(Value,3)), position=position_dodge(0.9),
            vjust=-0.3, size=3.5, color="white") +
  scale_fill_manual(values=c("#4cc9f0","#e94560")) +
  ylim(0, 1.08) +
  labs(title="Sensitivity vs Specificity under Class Imbalance Strategies",
       y="Value", x=NULL, fill=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=15, hjust=1))

#' 
#' ---
#' 
#' ## 🏆 Final Model Comparison
#' 
#' ---
#' 
#' ## Complete Model Leaderboard
#' 
## -----------------------------------------------------------------------------
#| label: leaderboard
#| output-location: slide

# Add imbalance-handled models
results <- bind_rows(results,
  tibble(Model="ROSE+RF",     Accuracy=cm_rose$overall["Accuracy"],
         Sensitivity=cm_rose$byClass["Sensitivity"],
         Specificity=cm_rose$byClass["Specificity"],
         Kappa=cm_rose$overall["Kappa"]),
  tibble(Model="SMOTE+RF",    Accuracy=cm_smote$overall["Accuracy"],
         Sensitivity=cm_smote$byClass["Sensitivity"],
         Specificity=cm_smote$byClass["Specificity"],
         Kappa=cm_smote$overall["Kappa"]),
  tibble(Model="Weighted+RF", Accuracy=cm_wt$overall["Accuracy"],
         Sensitivity=cm_wt$byClass["Sensitivity"],
         Specificity=cm_wt$byClass["Specificity"],
         Kappa=cm_wt$overall["Kappa"])
)

results |>
  arrange(desc(Accuracy)) |>
  mutate(across(where(is.numeric), ~round(.x, 4))) |>
  kbl(caption = "Model Leaderboard — Pima Diabetes Test Set") |>
  kable_styling(bootstrap_options = c("striped","hover","condensed"),
                full_width = FALSE, font_size = 14) |>
  row_spec(1, bold = TRUE, color = "white", background = "#e94560")

#' 
#' ---
#' 
#' ## Final ROC — All Models Together
#' 
## -----------------------------------------------------------------------------
#| label: final-roc
#| fig-height: 5.2
#| out-width: "95%"
#| output-location: slide

par(bg="#1e1e2e", col.axis="white", col.lab="white", col.main="white", fg="white",
    mar=c(4,4,3,2))

all_rocs <- list(
  `DT`         = roc_dt,
  `Naive Bayes`= roc_nb,
  `C5.0 Rules` = roc_c50,
  `Bagging`    = roc_bag,
  `Random Forest`= roc_rf,
  `GBM`        = roc_gbm,
  `SMOTE+RF`   = roc_smote
)

cols <- c("#808080","#4cc9f0","#f77f00","#e94560","#4361ee","#ffd60a","#06d6a0")

plot(all_rocs[[1]], col=cols[1], lwd=2, main="ROC Curves — All Models")
for (i in 2:length(all_rocs)) {
  plot(all_rocs[[i]], col=cols[i], lwd=2, add=TRUE)
}
abline(0,1, lty=2, col="#ffffff44")

legend("bottomright",
  legend = paste0(names(all_rocs), "  AUC=",
                  round(sapply(all_rocs, auc), 3)),
  col=cols, lwd=2, bty="n", text.col="white", cex=0.82)

