**Final Project \- Proposal**  
Andrew King & Levi Hood (Team 7\)  
INFO 523: Data Mining and Discovery  
March 31, 2026

1. **Dataset Selected:**

*NBA Player Shot Dataset (2023) \- A comprehensive analysis of LeBron James, James Harden, and Stephen Curry’s performance during the regular 2023 NBA season (Kaggle)*

[https://www.kaggle.com/datasets/dhavalrupapara/nba-2023-player-shot-dataset?select=2\_james\_harden\_shot\_chart\_2023.csv](https://www.kaggle.com/datasets/dhavalrupapara/nba-2023-player-shot-dataset?select=2_james_harden_shot_chart_2023.csv)

- Variables: 15   
  - Continuous: top, left, time\_remaining, distance\_ft, player\_team\_score, opponent\_team\_score  
  - Date: date  
  - Ordinal: qtr, season, shot\_type  
  - Binary: **result**, lead  
  - Nominal: opponent, team, color  
- Observations: 1,435

2. **Project Goals and Questions:**

*Do players have specific hot-zones where they have a significant advantage? Where are these hot-zones?*

The first goal is to identify court zones (identified by coordinates) where each player's field goal percentage is significantly above their overall average, ultimately identifying regions where they are, in effect, most likely to score. Using zone-level shooting percentages aggregated across the 2023 season, we will determine whether each player has a reliable spatial signature and compare how those signatures differ across the three players (Lebron, Harden, and Curry). Statistical testing will be applied to confirm whether observed hot zones are genuine patterns rather than random variation. Ideally, we would be able to create a heat map for each player to better visualize these trends.

*Who is the most consistent shooter?*

The second goal is to measure shooting consistency at the game level. Rather than evaluating raw efficiency alone, this analysis will quantify variance in field goal percentage across games using metrics such as standard deviation and coefficient of variation to determine which player delivers the most stable, predictable shooting output over the course of a season.

*Does the given player tend to perform better under pressure (or clutch up) compared to their overall average?*

Using the provided dataset, we apply predictive modeling on historical shot data to evaluate performance in high-stress scenarios. Clutch situations are defined by a combination of factors: whether the team is trailing or leading, the size of that margin, shot difficulty based on distance, and time remaining in the game or quarter. Together, these thresholds will allow us to isolate high-pressure moments and predict how likely each of the three selected NBA players is to convert a shot when it matters most, relative to their baseline shooting performance across the full game.

Together, these questions demonstrate how exploratory data analysis and descriptive statistics can extract meaningful basketball insights. This project applies sports analytics alongside foundational data science techniques, making it a practical application of coursework concepts on a real-world dataset.

3. **Tentative Analysis Plan:**

The tentative analysis plan includes the following steps: processing the data by handling missing or duplicated values and resolving inconsistencies, exploring the shape of the data and the various metrics available for each player to better understand what we will be comparing, and generating a summary statistics table. For the hot-zone analysis, we will begin by defining what constitutes a hot zone, as well as establishing the thresholds that determine what qualifies as a clutch situation. Finally, we plan to employ predictive modeling using logistic regression, with the shot result as the target variable, evaluating performance through a train/test split and exploring k-fold cross-validation.

4. **Expected Outcomes:**

We expect this analysis to result in a reliable predictive model capable of determining whether a shot is likely to be made given a set of defined variables. To strengthen model performance, we plan to explore k-fold cross-validation across our training and testing splits. Additionally, we aim to visualize our findings through a variety of plots, including geospatial visualizations that map shot locations and outcomes across the court, alongside other graphics to enhance the overall report.