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








#--------------------------------- Question 3 ----------------------------------
# 3. Does the given player tend to perform better under pressure (or clutch up) 
#    compared to their overall average?

