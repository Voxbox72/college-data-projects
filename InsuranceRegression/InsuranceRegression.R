# Importing packages
library(tidyverse)
library(gridExtra)

# Reading in files
insurance <- read.csv("insurance.csv")

# EDA (Exploratory Data Analysis)
# Check the structure and summary of the data
str(insurance)
summary(insurance)

# Data Visualization
# Scatter plots for 'age' vs 'charges' and 'bmi' vs 'charges'
plot.age <- ggplot(insurance, aes(x = age, y = charges)) + geom_point()
plot.bmi <- ggplot(insurance, aes(x = bmi, y = charges)) + geom_point()

# Boxplots for categorical variables
plot.sex <- ggplot(insurance, aes(x = sex, y = charges)) + geom_boxplot()
plot.smoker <- ggplot(insurance, aes(x = smoker, y = charges)) + geom_boxplot()
plot.child <- ggplot(insurance, aes(x = as.factor(children), y = charges)) + geom_boxplot()
plot.region <- ggplot(insurance, aes(x = region, y = charges)) + geom_boxplot()

# Combine plots
grid.arrange(plot.age, plot.bmi, plot.sex, plot.smoker, plot.child, plot.region, ncol = 2, nrow = 3)

# Simple linear regression using age as the predictor variable
mod1 <- lm(charges ~ age, data = insurance)
summary(mod1)

# Multiple linear regression using age and bmi as predictor variables
mod2 <- lm(charges ~ age + bmi, data = insurance)
summary(mod2)

# Create the third model with more factors
mod3 <- lm(charges ~ age + bmi + smoker + sex + children + region, data = insurance)
summary(mod3)

# Checking for residuals and outliers
insurance$residuals1 <- rstandard(mod1)
insurance$outlier1 <- abs(insurance$residuals1) > 2.5

insurance$residuals2 <- rstandard(mod2)
insurance$outlier2 <- abs(insurance$residuals2) > 2.5

insurance$residuals3 <- rstandard(mod3)
insurance$outlier3 <- abs(insurance$residuals3) > 2.5

# Percentage of cases with standardized residual > 2.5
percentage_outliers1 <- sum(insurance$outlier1) / nrow(insurance) * 100
percentage_outliers2 <- sum(insurance$outlier2) / nrow(insurance) * 100
percentage_outliers3 <- sum(insurance$outlier3) / nrow(insurance) * 100
# Output percentage of outliers
paste("Percentage of outliers for model 1:", round(percentage_outliers1, 2), "%")
paste("Percentage of outliers for model 2:", round(percentage_outliers2, 2), "%")
paste("Percentage of outliers for model 3:", round(percentage_outliers3, 2), "%")

# Predictions for all models
insurance$pred_mod1 <- predict(mod1)
insurance$pred_mod2 <- predict(mod2)
insurance$pred_mod3 <- predict(mod3)

# Visualization of models
# Scatter plot for model 1
plot_mod1 <- ggplot(insurance, aes(x = charges, y = pred_mod1)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") + # Add identity line
  labs(title = "Model 1: Age",
       x = "Observed Charges",
       y = "Predicted Charges")

# Scatter plot for model 2
plot_mod2 <- ggplot(insurance, aes(x = charges, y = pred_mod2)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") + # Add identity line
  labs(title = "Model 2: Age and BMI",
       x = "Observed Charges",
       y = "Predicted Charges")

# Scatter plot for model 3
plot_mod3 <- ggplot(insurance, aes(x = charges, y = pred_mod3)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") + # Add identity line
  labs(title = "Model 3: Age, BMI, Smoker, Sex, Children, Region",
       x = "Observed Charges",
       y = "Predicted Charges")

# Combine all plots
grid.arrange(plot_mod1, plot_mod2, plot_mod3, ncol = 3)

# Define hypothetical scenarios
scenarios <- data.frame(
  age = c(40, 45, 50),  # Age in years
  bmi = c(25, 30, 35),   # BMI (Body Mass Index)
  smoker = c("no", "yes", "no"),  # Smoking status
  sex = c("male", "female", "male"),  # Gender
  children = c(1, 2, 0),  # Number of children
  region = c("northeast", "southeast", "northwest")  # Region
)

# Predict charges for each observation using model 3
predicted_charges <- predict(mod3, newdata = scenarios, interval = "confidence")

# Print the confidence intervals for fitted values
print(predicted_charges)

# Print the Analysis of Variance Table
print(anova(mod3))

# Print the confidence interval of Bo and B1
print(confint(mod3))
  
