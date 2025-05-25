
# Load required libraries
library(readr)
library(caret)
library(e1071)
library(nnet)       # for multinom logistic regression
library(rpart)      # for decision tree
library(ggplot2)
library(dplyr)

# Load the dataset
iris_data <- read_csv("C:/Users/Administrator/Downloads/Iris.csv")

# Drop the Id column
iris_data <- iris_data %>% select(-Id)

# Convert Species to factor
iris_data$Species <- as.factor(iris_data$Species)

# Split into training and testing sets (80% train, 20% test)
set.seed(123)
train_index <- createDataPartition(iris_data$Species, p = 0.8, list = FALSE)
train_data <- iris_data[train_index, ]
test_data <- iris_data[-train_index, ]

# -----------------------------------------------
# 1. Logistic Regression (Multinomial)
# -----------------------------------------------
log_model <- train(Species ~ ., data = train_data, method = "multinom", trace = FALSE)
log_preds <- predict(log_model, newdata = test_data)
log_cm <- confusionMatrix(log_preds, test_data$Species)
print("Logistic Regression Accuracy:")
print(log_cm)

# -----------------------------------------------
# 2. Decision Tree
# -----------------------------------------------
tree_model <- train(Species ~ ., data = train_data, method = "rpart")
tree_preds <- predict(tree_model, newdata = test_data)
tree_cm <- confusionMatrix(tree_preds, test_data$Species)
print("Decision Tree Accuracy:")
print(tree_cm)

# -----------------------------------------------
# 3. Support Vector Machine (SVM)
# -----------------------------------------------
svm_model <- train(Species ~ ., data = train_data, method = "svmLinear")
svm_preds <- predict(svm_model, newdata = test_data)
svm_cm <- confusionMatrix(svm_preds, test_data$Species)
print("SVM Accuracy:")
print(svm_cm)

# -----------------------------------------------
# 4. Visualization with ggplot2
# -----------------------------------------------

# PCA for 2D visualization
pca_data <- prcomp(iris_data[,1:4], center = TRUE, scale. = TRUE)
pca_df <- data.frame(pca_data$x[,1:2], Species = iris_data$Species)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Species)) +
  geom_point(size = 3, alpha = 0.8) +
  theme_minimal() +
  labs(title = "PCA of Iris Dataset", x = "Principal Component 1", y = "Principal Component 2")
