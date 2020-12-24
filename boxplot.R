data <- read.csv('results.csv')
png(file='boxplot2.png')
boxplot(accuracy ~ model, data=data, xlab='Model', ylab='Accuracy (%)', main='Model Accuracies')
