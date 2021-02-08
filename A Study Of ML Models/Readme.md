The project aims to compare and study the performance of various machine learning models viz., Decision Trees, Neural Networks, KNN, Boosted Decision Trees and SVM.
For this purpose, two classification problems were identified and the models were trained and tested on both the problems.

# Adult Data Set

Problem1 is a binary classification problem that aims to classify if the income of a person is greater than 50K based on several census parameters, such as age,
education, marital status, hours of work per week, etc. The dataset used for solving the same is the Adult dataset from the UCI Machine learning repository.
This dataset consists of 48842 instances and 13 variables, of which 32,561 instances are presented as training set and the remaining constitute the test set.
The Adult dataset is a slightly imbalanced one with the majority class representing 75% of the dataset. Oversampling was done to account for the
imbalance and relevant metrics chosen to analyse the performance of various algorithms.

# Avila Data Set
Problem 2 is a multi label classification problem involving the Avila dataset from the UCI Machine learning repository. The Avila data set has been extracted from
800 images of the ‘Avila Bible’, an XII century giant Latin copy of the Bible. The prediction task is to associate each pattern of the script to a copyist. The features include numerical data such as intercolumnar distance, margins, interlinear spacing, etc. The dataset consists of a training set of 10430 samples, and a test set of 10437 samples. There are 12 classes to which the data belong to and the distribution is imbalanced with the majority class representing 4286 instances
and the minority class representing 5 instances. Oversampling was done to account for the imbalance and the model’s performance on the test set was evaluated with the original imbalanced testset.
