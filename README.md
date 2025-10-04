# ML Classification

Have you ever wondered how spam emails are detected, why some loan requests get rejected, or how Google Images can accurately recognize objects?
All of these are real-world applications of classification in Machine Learning.

In this section, we’ll explore what classification means, how to build a simple classification model, and the key metrics used to evaluate its performance.

## What is Classification?

In simple terms, classification is a type of supervised machine learning.

Supervised learning means the model is trained with labeled data, i.e., input features are provided along with their corresponding output labels.

- If the output variable is continuous, the task is called regression.

- If the output variable is categorical, the task is called classification.

## Types of Classification

1 .Binary Classification

  - The output has only two possible categories.

Example: Approving or rejecting a loan, or detecting whether an email is spam or not.

2. Multi-class Classification

 - The output has more than two discrete categories.

Example: Classifying different types of skin diseases.

## Classification Intuition

Let’s consider binary classification. Suppose we plot all the data points on a graph. If it is possible to draw a line that separates the two classes, then this line can be used to distinguish between the two possible outcomes.

Since this line separates the dataset into two distinct groups, we can think of it as a decision boundary or a simple model that differentiates the data points.

In linear regression, we fit a regression line by minimizing the Mean Squared Error (MSE). However, in classification, this approach faces challenges because:

A straight line may separate classes but does not give us the probability of a point belonging to a particular class.

Classification problems require not just a boundary, but also a way to estimate probabilities and handle overlapping points.

This is where models like Logistic Regression come into play, as they help estimate probabilities while drawing effective decision boundaries.


## Logistic Regression

Unlike linear regression, which predicts continuous values, logistic regression applies a sigmoid function to map predictions into a range between 0 and 1.

This output can then be interpreted as the probability of a data point belonging to a certain class.

By setting a threshold (commonly 0.5), we can classify whether a point belongs to Class 0 or Class 1.


## Data Imbalance

One of the most important issues in classification problems is data imbalance. For example, imagine we have a dataset where 95% of the samples belong to one class and only 5% belong to the other class. If we train a model on this dataset and see an accuracy of 95%, does that mean the model is good?

Not necessarily. Accuracy can be misleading in imbalanced datasets. In this scenario, the model could simply predict the majority class for every input and still achieve high accuracy, while completely ignoring the minority class. Behind the scenes, the model might have learned nothing about the smaller class, which can be critical depending on the application (e.g., fraud detection, disease diagnosis).

To properly evaluate models on imbalanced data, we should use metrics like:

- Precision, Recall, F1-score – to measure performance on each class separately
- Confusion Matrix – to visualize how many samples of each class are correctly or incorrectly classified
- ROC-AUC – to evaluate the model’s discriminative ability

We’ll discuss these metrics later, but for now, let’s look at the ways to address data imbalance.

## Data Imbalace Techniques

There are different techniques to tackle data imbalance problems. Some of them are:

### Random Undersampling

In this technique, instead of increasing the minority class, we reduce the number of samples in the majority class so that the dataset becomes more balanced.

- Disadvantages:
  - Important information from the majority class may be lost, which can affect model performance.
 
### Random Oversampling

In this technique, we increase the number of samples in the minority class, often by duplicating existing data points, to balance the dataset.

- Disadvantage:
  - It can reduce the originality of the data, as no new unique information is added, which may lead to overfitting
 
### Synthetic Minority Oversampling Technique (SMOTE)

In this technique, new synthetic samples for the minority class are generated based on the similarity between existing minority class samples.

- Instead of simply duplicating data, SMOTE creates new, slightly different data points to help balance the dataset.
- Advantage: Helps the model learn better patterns from the minority class without overfitting to duplicates.
- Disadvantage: May generate samples that overlap with the majority class, potentially introducing some noise.

### Adaptive Synthetic Sampling Method (ADASYN)

It is similar to SMOTE , but is more focused on the minority dataset which are hard to detect
- Advantage : Helps the model pay more attention to difficult cases, improving learning for the minority class.
- Disadvantage: Can introduce noise if too many synthetic samples are generated in sparse regions, potentially affecting model accuracy.

  SMOTETomek (Oversampling followed by Undersampling)

### SMOTETomek is a hybrid technique that combines SMOTE oversampling with Tomek link undersampling:

SMOTE: First, synthetic minority class samples are generated to balance the dataset.

Tomek Links: Then, Tomek links (pairs of nearest neighbor samples from opposite classes that are very close) are removed, usually from the majority class. This cleans overlapping samples and reduces class ambiguity.

- Advantage: Balances the dataset while removing noisy or overlapping majority samples, improving model performance.

- Disadvantage: Slightly more complex and may remove some useful majority class samples if not applied carefully.
