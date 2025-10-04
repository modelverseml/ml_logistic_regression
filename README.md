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
