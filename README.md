# Description

Getting and analysing existing data is the very first task of a data scientist. The next step is to find tendencies and to generalize.

For example, let's say we want to know what a cat is. We can learn by heart some pictures of cats and then classify as cat animals that are similar to the pictures. We then need a way to "measure" similarity. This is called instance-based learning.

Another way of generalizing is by creating a model from the existing examples and make prediction based on that model.

We clearly see a trend here, eventhough the data is quite noisy, it looks like the feature 2 goes up linearly as the feature 1 increases. So in a model selection step, we can decide to go for a linear model.

feature_2 = θ0 + θ1 . feature_1

This model has two parameters, θ0 and θ1. After choosing the right values for them, we can make our model represent a linear function matching the data:

# Task

A linear model makes predictions by computing a weighted sum of the features (plus a constant term called the bias term):

y = hθ(x) = θT·x = θnxn + ... + θ2x2 + θ1x1 + θ0

• y is the predicted value. • n is the number of features. • xi is the ith feature value (with x0 always equals to 1). • θj is the jth model feature weight (including the bias term θ0). • · is the dot product. • hθ is called the hypothesis function indexed by θ.

→ Write the linear hypothesis function.

def h(x, theta):
    ...
Now that we have our linear regression model, we need to define a cost function to train it, i.e measure how well the model performs and fits the data. One of the most commonly used function is the Root Mean Squared Error (RMSE). As it is a cost function, we will need to optimize it and find the value of theta which minimizes it.

Since the sqrt function is monotonous and increasing, we can minimize the square of RMSE, the Mean Square Error (MSE) and it will lead to the same result.

 m MSE(X, hθ) =   1⁄m ∑ (θT·x(i) - y(i))2          k=1

• X is a matrix which contains all the feature values. There is one row per instance. • m is the number of instances. • xi is the feature values vector of the ith instance • yi is the label (desired value) of the ith instance.

# Install & Usage
pip install numpy

python3 my_linear_regression.py