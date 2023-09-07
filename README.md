# Iris Species Prediction with AWS SageMaker Deployment 🌸

## Overview
This repository provides a comprehensive guide to building, validating, and deploying a machine learning model using the Iris dataset. We leverage Scikit-learn for model development and AWS SageMaker for productionalization, making the model accessible for real-time predictions in a cloud environment.

## Dataset
The Iris dataset is one of the most iconic datasets in pattern recognition literature. Consisting of 150 samples from three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor), it includes four features: sepal length, sepal width, petal length, and petal width.

## Prerequisites
Python 3.x
AWS Account
Boto3, the Python SDK for AWS
Scikit-learn library
Jupyter Notebook (for local experimentation)
Step-by-Step Guide
1. Model Development
Using Scikit-learn, we develop a Random Forest Classifier. After splitting the dataset into training and testing subsets, the model is trained on the former and validated on the latter.
