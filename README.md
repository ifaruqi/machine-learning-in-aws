# Iris Species Prediction with AWS SageMaker Deployment 🌸

## Overview
This repository provides a comprehensive guide to building, validating, and deploying a machine learning model using the Iris dataset. We leverage Scikit-learn for model development and AWS SageMaker for productionalization, making the model accessible for real-time predictions in a cloud environment.

## Dataset
The Iris dataset is one of the most iconic datasets in pattern recognition literature. Consisting of 150 samples from three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor), it includes four features: sepal length, sepal width, petal length, and petal width.

## Prerequisites
- Python 3.x
- AWS Account
- Boto3, the Python SDK for AWS
- Scikit-learn library
- Jupyter Notebook (for local experimentation)

## Step-by-Step Guide
### 1. Model Development
Using Scikit-learn, we develop a Random Forest Classifier. After splitting the dataset into training and testing subsets, the model is trained on the former and validated on the latter.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

### 2. AWS SageMaker Deployment
Once our model is trained and ready, we deploy it using AWS SageMaker:

1. Model Packaging: Convert the trained model into a .tar.gz file.
2. Upload to S3: Utilize Boto3 to upload the model file to an S3 bucket.
3. SageMaker Model Creation: Set up a SageMaker model using the S3 URI.
4. Endpoint Creation: Establish an endpoint for the model, allowing for real-time predictions.

### 3. Predictions
With the SageMaker endpoint in place, we can make real-time predictions:

```python
data = [[5.1, 3.5, 1.4, 0.2]]
response = client.invoke_endpoint(EndpointName=endpoint_name,
                                  ContentType='application/json',
                                  Body=json.dumps(data))
```

## Cleanup
To avoid incurring unnecessary costs, always remember to delete the SageMaker endpoint, S3 bucket, and the SageMaker model after use.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

