#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[2]:


import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# # Model Training

# In[6]:


# Split predictor and target variables
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# print(X.head())
print(X.shape)

# print(y.head())
print(y.shape)


# In[7]:


# Split Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[9]:


# Train the Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[10]:


# Save the model

import pickle

with open("random_forest_iris_model.pkl", "wb") as model_file:
    pickle.dump(rf, model_file)


# In[11]:


with open("random_forest_iris_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))


# # Deployment to AWS

# In[25]:


import boto3
import time

# AWS Configuration
region_name = "eu-north-1"
bucket_name = "iris-predictor-bucket"
model_name = "iris-predictor-model"

# Create S3 bucket
s3 = boto3.client('s3', region_name=region_name)
s3.create_bucket(Bucket=bucket_name,
                 CreateBucketConfiguration={'LocationConstraint': region_name})

# Create IAM role for SageMaker
iam = boto3.client('iam')
trust_relationship = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
role_name = "SageMaker-ExecutionRole-{}".format(int(time.time()))
create_role_response = iam.create_role(
    RoleName=role_name,
    AssumeRolePolicyDocument=json.dumps(trust_relationship)
)
role_arn = create_role_response["Role"]["Arn"]
iam.attach_role_policy(
    RoleName=role_name,
    PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
)
iam.attach_role_policy(
    RoleName=role_name,
    PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
)


# In[16]:


# 1. Create a tar.gz file
import tarfile
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("random_forest_iris_model.pkl")
    tar.add("inference.py")


# In[26]:


# 2. Upload the model to S3
s3 = boto3.client('s3', region_name=region_name)
with open("model.tar.gz", "rb") as f:
    s3.upload_fileobj(f, bucket_name, "model/model.tar.gz")


# In[28]:


# 3. Create a SageMaker model
sagemaker = boto3.client('sagemaker', region_name=region_name)
model_url = f"s3://{bucket_name}/model/model.tar.gz"

create_model_response = sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': '763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-sagemaker',  # The Docker image for the model, e.g., a pre-built SageMaker image
        'ModelDataUrl': model_url
    },
    ExecutionRoleArn=role_arn
)


# In[29]:


# 4. Create an endpoint configuration
endpoint_config_name = "iris-predictor-config"
endpoint_config_response = sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'DefaultVariant',
            'ModelName': model_name,
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1
        }
    ]
)


# In[30]:


# 5. Create an endpoint
endpoint_name = "iris-predictor-endpoint"
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)


# # Testing 

# In[32]:


import boto3
import json

# Initialize the SageMaker runtime client
client = boto3.client('runtime.sagemaker')

# Define the endpoint name
endpoint_name = 'iris-predictor-endpoint' 

# Sample input data for prediction
# This should be appropriately formatted. In this example, we assume the input is a JSON serialized 2D array.
data = [[2.4, 3.3, 4.4, 0.5]]

response = client.invoke_endpoint(EndpointName=endpoint_name,
                                  ContentType='application/json',
                                  Body=json.dumps(data))

# Extract and print the prediction result
result = json.loads(response['Body'].read().decode())
print(result)


# # Deleting All Resources

# In[23]:


import boto3

# Initialize the SageMaker client
sagemaker = boto3.client('sagemaker', region_name=region_name)

# Delete the endpoint
sagemaker.delete_endpoint(EndpointName=endpoint_name)
print(f"Deleted endpoint: {endpoint_name}")

# Delete the endpoint configuration:
sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
print(f"Deleted endpoint configuration: {endpoint_config_name}")


# Delete the bucket
s3 = boto3.resource('s3', region_name=region_name)
bucket = s3.Bucket(bucket_name)

for obj in bucket.objects.all():
    obj.delete()
    
bucket.delete()

print(f"Deleted bucket: {bucket_name}")

# Delete the model
sagemaker = boto3.client('sagemaker', region_name=region_name)

# Delete the model
sagemaker.delete_model(ModelName=model_name)

print(f"Deleted model: {model_name}")

