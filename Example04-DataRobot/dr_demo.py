## Overview
#This notebook shows a very basic workflow using DataRobot’s Python SDK. For simplicity, this notebook shows only a fraction of available functionality. However full documentation for the Python library is available at http://pythonhosted.org/datarobot/entities/quickstart.html (http://pythonhosted.org/datarobot/entities/quickstart.html) The datarobot library can be installed with pip install datarobot


## Starting a Modeling Project
# We will create a project using the source data file diabetic_data.csv. In case DataRobot is being used in a workflow involving Pandas, the user may also provide a Pandas DataFrame as the argument to sourcedata in the command below.

import datarobot as dr
import pandas as pd
import os

#We'll train with the first 10k data points, then use the rest to test our model. 
diabetes_df=pd.read_csv('DataRobot/dataset_diabetes/diabetic_data.csv')
diabetes_df['readmitted'][diabetes_df['readmitted'] != 'NO'] = 'YES'
train_df=diabetes_df[:10000]
test_df=diabetes_df[10000:]

#Connecting to DataRobot + Starting a project
your_token=os.environ["DR_API_TOKEN"]
dr.Client(token=your_token, endpoint='https://app.datarobot.com/api/v2')
project = dr.Project.start(train_df,project_name='My first project',target='readmitted', worker_count=4)

#While the DataRobot library provides all functionality a user will need for automated workflows, many users like to explore the data and model visualizations available in our web-based GUI.
import webbrowser
project_url=project.get_leaderboard_ui_permalink()
webbrowser.open(project_url)


## Working with Models 
project.wait_for_autopilot() #block until all models finish
models = project.get_models(with_metric='LogLoss')
best_model = models[0] # choose best model

try:
  feature_impacts = best_model.get_feature_impact() # if they've already been computed
except dr.errors.ClientError as e:
  assert e.status_code == 404 # if the feature impact score haven't been computed already
  impact_job = best_model.request_feature_impact()
  feature_impacts = impact_job.get_result_when_complete()
feature_impacts.sort(key=lambda x: x['impactNormalized'])
feature_impacts_df = pd.DataFrame(feature_impacts).set_index('featureName')
feature_impacts_df.impactNormalized.plot.bar()

## Make Predictions
#use the testing data to make predictions
prediction_data_in_dr = project.upload_dataset(test_df) # In most use cases, use new data for this
predict_job = best_model.request_predictions(prediction_data_in_dr.id)
predictions = predict_job.get_result_when_complete()
print(predictions.head(25))
