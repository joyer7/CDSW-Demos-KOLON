import requests
import json
import time
import os

HOST = "http://" + os.environ['CDSW_DOMAIN']
API_KEY = os.environ['API_KEY']

def listProjects(context):
  url = "/".join([HOST, "api/v1/users", context, "projects"])
  res = requests.get(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, "")
  )
  return res.json()

def executeRun(projectId, script, arguments):
  url = "/".join([HOST, "api/altus-ds-1/ds/run"])
  job_params = {"project":projectId, "script":script, "arguments":arguments}
  res = requests.post(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, ""),
      data = json.dumps(job_params)
  )
  return res.json()

def describeRun(runId):
  url = "/".join([HOST, "api/altus-ds-1/ds/describeRun"])
  job_params = {"id":runId}
  res = requests.post(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, ""),
      data = json.dumps(job_params)
  )
  return res.json()

def listRuns(projectId, metric):
  url = "/".join([HOST, "api/altus-ds-1/ds/listruns"])
  job_params = {"project":projectId, "pageSize":30, "metricsOrder":metric, "orderSort":"desc"}
  res = requests.post(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, ""),
      data = json.dumps(job_params)
  )
  return res.json()

def promoteOutput(expirementId, file):
  url = "/".join([HOST, "api/altus-ds-1/ds/promoteRunOutput"])
  job_params = {"id":expirementId, "files":[file]}
  res = requests.post(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, ""),
      data = json.dumps(job_params)
  )
  return res.status_code

def downloadFile(project, file):
  url = "/".join([HOST, "api/v1/projects", project, "files", file])
  r = requests.get(url, stream=True, auth = (API_KEY, ""))
  with open(file + "1", 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024): 
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)
  return file

def createModels(name, description, projectId):
  url = "/".join([HOST, "api/altus-ds-1/models/create-model"])
  job_params = {"name":name, "description":description, "projectId":projectId}
  res = requests.post(
      url,
      headers = {"Content-Type": "application/json"},
      auth = (API_KEY, ""),
      data = json.dumps(job_params)
  )
  return res.json()

projects = listProjects("jhubbard")
projectId = projects[0]["id"]

run = executeRun(str(projectId), "ds-for-telco/dsfortelcosklearn_exp.py", "20 20 gini")
while not run["ended"]:
  time.sleep(5)
  print("checking run")
  run = describeRun(str(run["id"]))

expirements = listRuns(str(projectId), "auroc")["runs"]
if (run["status"] == "succeeded" and 
    run["id"] == expirements[0]["id"]):
  print("promoting and downloading run")
  promoteOutput(str(run["id"]), "sklearn_rf.pkl")
  downloadFile("jhubbard/ds-for-telco", "sklearn_rf.pkl")

best = next(exp for exp in expirements if exp["status"] == "succeeded")

models = listModels(projectId)

