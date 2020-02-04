import pickle
import numpy as np

model = pickle.load(open("models/sklearn_rf.pkl","rb"))

def predict(args):
  account=np.array(args["feature"].split(",")).reshape(1,-1)
  return {"result" : model.predict(account)[0]}
  
