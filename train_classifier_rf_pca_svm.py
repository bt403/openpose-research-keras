from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_validate
import argparse
import pandas as pd
from util_classifier_training import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='./data/estimates/train_data.pkl')
parser.add_argument('--test_data', type=str, default='./data/estimates/test_data.pkl')
args = parser.parse_args()

train = pd.read_pickle(args.train_data)
test = pd.read_pickle(args.test_data)

targets = ["target", ["cheeks", "nose", "mouth", "eyes", "ears", "forehead"]]

augmented = True
results = {"model_001":[]}

for i in targets:
  print("------")
  print("RESULTS FOR: " + str(i))
  if not isinstance(i, str):
    for k in results.keys():
      results[k] = {}
      for t in i:
        results[k][t] = []
  inverse = i != "target"
  
  #############
  weights = "balanced"
  if not isinstance(i, str):
    train = train[train["target"] == "on-head"]
    test = test[test["target"] == "on-head"]
    binary = False
  runCrossValidationSVM(results, train, test, i, "model_bright", weights, binary)
  #############

  for k in results.keys():
    df_results = pd.DataFrame(results[k])
    print("------")
    print("Results for " + k + "")
    display(df_results)
