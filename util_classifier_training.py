import statistics
from scipy import stats
from skmultilearn.ext import Keras
from sklearn.decomposition import PCA
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.problem_transform import ClassifierChain
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import LabelPowerset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from statistics import mean, median
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def get_columns_hand():
  columns_hand = []
  id_c = 0
  results_hand_locations_list = ["hand_found_R", "hand_found_L", "origin_x_hand_L", "origin_y_hand_L", "origin_x_hand_R", "origin_y_hand_R",
          "thumb_finger_x_hand_L", "thumb_finger_y_hand_L", "thumb_finger_x_hand_R", "thumb_finger_y_hand_R",
  "index_finger_x_hand_L", "index_finger_y_hand_L", "index_finger_x_hand_R", "index_finger_y_hand_R",
 "middle_finger_x_hand_L", "middle_finger_y_hand_L", "middle_finger_x_hand_R", "middle_finger_y_hand_R",
 "ring_finger_x_hand_L", "ring_finger_y_hand_L", "ring_finger_x_hand_R", "ring_finger_y_hand_R", 
 "pinky_finger_x_hand_L", "pinky_finger_y_hand_L", "pinky_finger_x_hand_R", "pinky_finger_y_hand_R"]
  for i in results_hand_locations_list:
    if id_c < 2:
      continue
    else:
      if (id_c%2 == 0): #x coordinate
        i2 = results_hand_locations_list[id_c+1]
        #To Right Eye
        L1_dist_x = "Lcoord_REyeX_" + i
        L1_dist_y = "Lcoord_REyeY_" + i2
        L1_dist = "Lcoord_REye_" + i.replace("_x", "").replace("_y", "")
        #To Left Eye
        L2_dist_x = "Lcoord_LEyeX_" + i
        L2_dist_y = "Lcoord_LEyeY_" + i2
        L2_dist = "Lcoord_LEye_" + i.replace("_x", "").replace("_y", "")
        #To Nose
        L3_dist_x = "Lcoord_NoseX_" + i
        L3_dist_y = "Lcoord_NoseY_" + i2
        L3_dist = "Lcoord_Nose_" + i.replace("_x", "").replace("_y", "")
        columns_hand.append(L1_dist_x)
        columns_hand.append(L1_dist_y)
        columns_hand.append(L1_dist)
        columns_hand.append(L2_dist_x)
        columns_hand.append(L2_dist_y)
        columns_hand.append(L2_dist)
        columns_hand.append(L3_dist_x)
        columns_hand.append(L3_dist_y)
        columns_hand.append(L3_dist)
  id_c += 1
  return columns_hand

def create_model_multiclass(input_dim, output_dim):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def confusion_matrix_scorer(clf, X, y):
  y_pred = clf.predict(X)
  cm = confusion_matrix(y, y_pred)
  res = {}
  ci = 1
  return {'tn': cm[0, 0], 'fp': cm[0, 1],
          'fn': cm[1, 0], 'tp': cm[1, 1]}
def confusion_matrix_scorer_multi(clf, X, y):
  y_pred = clf.predict(X)
  y_pred = y_pred.todense()
  matrices = {}
  for i in range(len(y_pred[0].tolist()[0])):
    i_pred = np.array(y_pred[:,i].tolist()).flatten()
    i_test = y.iloc[:, i] 
    cm = confusion_matrix(i_test, i_pred)
    if len(cm[0]) == 1:
      matrices['tn_' + str(i)] = cm[0, 0]
      matrices['fp_' + str(i)] = 0
      matrices['fn_' + str(i)] = 0
      matrices['tp_' + str(i)] = 0
    else:
      matrices['tn_' + str(i)] = cm[0, 0]
      matrices['fp_' + str(i)] = cm[0, 1]
      matrices['fn_' + str(i)] = cm[1, 0]
      matrices['tp_' + str(i)] = cm[1, 1]
  print(matrices)
  return matrices

def calculate_significance_test(table):
  result = mcnemar(table, exact=True, correction=True)
  print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
  alpha = 0.05
  if result.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
  else:
    print('Different proportions of errors (reject H0)')

def train_svm(results, x_train, y_train, x_test, y_test, result_name, params, groups, weights="balanced", inverse=False, binary=True, with_pca=False):
  # Validation Set - 5 Fold CV
  splitter = GroupKFold(n_splits=5)
  if binary:
    svclassifier = SVC(kernel='rbf', class_weight=weights, random_state=1, gamma=params['model__gamma'], C=params['model__C'])
    if with_pca:
        pca = PCA(n_components=params["pca__n_components"])
        pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("pca", pca), ("model", svclassifier)])
    else:
        pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("model", svclassifier)])
    validation = cross_validate(pipe, x_train, y_train, scoring=confusion_matrix_scorer, cv=splitter, groups = groups)
  else:
    svc = SVC(kernel='rbf', class_weight=weights, random_state=1, gamma=params['model__classifier__gamma'], C=params['model__classifier__C'])
    svclassifier =  LabelPowerset(svc)
    if with_pca:
        pca = PCA(n_components=params["pca__n_components"])
        pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("pca", pca), ("model", svclassifier)])
    else:
        pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("model", svclassifier)])
    validation = cross_validate(pipe, x_train, y_train, scoring=confusion_matrix_scorer_multi, cv=splitter, groups = groups)

  print("VALIDATION")
  print(validation)

  if binary:
    precision_outside_head_c = []
    precision_on_head_c = []
    recall_outside_head_c = []
    recall_on_head_c = []
    majority_vote_c = []
    accuracy_c = []

    for i in range(5):
      fn = validation["test_fn"][i]
      fp = validation["test_fp"][i]
      tn = validation["test_tn"][i]
      tp = validation["test_tp"][i]

      accuracy = (tp + tn)/(fn + tp + tn + fp)
      precision_outside_head = tn/(tn + fn)
      precision_on_head = tp/(tp + fp)
      recall_outside_head = tn/(tn + fp)
      recall_on_head = tp/(tp + fn)
      majority_vote = (fn + tp)/(fn + tp + tn + fp)

      accuracy_c.append(accuracy)
      precision_outside_head_c.append(precision_outside_head)
      precision_on_head_c.append(precision_on_head)
      recall_outside_head_c.append(recall_outside_head)
      recall_on_head_c.append(recall_on_head)
      majority_vote_c.append(majority_vote)

    accuracy = statistics.mean(accuracy_c)
    precision_on_head = statistics.mean(precision_on_head_c)
    precision_outside_head = statistics.mean(precision_outside_head_c)
    recall_outside_head = statistics.mean(recall_outside_head_c)
    recall_on_head = statistics.mean(recall_on_head_c)
    majority_vote = statistics.mean(majority_vote_c)

    if (majority_vote < 0.5):
      results[result_name].append(["CV Majority Vote", 1 - majority_vote ])
    else:
      results[result_name].append(["CV Majority Vote", majority_vote ])
    results[result_name].append(["CV Accuracy", accuracy])
    print("CV - " + str(accuracy))
    results[result_name].append(["CV Precision On Head", precision_on_head])
    results[result_name].append(["CV Precision Outside Head", precision_outside_head])
    results[result_name].append(["CV Recall On Head", recall_on_head])
    results[result_name].append(["CV Recall Outside Head", recall_outside_head])
    results[result_name].append(["CV F1 Score On Head", 2*precision_on_head*recall_on_head/(precision_on_head + recall_on_head)])
    results[result_name].append(["CV F1 Score Outside Head", 2*precision_outside_head*recall_outside_head/(precision_outside_head + recall_outside_head)])
  else:
    for t in range(len(y_train.iloc[0])):
      keys_list = list(results[result_name])
      key = keys_list[t]

      precision_not_target_c = []
      precision_on_target_c = []
      recall_not_target_c = []
      recall_on_target_c = []
      majority_vote_c = []
      accuracy_c = []
      print("target number " + str(t)) 
      print("-----------")
      for i in range(5):
        fn = validation["test_fn_" + str(t)][i]
        fp = validation["test_fp_" + str(t)][i]
        tn = validation["test_tn_" + str(t)][i]
        tp = validation["test_tp_" + str(t)][i]

        accuracy = (tp + tn)/(fn + tp + tn + fp)
        precision_not_target = tn/(tn + fn)
        precision_on_target = tp/(tp + fp)
        recall_not_target = tn/(tn + fp)
        recall_on_target = tp/(tp + fn)
        majority_vote = (fn + tp)/(fn + tp + tn + fp)

        accuracy_c.append(accuracy)
        precision_not_target_c.append(precision_not_target)
        precision_on_target_c.append(precision_on_target)
        recall_not_target_c.append(recall_not_target)
        recall_on_target_c.append(recall_on_target)
        majority_vote_c.append(majority_vote)

      accuracy = statistics.mean(accuracy_c)
      precision_on_target = statistics.mean(precision_on_target_c)
      precision_not_target = statistics.mean(precision_not_target_c)
      recall_not_target = statistics.mean(recall_not_target_c)
      recall_on_target = statistics.mean(recall_on_target_c)
      majority_vote = statistics.mean(majority_vote_c)

      if (majority_vote < 0.5):
        results[result_name][key].append(["CV Majority Vote", 1 - majority_vote ])
      else:
        results[result_name][key].append(["CV Majority Vote", majority_vote ])
      results[result_name][key].append(["CV Accuracy", accuracy])
      results[result_name][key].append(["CV Precision On Head", precision_on_target])
      results[result_name][key].append(["CV Precision Outside Head", precision_not_target])
      results[result_name][key].append(["CV Recall On Head", recall_on_target])
      results[result_name][key].append(["CV Recall Outside Head", recall_not_target])
      results[result_name][key].append(["CV F1 Score On Head", 2*precision_on_target*recall_on_target/(precision_on_target + recall_on_target)])
      results[result_name][key].append(["CV F1 Score Outside Head", 2*precision_not_target*recall_not_target/(precision_not_target + recall_not_target)])
  
  # Test on Test Set
  pipe.fit(x_train, y_train)
  y_pred = pipe.predict(x_test)
  if binary:
    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test, y_pred)

    positives_model = cm[1, 1] + cm[0, 1]
    negatives_model = cm[1, 0] + cm[0, 0]
    positives_real = cm[1, 1] + cm[1, 0]
    negatives_real = cm[0, 1] + cm[0, 0]

    confusion_table_model_vs_majority_vote = [[0, positives_model],[0, negatives_model]]
    confusion_table_model_vs_random_vote = [[int(positives_real/2), int(positives_real/2)],[int(negatives_real/2), int(negatives_real/2)]]
    print(confusion_table_model_vs_majority_vote)
    print(confusion_table_model_vs_random_vote)
    calculate_significance_test(confusion_table_model_vs_majority_vote)
    calculate_significance_test(confusion_table_model_vs_random_vote)

    print(classification_report(y_test,y_pred))
    precision_on_head = (cm[0][0])/(cm[0][0] + cm[1][0])
    precision_outside_head = (cm[1][1])/(cm[1][1] + cm[0][1])
    recall_on_head = (cm[0][0])/(cm[0][0] + cm[0][1])
    recall_outside_head = (cm[1][1])/(cm[1][1] + cm[1][0])

    maj_vote = (cm[1][0] + cm[1][1])/(cm[1][0] + cm[1][1] + cm[0][0] + cm[0][1])
    if (maj_vote < 0.5):
      results[result_name].append(["Majority Vote", 1 - (maj_vote) ])
    else:
      results[result_name].append(["Majority Vote", (maj_vote) ])
    results[result_name].append(["Accuracy", accuracy])
    results[result_name].append(["Precision On Head", precision_on_head])
    results[result_name].append(["Precision Outside Head", precision_outside_head])
    results[result_name].append(["Recall On Head", recall_on_head])
    results[result_name].append(["Recall Outside Head", recall_outside_head])
    results[result_name].append(["F1 Score On Head", 2*precision_on_head*recall_on_head/(precision_on_head + recall_on_head)])
    results[result_name].append(["F1 Score Outside Head", 2*precision_outside_head*recall_outside_head/(precision_outside_head + recall_outside_head)])
  else:
    y_pred = y_pred.todense()
    for i in range(len(y_pred[0].tolist()[0])):
      keys_list = list(results[result_name])
      key = keys_list[i]

      print("Target : " + key)
      i_pred = np.array(y_pred[:,i].tolist()).flatten()
      i_test = y_test.iloc[:, i] 
      accuracy = accuracy_score(i_test,i_pred)
      cm = confusion_matrix(i_test, i_pred)
      print(classification_report(i_test,i_pred))
      positives_model = cm[1, 1] + cm[0, 1]
      negatives_model = cm[1, 0] + cm[0, 0]
      positives_real = cm[1, 1] + cm[1, 0]
      negatives_real = cm[0, 1] + cm[0, 0]

      confusion_table_model_vs_majority_vote = [[0, positives_model],[0, negatives_model]]
      confusion_table_model_vs_random_vote = [[int(positives_real/2), int(positives_real/2)],[int(negatives_real/2), int(negatives_real/2)]]
      print(confusion_table_model_vs_majority_vote)
      print(confusion_table_model_vs_random_vote)
      calculate_significance_test(confusion_table_model_vs_majority_vote)
      calculate_significance_test(confusion_table_model_vs_random_vote)

      precision_outside_target = (cm[0][0])/(cm[0][0] + cm[1][0])
      precision_on_target= (cm[1][1])/(cm[1][1] + cm[0][1])
      recall_outside_target = (cm[0][0])/(cm[0][0] + cm[0][1])
      recall_on_target = (cm[1][1])/(cm[1][1] + cm[1][0])

      maj_vote = (cm[1][0] + cm[1][1])/(cm[1][0] + cm[1][1] + cm[0][0] + cm[0][1])
      if (maj_vote < 0.5):
        results[result_name][key].append("Majority Vote" + " - " + str(1 - maj_vote))
        print("Majority Vote: " +  str(1 - maj_vote))
      else:
        print("Majority Vote: " +  str( maj_vote))
        results[result_name][key].append("Majority Vote" + " - " + str(maj_vote))
      results[result_name][key].append("Accuracy" + " - " + str(accuracy))
      results[result_name][key].append("Precision On Target" + " - " + str(precision_on_target))
      results[result_name][key].append("Precision Outside Target" + " - " + str(precision_outside_target))
      results[result_name][key].append("Recall On Target" + " - " + str(recall_on_target))
      results[result_name][key].append("Recall Outside Target" + " - " + str(recall_outside_target))
      results[result_name][key].append("F1 Score On Target" + " - " + str(2*precision_on_target*recall_on_target/(precision_on_target + recall_on_target)))
      results[result_name][key].append("F1 Score Outside Target" + " - " + str(2*precision_outside_target*recall_outside_target/(precision_outside_target + recall_outside_target)))


columns_filter_distances = [ "L1_dist", "L1_dist_x", "L1_dist_y", "L2_dist", "L2_dist_x", "L2_dist_y", "L3_dist", "L3_dist_x", "L3_dist_y", "L4_dist", "L4_dist_x", "L4_dist_y", "L5_dist", "L5_dist_x", "L5_dist_y", "L6_dist", "L6_dist_x", "L6_dist_y", "L7_dist", "L7_dist_x", "L7_dist_y", "L8_dist", "L8_dist_x", "L8_dist_y", "L9_dist", "L9_dist_x", "L9_dist_y", "L10_dist", "L10_dist_x", "L10_dist_y", "L11_dist", "L11_dist_x", "L11_dist_y", "L12_dist", "L12_dist_x", "L12_dist_y", "coord_REye_Exists", "coord_LEye_Exists", "coord_REar_Exists", "coord_LEar_Exists", "coord_Nose_Exists"]
columns_filter_angular = ["angle_LElbow", "angle_RElbow", "sumAnglesR", "sumAnglesL", "angle_RShoulder", "angle_LShoulder"]
columns_filter_temporal = ["speed_LWrist", "speed_RWrist", "speed_RElbow", "speed_LElbow", "displacement_RWrist", "displacement_LWrist", "displacement_RElbow", "displacement_LElbow", "acceleration_RWrist_x", "acceleration_RWrist_y", "acceleration_LWrist_x", "acceleration_LWrist_y"]
columns_hand = get_columns_hand()
columns_filter_hands =  ["hand_found_L", "hand_found_R"] + columns_hand
imp = SimpleImputer(missing_values=np.nan, strategy='median')

def featureSelection(data, columns_v, target_val, ratio=1.0):
  print(columns_v)
  RFC = RandomForestClassifier(n_estimators = 200, random_state=1)
  pipeline = Pipeline(steps=[('i', imp), ("scale", StandardScaler()), ('m', RFC)])

  cv_rfc = cross_validate(pipeline, data[columns_v], data[target_val], cv = 5, scoring = 'accuracy', groups=data["video_name"], return_estimator =True)
  feature_importances = cv_rfc['estimator'][0].steps[2][1].feature_importances_
  print(feature_importances)
  count = 1
  for idx,sel in enumerate(cv_rfc['estimator']):
    if (idx == 0):
      continue
    else:
      feature_importances = feature_importances + sel.steps[2][1].feature_importances_
      count += 1
  feature_importances = feature_importances/count
  threshold = median(feature_importances)*ratio
  filtered_importances = np.array(feature_importances)[feature_importances >= threshold]
  filtered_features = np.array(columns_v)[feature_importances >= threshold]
  return list(filtered_features)

def getFeatures(data, target_val):
  features_distances = featureSelection(data, columns_filter_distances, target_val, 1.2)
  if ("L1_dist" in features_distances): features_distances.append("L2_dist")
  if ("L1_dist_y" in features_distances): features_distances.append("L2_dist_y")
  if ("L1_dist_x" in features_distances): features_distances.append("L2_dist_x")
  if ("L2_dist" in features_distances): features_distances.append("L1_dist")
  if ("L2_dist_y" in features_distances): features_distances.append("L1_dist_y")
  if ("L2_dist_x" in features_distances): features_distances.append("L1_dist_x")
  if ("L3_dist" in features_distances): features_distances.append("L4_dist")
  if ("L3_dist_y" in features_distances): features_distances.append("L4_dist_y")
  if ("L3_dist_x" in features_distances): features_distances.append("L4_dist_x")
  if ("L4_dist" in features_distances): features_distances.append("L3_dist")
  if ("L4_dist_y" in features_distances): features_distances.append("L3_dist_y")
  if ("L4_dist_x" in features_distances): features_distances.append("L3_dist_x")
  if ("L5_dist" in features_distances): features_distances.append("L6_dist")
  if ("L5_dist_y" in features_distances): features_distances.append("L6_dist_y")
  if ("L5_dist_x" in features_distances): features_distances.append("L6_dist_x")
  if ("L6_dist" in features_distances): features_distances.append("L5_dist")
  if ("L6_dist_y" in features_distances): features_distances.append("L5_dist_y")
  if ("L6_dist_x" in features_distances): features_distances.append("L5_dist_x")
  if ("L7_dist" in features_distances): features_distances.append("L8_dist")
  if ("L7_dist_y" in features_distances): features_distances.append("L8_dist_y")
  if ("L7_dist_x" in features_distances): features_distances.append("L8_dist_x")
  if ("L8_dist" in features_distances): features_distances.append("L7_dist")
  if ("L8_dist_y" in features_distances): features_distances.append("L7_dist_y")
  if ("L8_dist_x" in features_distances): features_distances.append("L7_dist_x")
  if ("L9_dist" in features_distances): features_distances.append("L12_dist")
  if ("L9_dist_y" in features_distances): features_distances.append("L12_dist_y")
  if ("L9_dist_x" in features_distances): features_distances.append("L12_dist_x")
  if ("L12_dist" in features_distances): features_distances.append("L9_dist")
  if ("L12_dist_y" in features_distances): features_distances.append("L9_dist_y")
  if ("L12_dist_x" in features_distances): features_distances.append("L9_dist_x")
  if ("L10_dist" in features_distances): features_distances.append("L11_dist")
  if ("L10_dist_y" in features_distances): features_distances.append("L11_dist_y")
  if ("L10_dist_x" in features_distances): features_distances.append("L11_dist_x")
  if ("L11_dist" in features_distances): features_distances.append("L10_dist")
  if ("L11_dist_y" in features_distances): features_distances.append("L10_dist_y")
  if ("L11_dist_x" in features_distances): features_distances.append("L10_dist_x")
  if ("coord_REye_Exists" in features_distances): features_distances.append("coord_LEye_Exists")
  if ("coord_LEye_Exists" in features_distances): features_distances.append("coord_REye_Exists")
  if ("coord_REar_Exists" in features_distances): features_distances.append("coord_LEar_Exists")
  if ("coord_LEar_Exists" in features_distances): features_distances.append("coord_REar_Exists")
  features_distances = list(set(features_distances))

  features_angular = featureSelection(data, columns_filter_angular, target_val, 1.2)
  if ("sumAnglesL" in features_angular): features_angular.append("sumAnglesR")
  if ("sumAnglesR" in features_angular): features_angular.append("sumAnglesL")
  if ("angle_RElbow" in features_angular): features_angular.append("angle_LElbow")
  if ("angle_LElbow" in features_angular): features_angular.append("angle_RElbow")
  if ("angle_RShoulder" in features_angular): features_angular.append("angle_LShoulder")
  if ("angle_LShoulder" in features_angular): features_angular.append("angle_RShoulder")
  features_angular = list(set(features_angular))

  features_temporal = featureSelection(data, columns_filter_temporal, target_val, 1.2)
  if ("displacement_RWrist" in features_temporal): features_temporal.append("displacement_LWrist")
  if ("displacement_LWrist" in features_temporal): features_temporal.append("displacement_RWrist")
  if ("displacement_RElbow" in features_temporal): features_temporal.append("displacement_LElbow")
  if ("displacement_LElbow" in features_temporal): features_temporal.append("displacement_RElbow")
  if ("speed_RWrist" in features_temporal): features_temporal.append("speed_LWrist")
  if ("speed_LWrist" in features_temporal): features_temporal.append("speed_RWrist")
  if ("speed_RElbow" in features_temporal): features_temporal.append("speed_LElbow")
  if ("speed_LElbow" in features_temporal): features_temporal.append("speed_RElbow")
  if ("acceleration_RWrist_y" in features_temporal): features_temporal.append("acceleration_LWrist_y")
  if ("acceleration_LWrist_y" in features_temporal): features_temporal.append("acceleration_RWrist_y")
  if ("acceleration_RWrist_x" in features_temporal): features_temporal.append("acceleration_LWrist_x")
  if ("acceleration_LWrist_x" in features_temporal): features_temporal.append("acceleration_RWrist_x")
  features_temporal = list(set(features_temporal))

  features_hands = featureSelection(data, columns_filter_hands, target_val, 1.2)
  if ("hand_found_L" in features_hands): features_hands.append("hand_found_R")
  if ("hand_found_R" in features_hands): features_hands.append("hand_found_L")

  if('Lcoord_REyeX_origin_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_origin_x_hand_R')
  if('Lcoord_LEyeX_origin_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_origin_x_hand_L')

  if('Lcoord_REyeY_origin_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_origin_y_hand_R')
  if('Lcoord_LEyeY_origin_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_origin_y_hand_L')

  if('Lcoord_REye_origin_hand_L' in features_hands): features_hands.append('Lcoord_LEye_origin_hand_R')
  if('Lcoord_LEye_origin_hand_R' in features_hands): features_hands.append('Lcoord_REye_origin_hand_L')

  if('Lcoord_LEyeX_origin_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_origin_x_hand_R')
  if('Lcoord_REyeX_origin_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_origin_x_hand_L')

  if('Lcoord_LEyeY_origin_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_origin_y_hand_R')
  if('Lcoord_REyeY_origin_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_origin_y_hand_L')

  if('Lcoord_LEye_origin_hand_L' in features_hands): features_hands.append('Lcoord_REye_origin_hand_R')
  if('Lcoord_REye_origin_hand_R' in features_hands): features_hands.append('Lcoord_LEye_origin_hand_L')

  if('Lcoord_NoseX_origin_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_origin_x_hand_R')
  if('Lcoord_NoseX_origin_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_origin_x_hand_L')

  if('Lcoord_NoseY_origin_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_origin_y_hand_R')
  if('Lcoord_NoseY_origin_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_origin_y_hand_L')

  if('Lcoord_Nose_origin_hand_L' in features_hands): features_hands.append('Lcoord_Nose_origin_hand_R')
  if('Lcoord_Nose_origin_hand_R' in features_hands): features_hands.append('Lcoord_Nose_origin_hand_L')

  if('Lcoord_REyeX_thumb_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_thumb_finger_x_hand_R')
  if('Lcoord_LEyeX_thumb_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_thumb_finger_x_hand_L')

  if('Lcoord_REyeY_thumb_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_thumb_finger_y_hand_R')
  if('Lcoord_LEyeY_thumb_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_thumb_finger_y_hand_L')

  if('Lcoord_REye_thumb_finger_hand_L' in features_hands): features_hands.append('Lcoord_LEye_thumb_finger_hand_R')
  if('Lcoord_LEye_thumb_finger_hand_R' in features_hands): features_hands.append('Lcoord_REye_thumb_finger_hand_L')

  if('Lcoord_LEyeX_thumb_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_thumb_finger_x_hand_R')
  if('Lcoord_REyeX_thumb_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_thumb_finger_x_hand_L')

  if('Lcoord_LEyeY_thumb_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_thumb_finger_y_hand_R')
  if('Lcoord_REyeY_thumb_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_thumb_finger_y_hand_L')

  if('Lcoord_LEye_thumb_finger_hand_L' in features_hands): features_hands.append('Lcoord_REye_thumb_finger_hand_R')
  if('Lcoord_REye_thumb_finger_hand_R' in features_hands): features_hands.append('Lcoord_LEye_thumb_finger_hand_L')

  if('Lcoord_NoseX_thumb_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_thumb_finger_x_hand_R')
  if('Lcoord_NoseX_thumb_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_thumb_finger_x_hand_L')

  if('Lcoord_NoseY_thumb_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_thumb_finger_y_hand_R')
  if('Lcoord_NoseY_thumb_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_thumb_finger_y_hand_L')

  if('Lcoord_Nose_thumb_finger_hand_L' in features_hands): features_hands.append('Lcoord_Nose_thumb_finger_hand_R')
  if('Lcoord_Nose_thumb_finger_hand_R' in features_hands): features_hands.append('Lcoord_Nose_thumb_finger_hand_L')

  if('Lcoord_REyeX_index_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_index_finger_x_hand_R')
  if('Lcoord_LEyeX_index_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_index_finger_x_hand_L')

  if('Lcoord_REyeY_index_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_index_finger_y_hand_R')
  if('Lcoord_LEyeY_index_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_index_finger_y_hand_L')

  if('Lcoord_REye_index_finger_hand_L' in features_hands): features_hands.append('Lcoord_LEye_index_finger_hand_R')
  if('Lcoord_LEye_index_finger_hand_R' in features_hands): features_hands.append('Lcoord_REye_index_finger_hand_L')

  if('Lcoord_LEyeX_index_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_index_finger_x_hand_R')
  if('Lcoord_REyeX_index_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_index_finger_x_hand_L')

  if('Lcoord_LEyeY_index_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_index_finger_y_hand_R')
  if('Lcoord_REyeY_index_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_index_finger_y_hand_L')

  if('Lcoord_LEye_index_finger_hand_L' in features_hands): features_hands.append('Lcoord_REye_index_finger_hand_R')
  if('Lcoord_REye_index_finger_hand_R' in features_hands): features_hands.append('Lcoord_LEye_index_finger_hand_L')

  if('Lcoord_NoseX_index_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_index_finger_x_hand_R')
  if('Lcoord_NoseX_index_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_index_finger_x_hand_L')

  if('Lcoord_NoseY_index_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_index_finger_y_hand_R')
  if('Lcoord_NoseY_index_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_index_finger_y_hand_L')

  if('Lcoord_Nose_index_finger_hand_L' in features_hands): features_hands.append('Lcoord_Nose_index_finger_hand_R')
  if('Lcoord_Nose_index_finger_hand_R' in features_hands): features_hands.append('Lcoord_Nose_index_finger_hand_L')

  if('Lcoord_REyeX_middle_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_middle_finger_x_hand_R')
  if('Lcoord_LEyeX_middle_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_middle_finger_x_hand_L')

  if('Lcoord_REyeY_middle_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_middle_finger_y_hand_R')
  if('Lcoord_LEyeY_middle_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_middle_finger_y_hand_L')

  if('Lcoord_REye_middle_finger_hand_L' in features_hands): features_hands.append('Lcoord_LEye_middle_finger_hand_R')
  if('Lcoord_LEye_middle_finger_hand_R' in features_hands): features_hands.append('Lcoord_REye_middle_finger_hand_L')

  if('Lcoord_LEyeX_middle_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_middle_finger_x_hand_R')
  if('Lcoord_REyeX_middle_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_middle_finger_x_hand_L')

  if('Lcoord_LEyeY_middle_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_middle_finger_y_hand_R')
  if('Lcoord_REyeY_middle_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_middle_finger_y_hand_L')

  if('Lcoord_LEye_middle_finger_hand_L' in features_hands): features_hands.append('Lcoord_REye_middle_finger_hand_R')
  if('Lcoord_REye_middle_finger_hand_R' in features_hands): features_hands.append('Lcoord_LEye_middle_finger_hand_L')

  if('Lcoord_NoseX_middle_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_middle_finger_x_hand_R')
  if('Lcoord_NoseX_middle_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_middle_finger_x_hand_L')

  if('Lcoord_NoseY_middle_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_middle_finger_y_hand_R')
  if('Lcoord_NoseY_middle_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_middle_finger_y_hand_L')

  if('Lcoord_Nose_middle_finger_hand_L' in features_hands): features_hands.append('Lcoord_Nose_middle_finger_hand_R')
  if('Lcoord_Nose_middle_finger_hand_R' in features_hands): features_hands.append('Lcoord_Nose_middle_finger_hand_L')

  if('Lcoord_REyeX_ring_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_ring_finger_x_hand_R')
  if('Lcoord_LEyeX_ring_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_ring_finger_x_hand_L')

  if('Lcoord_REyeY_ring_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_ring_finger_y_hand_R')
  if('Lcoord_LEyeY_ring_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_ring_finger_y_hand_L')

  if('Lcoord_REye_ring_finger_hand_L' in features_hands): features_hands.append('Lcoord_LEye_ring_finger_hand_R')
  if('Lcoord_LEye_ring_finger_hand_R' in features_hands): features_hands.append('Lcoord_REye_ring_finger_hand_L')

  if('Lcoord_LEyeX_ring_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_ring_finger_x_hand_R')
  if('Lcoord_REyeX_ring_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_ring_finger_x_hand_L')

  if('Lcoord_LEyeY_ring_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_ring_finger_y_hand_R')
  if('Lcoord_REyeY_ring_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_ring_finger_y_hand_L')

  if('Lcoord_LEye_ring_finger_hand_L' in features_hands): features_hands.append('Lcoord_REye_ring_finger_hand_R')
  if('Lcoord_REye_ring_finger_hand_R' in features_hands): features_hands.append('Lcoord_LEye_ring_finger_hand_L')

  if('Lcoord_NoseX_ring_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_ring_finger_x_hand_R')
  if('Lcoord_NoseX_ring_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_ring_finger_x_hand_L')

  if('Lcoord_NoseY_ring_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_ring_finger_y_hand_R')
  if('Lcoord_NoseY_ring_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_ring_finger_y_hand_L')

  if('Lcoord_Nose_ring_finger_hand_L' in features_hands): features_hands.append('Lcoord_Nose_ring_finger_hand_R')
  if('Lcoord_Nose_ring_finger_hand_R' in features_hands): features_hands.append('Lcoord_Nose_ring_finger_hand_L')

  if('Lcoord_REyeX_pinky_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_LEyeX_pinky_finger_x_hand_R')
  if('Lcoord_LEyeX_pinky_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_REyeX_pinky_finger_x_hand_L')

  if('Lcoord_REyeY_pinky_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_LEyeY_pinky_finger_y_hand_R')
  if('Lcoord_LEyeY_pinky_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_REyeY_pinky_finger_y_hand_L')

  if('Lcoord_REye_pinky_finger_hand_L' in features_hands): features_hands.append('Lcoord_LEye_pinky_finger_hand_R')
  if('Lcoord_LEye_pinky_finger_hand_R' in features_hands): features_hands.append('Lcoord_REye_pinky_finger_hand_L')

  if('Lcoord_LEyeX_pinky_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_REyeX_pinky_finger_x_hand_R')
  if('Lcoord_REyeX_pinky_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_LEyeX_pinky_finger_x_hand_L')

  if('Lcoord_LEyeY_pinky_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_REyeY_pinky_finger_y_hand_R')
  if('Lcoord_REyeY_pinky_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_LEyeY_pinky_finger_y_hand_L')

  if('Lcoord_LEye_pinky_finger_hand_L' in features_hands): features_hands.append('Lcoord_REye_pinky_finger_hand_R')
  if('Lcoord_REye_pinky_finger_hand_R' in features_hands): features_hands.append('Lcoord_LEye_pinky_finger_hand_L')

  if('Lcoord_NoseX_pinky_finger_x_hand_L' in features_hands): features_hands.append('Lcoord_NoseX_pinky_finger_x_hand_R')
  if('Lcoord_NoseX_pinky_finger_x_hand_R' in features_hands): features_hands.append('Lcoord_NoseX_pinky_finger_x_hand_L')

  if('Lcoord_NoseY_pinky_finger_y_hand_L' in features_hands): features_hands.append('Lcoord_NoseY_pinky_finger_y_hand_R')
  if('Lcoord_NoseY_pinky_finger_y_hand_R' in features_hands): features_hands.append('Lcoord_NoseY_pinky_finger_y_hand_L')

  if('Lcoord_Nose_pinky_finger_hand_L' in features_hands): features_hands.append('Lcoord_Nose_pinky_finger_hand_R')
  if('Lcoord_Nose_pinky_finger_hand_R' in features_hands): features_hands.append('Lcoord_Nose_pinky_finger_hand_L')
  features_hands = list(set(features_hands))
  print(features_distances)
  filtered_columns = np.concatenate([features_distances, features_angular, features_temporal, features_hands])
  print(len(filtered_columns))
  return filtered_columns

def getBestParams(x_data, target_data, weights="balanced", with_pca = True, binary=True, groups=None):
  target_data = target_data.values
  print(target_data)
  if (with_pca):
    if (binary):
      pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()),("pca", PCA()), ("model", SVC())])
      param_grid = {
          "pca__n_components": [0.75, 0.8, 0.85, 0.9],
          "model__C": [100.0, 10.0, 1.0, 0.1, 0.01],
          "model__gamma": ["scale", 0.1],
          "model__class_weight": [weights]
      }
    else:
      pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()),("pca", PCA()), ("model", LabelPowerset())])
     
      param_grid = [ {
          "pca__n_components": [0.75, 0.8, 0.85, 0.9],
          "model__classifier": [SVC()],
          "model__classifier__C": [100.0, 10.0, 1.0, 0.1, 0.01],
          "model__classifier__gamma": ["scale", 0.1],
          "model__classifier__class_weight": [weights]
          }
      ]

  else:
    if (binary):
      pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("model", SVC())])
      param_grid = {
          "model__C": [100.0, 10.0, 1.0, 0.1, 0.01],
          "model__gamma": ["scale", 0.1],
          "model__class_weight": [weights]
      }   
    else:
      #pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("model", OneVsRestClassifier(SVC()) )])
      pipe = Pipeline(steps=[('imputation',imp), ("scale", StandardScaler()), ("model", LabelPowerset() )])
      param_grid = [ {
          "model__classifier": [SVC()],
          "model__classifier__C": [100.0, 10.0, 1.0, 0.1, 0.01],
          "model__classifier__gamma": ["scale", 0.1],
          "model__classifier__class_weight": [weights]
          }
      ]

  splitter = GroupKFold(n_splits=5)
  if groups is not None:
    search = GridSearchCV(pipe, param_grid, verbose=2, cv=splitter)#, n_jobs=4)
    search.fit(x_data, target_data, groups=groups)
    print("group search")
  else:
    search = GridSearchCV(pipe, param_grid, verbose=2)#, n_jobs=4)
    search.fit(x_data, target_data)
  print("Best parameter (CV score=%0.3f):" % search.best_score_)
  print(search.best_params_)
  return search.best_params_


def runCrossValidationSVM(x_data, x_data_test, target_val, model_name, weights="balanced", binary= True):
  print("--- Running Cross Validation for " + str(model_name) + " and target type: " + str(target_val) + " ---")
  filtered_columns = getFeatures(x_data, target_val)
  params = getBestParams(x_data[filtered_columns], x_data[target_val], weights, binary=binary, groups=x_data["video_name"])
  train_svm(x_data[filtered_columns], x_data[target_val], x_data_test[filtered_columns], x_data_test[target_val], model_name, params, x_data["video_name"], weights, inverse, binary, with_pca=True)

