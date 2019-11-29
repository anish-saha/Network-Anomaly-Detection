import os
import pandas as pd
import numpy as np

from avd.graph_learning_controller import GraphLearningController
from avd.learners.sklearner import SkLearner
from avd.configs import config
from avd.datasets.twitter import load_data


output_folder = os.getcwd() + "/data/output"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

labels = {"neg": "Real", "pos": "Fake"}

twitter_graph, twitter_config = load_data(dataset_file_name="twitter_filtered.csv", labels_file_name="twitter_labels_filtered.csv",
                                          labels_map=labels, limit=6000000) # Loads filtered dataset.
print(len(twitter_graph.vertices))

if twitter_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

twitter_config._name = "twitter_" + "RandomForest"
learner = SkLearner(labels=labels)
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph,
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "LogisticRegression"
learner = SkLearner(labels=labels).set_logistic_regression_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "Adaboost"
learner = SkLearner(labels=labels).set_adaboost_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "Bagging"
learner = SkLearner(labels=labels).set_bagging_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "RFBagging"
learner = SkLearner(labels=labels).set_rf_bagging_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "GradientBoosting"
learner = SkLearner(labels=labels).set_gradient_boosting_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "IsolationForest"
learner = SkLearner(labels=labels).set_isolation_forest_classifier()
glc = GraphLearningController(learner, twitter_config)
result_path = os.path.join(output_folder, twitter_config.name  + "res.csv")
glc.classify_by_links(twitter_graph, 
                      result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


def aggreagate_res(data_folder, res_path):
    results_frame = pd.DataFrame()
    for f in os.listdir(data_folder):
        temp_df = pd.read_csv(data_folder + "/" + f,index_col=0, encoding='utf-8', engine='python')
        results_frame = results_frame.append(temp_df)
    results_frame = results_frame.groupby("src_id").mean()

    return results_frame.reset_index()

df = aggreagate_res(output_folder, "res.csv").sort_values("mean_link_label", ascending=False)

df["actual_sum"] = df["actual"].cumsum()
df["k"] = 1
df["k"] = df["k"].cumsum()

df.head(10)

df["p@k"] = df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)

df[["k", "p@k"]].head(10)

import matplotlib.pyplot as plt

plt.figure()
df[["k", "p@k"]][:500].plot(x="k", y= "p@k")
plt.plot(df[["k"]].values, np.full((len(df[["k"]]),1), 0.06))


