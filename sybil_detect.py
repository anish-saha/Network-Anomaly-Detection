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
rf_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph,
                      rf_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)

twitter_config._name = "twitter_" + "LogisticRegression"
learner = SkLearner(labels=labels).set_logistic_regression_classifier()
glc = GraphLearningController(learner, twitter_config)
lr_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      lr_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "Adaboost"
learner = SkLearner(labels=labels).set_adaboost_classifier()
glc = GraphLearningController(learner, twitter_config)
adaboost_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      adaboost_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "Bagging"
learner = SkLearner(labels=labels).set_bagging_classifier()
glc = GraphLearningController(learner, twitter_config)
bagging_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      bagging_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "RFBagging"
learner = SkLearner(labels=labels).set_rf_bagging_classifier()
glc = GraphLearningController(learner, twitter_config)
baggingRF_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      baggingRF_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "GradientBoosting"
learner = SkLearner(labels=labels).set_gradient_boosting_classifier()
glc = GraphLearningController(learner, twitter_config)
gradboost_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      gradboost_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)


twitter_config._name = "twitter_" + "IsolationForest"
learner = SkLearner(labels=labels).set_isolation_forest_classifier()
glc = GraphLearningController(learner, twitter_config)
iso_result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
glc.classify_by_links(twitter_graph, 
                      iso_result_path,
                      test_size={"neg": 2000, "pos": 200},
                      train_size={"neg": 5000, "pos": 5000},
                      meta_data_cols=meta_data_cols)

def aggregate_res(res_path):
    results_frame = pd.DataFrame()
    temp_df = pd.read_csv(res_path, index_col=0, encoding='utf-8', engine='python')
    results_frame = results_frame.append(temp_df)
    results_frame = results_frame.groupby("src_id").mean()

    return results_frame.reset_index()

rf_df = aggregate_res(rf_result_path).sort_values("mean_link_label", ascending=False)
lr_df = aggregate_res(lr_result_path).sort_values("mean_link_label", ascending=False)
adaboost_df = aggregate_res(adaboost_result_path).sort_values("mean_link_label", ascending=False)
bagging_df = aggregate_res(bagging_result_path).sort_values("mean_link_label", ascending=False)
baggingRF_df = aggregate_res(baggingRF_result_path).sort_values("mean_link_label", ascending=False)
gradboost_df = aggregate_res(gradboost_result_path).sort_values("mean_link_label", ascending=False)
iso_df = aggregate_res(iso_result_path).sort_values("mean_link_label", ascending=False)

def calc_pk(df):
    df["actual_sum"] = df["actual"].cumsum()
    df["k"] = 1
    df["k"] = df["k"].cumsum()
    return df

rf_df = calc_pk(rf_df)
lr_df = calc_pk(lr_df)
adaboost_df = calc_pk(adaboost_df)
bagging_df = calc_pk(bagging_df)
baggingRF_df = calc_pk(baggingRF_df)
gradboost_df = calc_pk(gradboost_df)
iso_df = calc_pk(iso_df)

rf_df.head(5)

rf_df["p@k"] = rf_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
lr_df["p@k"] = lr_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
adaboost_df["p@k"] = adaboost_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
bagging_df["p@k"] = bagging_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
baggingRF_df["p@k"] = baggingRF_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
gradboost_df["p@k"] = gradboost_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)
iso_df["p@k"] = iso_df.apply(lambda x: x["actual_sum"]/x["k"], axis=1)

rf_df[["k", "p@k"]].head(5)


import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(20,10))

plt.plot(rf_df["k"][:500], rf_df["p@k"][:500], marker='',
         color=palette(0), linewidth=1, alpha=0.9, label="Random Forest")
plt.plot(lr_df["k"][:500], lr_df["p@k"][:500], marker='',
         color=palette(1), linewidth=1, alpha=0.9, label="Logistic Regression")
plt.plot(adaboost_df["k"][:500], adaboost_df["p@k"][:500], marker='',
         color=palette(2), linewidth=1, alpha=0.9, label="Adaboost")
plt.plot(bagging_df["k"][:500], bagging_df["p@k"][:500], marker='',
         color=palette(3), linewidth=1, alpha=0.9, label="Bagging")
plt.plot(baggingRF_df["k"][:500], baggingRF_df["p@k"][:500], marker='',
         color=palette(4), linewidth=1, alpha=0.9, label="Bagging Random FOrest")
plt.plot(gradboost_df["k"][:500], gradboost_df["p@k"][:500], marker='',
         color=palette(5), linewidth=1, alpha=0.9, label="Gradient Boosting")
plt.plot(iso_df["k"][:500], iso_df["p@k"][:500], marker='',
         color=palette(6), linewidth=1, alpha=0.9, label="Isolation Forest")

plt.legend(fontsize=14)
plt.title("Precision @ k for Different First-Stage Models", loc='left', fontsize=20, fontweight=4, color='black')
plt.xlabel("k", fontsize=14, fontweight=2)
plt.ylabel("precision @ k", fontsize=14, fontweight=2)

plt.savefig('precision_at_k.png')
