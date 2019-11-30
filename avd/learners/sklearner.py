import numpy as np
import pandas as pd

from sklearn import linear_model, svm, tree, ensemble, feature_extraction, preprocessing
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest

from avd.configs.config import *
from avd.learners import AbstractLearner
from avd.utils.dataset import DataSetFactory, DataSet
from avd.utils.exceptions import *


def dict_to_array(my_dict):
    vec = feature_extraction.DictVectorizer()
    return vec.fit_transform(my_dict).toarray()


def encode_labels(labels):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(labels)


class SkLearner(AbstractLearner):
    def __init__(self, classifier=None, labels=None, params=None):
        super(SkLearner, self).__init__(classifier)
        self._params = params
        if labels is not None and isinstance(labels, dict):
            if len(labels) == 2:
                self.fit_labels([labels['neg'], labels['pos']])
            else:
                raise NonBinaryLabels("Must be only two labels, negative and positive")
        else:
            self._label_encoder = labels

    def merge_with_labels(self, classified, labels_path, merge_col_name="id", default_label=0):
        node_labels = pd.read_csv(labels_path, dtype={"id": str})
        labels = node_labels.pop("label").values
        node_labels["actual"] = labels
        merged_data = pd.merge(classified, node_labels, left_on='src_id', right_on=merge_col_name, how='left')
        merged_data = merged_data.drop(["id"], axis=1)
        merged_data["actual"].fillna(default_label, inplace=True)
        merged_data["actual"] = self.transform_labels(merged_data["actual"])
        return merged_data

    @staticmethod
    def fit_labels(labels):
        return label_encoder.fit(*labels)

    @staticmethod
    def transform_labels(labels):
        return label_encoder.transform(labels)

    @staticmethod
    def inverse_transform_labels(labels):
        return label_encoder.inverse_transform(labels)

    def convert_data_to_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=None):
        return DataSetFactory().convert_data_to_sklearn_format(features, labels, feature_id_col_name, metadata_cols)

    def set_decision_tree_classifier(self, tree_number=100):
        params = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 40, 60, 80, 100, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [100, 200, 400, 600, 800, 1000]}
        return SkLearner(tree.DecisionTreeClassifier(random_state=42), params=params)

    def set_svm_classifier(self):
        params = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'gammas': [0.001, 0.01, 0.1, 1],
                  'tol': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                  'kernel': ['rbf', 'linear']}
        return SkLearner(svm.SVC(random_state=42), labels=self._label_encoder, params=params)

    def set_randomforest_classifier(self):
        params = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 40, 60, 80, 100, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [100, 200, 400, 600, 800, 1000]}
        return SkLearner(ensemble.RandomForestClassifier(random_state=42, n_jobs=-1, criterion="entropy"), params=params)

    def set_adaboost_classifier(self):
        params = {'n_estimators': [50, 100, 200, 400, 600, 800, 1000],
                  'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                  'algorithm': ['SAMME', 'SAMME.R']}
        return SkLearner(ensemble.AdaBoostClassifier(random_state=42), params=params)

    def set_rf_bagging_classifier(self):
        params = {'base_estimator__max_depth' : [10, 20, 40, 60, 80, 100, None],
                  'base_estimator__n_estimators' : [100, 200, 400, 600, 800, 1000],
                  'n_estimators': [50, 100, 200, 400, 600, 800, 1000],
                  'max_samples' : [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                  'bootstrap': [True, False],
                  'oob_score': [True, False]}
        return SkLearner(ensemble.BaggingClassifier(
            ensemble.RandomForestClassifier(criterion="entropy"), random_state=42, n_jobs=-1), params=params)

    def set_bagging_classifier(self):
        params = {'base_estimator__max_depth' : [10, 20, 40, 60, 80, 100, None],
                  'base_estimator__n_estimators' : [100, 200, 400, 600, 800, 1000],
                  'n_estimators': [50, 100, 200, 400, 600, 800, 1000],
                  'max_samples' : [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                  'bootstrap': [True, False],
                  'oob_score': [True, False]}
        return SkLearner(ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), random_state=42, n_jobs=-1), params=params)

    def set_gradient_boosting_classifier(self):
        params = {'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'max_features': ['auto', 'sqrt'],
                  'subsample':[0.7, 0.8, 0.9, 1],
                  'n_estimators': [50, 100, 200, 400, 600, 800, 1000],
                  'max_depth': [10, 20, 40, 60, 80, 100, None]}
        return SkLearner(ensemble.GradientBoostingClassifier(loss='exponential', random_state=42), params=params)

    def set_logistic_regression_classifier(self):
        params = {'dual': [True, False],
                  'max_iter': [75, 100, 125, 150],
                  'C': [0.5, 1.0, 1.5, 2.0, 2.5],
                  'penalty': ['l1', 'l2'],
                  'tol': [0.00001, 0.00005, 0.0001, 0.0005, 0.001]}
        return SkLearner(linear_model.LogisticRegression(random_state=42, solver='lbfgs', multi_class='ovr'), params=params)

    def set_isolation_forest_classifier(self):
        '''
        Deprecated for now, no meaningful results - performance metrics were similar to baseline results.
        '''
        return SkLearner(ensemble.IsolationForest(max_samples=100, random_state=42, contamination=0.1))

    def optimize_hyperparameters(self):
        '''
        Performs hyperparameter optimization using the RandomizedSearchCV library.
        Random search will have a 99% chance probability of finding a combination of hyperparameters 
        within the 5% optima with 90 iterations; also avoids issues related to finding local optima.
        Formula: $\P(X_{opt}) = 1 - (1-0.05)^n$, for $n$ independently sampled random search iterations
        '''
        if self._params:
            # print(self._classifier.get_params().keys())
            return RandomizedSearchCV(estimator=self._classifier, param_distributions=self._params, 
                                      n_iter=90, cv=2, verbose=2, random_state=42, n_jobs=-1)
        else:
            print("No hyperparameters specified. Using default hyperparameters for classifier.")
            return self._classifier
    
    def train_classifier(self, dataset):
        # RandomizedSearchCV to optimize parameters
        self._classifier = self.optimize_hyperparameters()
        # Train classifier on dataset
        print("Training classifier...")
        self._classifier = self._classifier.fit(dataset.features, dataset.labels).best_estimator_
        print("Optimized hyperparameters:\n" + self._classifier.get_params())
        return self

    def get_prediction(self, prediction_data):
        if isinstance(prediction_data, DataSet):
            return self._classifier.predict(prediction_data.features)
        else:
            return self._classifier.predict(prediction_data)

    def get_prediction_probabilities(self, prediction_data):
        if isinstance(prediction_data, DataSet):
            return self._classifier.predict_proba(prediction_data.features)
        else:
            return self._classifier.predict_proba(prediction_data)

    def split_kfold(self, features, labels=None, n_folds=10):
        skf = StratifiedKFold(n_folds)
        for train_index, test_index in skf.split(features, labels):
            yield train_index, test_index

    def get_classification_metrics(self, l_test, prediction, probas):
        false_positive = float(
            len(np.where(l_test - prediction == -1)[0]))  # 0 (truth) - 1 (prediction) == -1 which is a false positive
        true_negative = float(
            len(np.where(l_test + prediction == 0)[0]))  # 0 (truth) - 0 (prediction) == 0 which is a true positive

        metrics = {"auc": roc_auc_score(l_test, probas),
                   "recall": recall_score(l_test, prediction),
                   "precision": precision_score(l_test, prediction),
                   "accuracy": accuracy_score(l_test, prediction),
                   "fpr": false_positive / (true_negative + false_positive),
                   "tnr": true_negative / (true_negative + false_positive)}
        return metrics

    def cross_validate(self, dataset, n_folds=10):
        roc_auc, recall, precision, accuracy, fpr, tnr, f1 = [], [], [], [], [], [], []
        for train_index, test_index in self.split_kfold(dataset.features, dataset.labels, n_folds):
            f_train, f_test = dataset.features[train_index], dataset.features[test_index]
            l_train, l_test = dataset.labels[train_index], dataset.labels[test_index]
            model = self.train_classifier(DataSet(f_train, l_train))
            prediction = model.get_prediction(f_test)
            probas = model.get_prediction_probabilities(f_test)[:, 1]
            metrics = self.get_classification_metrics(l_test, prediction, probas)
            accuracy.append(metrics["accuracy"])
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])  # TPR
            f1.append(metrics["f1"])
            roc_auc.append(metrics["auc"])
            fpr.append(metrics["fpr"])
            tnr.append(metrics["tnr"])
        return {
                "accuracy": np.mean(accuracy),
                "precision": np.mean(precision),
                "recall": np.mean(recall),
                "f1": np.mean(f1),
                "auc": np.mean(roc_auc),
                "fpr": np.mean(fpr),
                "tnr": np.mean(tnr)
                }

    def get_evaluation(self, data):
        prediction = self.get_prediction(data)
        probas = self.get_prediction_probabilities(data)[:, 1]
        data.merge_dataset_with_predictions(prediction)
        return self.get_classification_metrics(data.labels, prediction, probas)

    def validate_prediction_by_links(self, prediction):
        roc_auc, recall, precision, accuracy, fpr, tnr, f1 = [], [], [], [], [], [], []
        try:
            metrics = self.get_classification_metrics( \
                                        prediction["actual"].values, \
                                        prediction["predicted_label"].values, \
                                        prediction["pos probability"].values)
            accuracy.append(metrics["accuracy"])
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])  # TPR
            f1.append(metrics["f1"])
            roc_auc.append(metrics["auc"])
            fpr.append(metrics["fpr"])
        except ValueError:
            print("Error")
        return {
                "accuracy": np.mean(accuracy),
                "precision": np.mean(precision),
                "recall": np.mean(recall),
                "f1": np.mean(f1),
                "auc": np.mean(roc_auc),
                "fpr": np.mean(fpr)
                }

    def classify_by_links_probability(self, probas, features_ids, labels=None, threshold=0.5):
        if not labels:
            labels = {"neg": 0, "pos": 1}
        train_df = pd.DataFrame(probas)
        train_df["src_id"] = pd.DataFrame(features_ids)
        train_df["link_label"] = train_df[0].apply(lambda avg: 1 if avg <= threshold else 0)
        train_df = train_df.groupby("src_id", as_index=False).agg(
            {0: ['mean', 'count'], 1: 'mean', "link_label": ['mean', 'sum']})
        train_df.columns = ['src_id', "neg probability", 'edge number', "pos probability", 'mean_link_label',
                            'sum_link_label']
        train_df["predicted_label"] = train_df["pos probability"].apply(
            lambda avg: labels["pos"] if avg >= threshold else labels["neg"])
        return train_df
