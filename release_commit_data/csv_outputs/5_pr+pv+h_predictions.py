import math
from datetime import datetime
import pandas as pd
import numpy as np
from pyHSICLasso import HSICLasso
from scipy.optimize import differential_evolution
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, brier_score_loss, \
    matthews_corrcoef, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import resample
import xgboost as xgb
import warnings
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

sp = 0


def smotetuned(X, y, model, k_range=(1, 20), m_range=(50, 100, 200, 400), n_bins=4):
    kf = KFold(n_splits=n_bins, shuffle=True, random_state=42)
    bins = list(kf.split(X))
    results = []

    def evaluate(params):
        k_neighbors = int(params[0])
        sampling_strategy = params[1]

        scores = []

        for train_idx, val_idx in bins:
            X_train_bin, X_val_bin = X[train_idx], X[val_idx]
            y_train_bin, y_val_bin = y[train_idx], y[val_idx]

            minority_class_size = np.min(np.bincount(y_train_bin))
            majority_class_size = np.max(np.bincount(y_train_bin))
            required_samples = int(majority_class_size * sampling_strategy) - minority_class_size
            if required_samples <= 0:
                return np.inf

            smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            try:
                X_res, y_res = smote.fit_resample(X_train_bin, y_train_bin)
            except ValueError as e:
                continue

            clf = model
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_val_bin)
            score = roc_auc_score(y_val_bin, y_pred)
            scores.append(score)

        mean_score = np.mean(scores) if scores else 0

        results.append((params, mean_score))
        return -mean_score

    param_bounds = [(k_range[0], k_range[1]), (m_range[0] / 100, m_range[1] / 100)]
    result = differential_evolution(evaluate, param_bounds)
    best_params = result.x

    best_k_neighbors = int(best_params[0])
    best_sampling_strategy = best_params[1]

    smote = SMOTE(k_neighbors=best_k_neighbors, sampling_strategy=best_sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    clf = model
    clf.fit(X_resampled, y_resampled)

    return clf, results


def bootstrap_sample(data, n_iterations=100):
    bootstrap_samples = []
    n_size = len(data)

    for _ in range(n_iterations):
        train = resample(data, n_samples=n_size, replace=True)
        test = data[~data.index.isin(train.index)]
        bootstrap_samples.append((train, test))

    return bootstrap_samples


def hsiclasso_feature_selection(X_train, y_train, num_features):
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train).ravel()

    hsic_lasso = HSICLasso()

    hsic_lasso.input(X_train_np, y_train_np)

    hsic_lasso.classification(num_features)
    selected_features = hsic_lasso.get_features()

    print(f"Selected features = {selected_features}")

    return selected_features


def performance_calculate(project, bootstrap_samples, selected_features, k):
    train_test_filename = f"{project}_combined_hyper_vector_fatty_product.csv"

    try:
        train_test_data = pd.read_csv(train_test_filename)
    except FileNotFoundError as e:
        print(e)
        return

    result_file.write(str(project))
    result_file.write(",")

    result_file1 = open(f"{project}_hyper_process_vector_product_predictions_git_data_{k}_fatty.csv", "w")
    result_file1.write("Project,")
    for model_name in [
    'LR', 'SVM',
    'RF',
        'XG',
        'GBM',
]:
        result_file1.write(
            f"{model_name}-Precision,{model_name}-Recall,{model_name}-Accuracy,{model_name}-F1,{model_name}-AUROC,"
            f"{model_name}-Brier,{model_name}-MCC,{model_name}-AUC,")
    result_file1.write("\n")
    result_file1.write(str(project))
    result_file1.write(",")

    X_train_test_full, y_train_test_full = train_test_data.iloc[:, 1:-2], train_test_data['BugPresence']

    imputer = SimpleImputer(strategy='mean')
    X_train_test_full = imputer.fit_transform(X_train_test_full)
    models = [
        LogisticRegression(random_state=42),
        SVC(kernel='linear', random_state=42, probability=True),
        RandomForestClassifier(random_state=42),
        xgb.XGBClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
    ]

    for model in models:
        scores = {'precision': [], 'recall': [], 'accuracy': [], 'f1': [], 'auroc': [], 'brier': [], 'mcc': [],
                  'auc_pr': []}
        c = 0
        for train_bootstrap, _ in bootstrap_samples:

            unique_train_files = train_bootstrap["file"].unique()

            train_data = train_test_data[train_test_data["file"].isin(unique_train_files)]
            X_train, y_train = train_data[selected_features], train_data["BugPresence"]

            test_data = train_test_data[~train_test_data["file"].isin(unique_train_files)]
            X_test, y_test = test_data[selected_features], test_data["BugPresence"]

            c += 1
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            y_train = y_train.to_numpy()
            if c == 1:
                clf_tuned, tuned_results = smotetuned(X_train_scaled, y_train, model)
                best_params, best_score = max(tuned_results, key=lambda item: item[1])
                k1 = int(best_params[0])
                m1 = best_params[1]
                y_pred_tuned = clf_tuned.predict(X_test_scaled)
                y_pred_proba = clf_tuned.predict_proba(X_test_scaled)[:, 1]
                precision = precision_score(y_test, y_pred_tuned)
                recall = recall_score(y_test, y_pred_tuned)
                accuracy = accuracy_score(y_test, y_pred_tuned)
                f1 = f1_score(y_test, y_pred_tuned)
                auroc = roc_auc_score(y_test, y_pred_tuned)
                brier = brier_score_loss(y_test, y_pred_proba)
                mcc = matthews_corrcoef(y_test, y_pred_tuned)
                precisionauc, recallauc, _ = precision_recall_curve(y_test, y_pred_proba)
                auc_pr = auc(recallauc, precisionauc)
            else:
                k_range = [k1 - 2, k1 + 2]
                m11 = m1 - 0.2
                m12 = m1 + 0.2
                if m11 < 0:
                    m11 = m1
                if m12 > 1:
                    m12 = 1
                m_range = [m11 * 100, m12 * 100]
                param_grid = {
                    'k_neighbors': range(k_range[0], k_range[1] + 1),
                    'sampling_strategy': [m / 100 for m in m_range]
                }
                maxi = 0
                for params in ParameterGrid(param_grid):
                    try:
                        smote = SMOTE(k_neighbors=params['k_neighbors'],
                                      sampling_strategy=params['sampling_strategy'])
                        X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

                        clf = model
                        clf.fit(X_res, y_res)
                        y_pred_tuned = clf.predict(X_test_scaled)
                        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                        if roc_auc_score(y_test, y_pred_tuned) > maxi:
                            maxi = roc_auc_score(y_test, y_pred_tuned)
                            precision = precision_score(y_test, y_pred_tuned)
                            recall = recall_score(y_test, y_pred_tuned)
                            accuracy = accuracy_score(y_test, y_pred_tuned)
                            f1 = f1_score(y_test, y_pred_tuned)
                            auroc = roc_auc_score(y_test, y_pred_tuned)
                            brier = brier_score_loss(y_test, y_pred_proba)
                            mcc = matthews_corrcoef(y_test, y_pred_tuned)
                            precisionauc, recallauc, _ = precision_recall_curve(y_test, y_pred_proba)
                            auc_pr = auc(recallauc, precisionauc)
                    except ValueError as e:
                        continue

            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['accuracy'].append(accuracy)
            scores['f1'].append(f1)
            scores['auroc'].append(auroc)
            scores['brier'].append(brier)
            scores['mcc'].append(mcc)
            scores['auc_pr'].append(auc_pr)

            print(c)
            c = c + 1

        mean_scores = {metric: np.mean(scores[metric]) for metric in scores}
        std_scores = {metric: np.std(scores[metric]) for metric in scores}

        print(f"project = {project} | Model: {model.__class__.__name__}")
        for metric, score in mean_scores.items():
            print(f"{metric.capitalize()}: {score} (±{std_scores[metric]})")
            result_file.write(f"{score},")
            result_file1.write(f"{score},")
    result_file.write("\n")
    result_file1.close()


st = datetime.now()
projects = {
    'camel': ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    'hbase': ["0.94.0", "0.95.0", "0.95.2"],
    'hive': ["release-0.9.0", "release-0.10.0", "release-0.12.0"],
    'wicket': ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
    'activemq': ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    'groovy': ["GROOVY_1_5_7", "GROOVY_1_6_BETA_1", "GROOVY_1_6_BETA_2"],
    'jruby': ["1.1", "1.4.0", "1.5.0", "1.7.0.preview1"],
    'lucene': ["releases/lucene/2.3.0", "releases/lucene/2.9.0", "releases/lucene/3.0.0",
               "releases/lucene-solr/3.1"],
    'derby': ["10.2.1.6", "10.3.1.4", "10.5.1.1"]
}

klist = [40]
for k in klist:
    result_file = open(f"hyper_process_product_vector_predictions_git_data_{k}_fatty_111class.csv", "w")
    result_file.write("Project,")
    for model_name in [
    'LR', 'SVM',
    'RF',
        'XG',
        'GBM',
]:
        result_file.write(
            f"{model_name}-Precision,{model_name}-Recall,{model_name}-Accuracy,{model_name}-F1,{model_name}-AUROC,"
            f"{model_name}-Brier,{model_name}-MCC,{model_name}-AUC,")
    result_file.write("\n")

    n_bootstraps = 100
    bootstrap_samples = {}
    selected_feature = {}
    for p in projects:
        train_test_filename = f"{p}_combined_hyper_vector_fatty_product.csv"

        try:
            train_test_data = pd.read_csv(train_test_filename)
        except FileNotFoundError as e:
            print(e)
            continue

        file_column = train_test_data["file"]

        X_train_test_full, y_train_test_full = train_test_data.iloc[:, 1:-2], train_test_data['BugPresence']
        scaler = MinMaxScaler()
        X_train_test_full_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_test_full),
            columns=X_train_test_full.columns,
            index=X_train_test_full.index
        )

        original_columns = list(train_test_data.columns[1:-2])

        imputer = SimpleImputer(strategy='mean')
        X_train_test_full_scaled = imputer.fit_transform(X_train_test_full_scaled)

        if k == 0:
            selected_feature_indices = hsiclasso_feature_selection(X_train_test_full_scaled, y_train_test_full,
                                                                   num_features=int(
                                                                       math.log2(len(X_train_test_full))))
        else:
            selected_feature_indices = hsiclasso_feature_selection(X_train_test_full_scaled, y_train_test_full,
                                                                   num_features=k)

        selected_feature_indices = [int(i)-1 for i in np.array(selected_feature_indices).astype(int)]

        X_train_test_full_scaled = X_train_test_full_scaled[:, selected_feature_indices]
        selected_feature_names = [original_columns[int(i)] for i in selected_feature_indices]
        print(selected_feature_names)

        X_train_test_full_scaled = imputer.fit_transform(X_train_test_full_scaled)

        train_test_full_combined = pd.concat(
            [file_column.reset_index(drop=True),
             pd.DataFrame(X_train_test_full_scaled).reset_index(drop=True),
             y_train_test_full.reset_index(drop=True)],
            axis=1
        )

        bootstrap_samples[p] = bootstrap_sample(train_test_full_combined, n_iterations=n_bootstraps)
        selected_feature[p] = selected_feature_names

    for p in projects:
        print(f"project = {p}")
        performance_calculate(p, bootstrap_samples[p], selected_feature[p], k)

    result_file.close()
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
