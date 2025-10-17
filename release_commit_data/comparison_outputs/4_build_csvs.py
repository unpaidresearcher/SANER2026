import os
import pickle
import pandas as pd
import networkx as nx
from datetime import datetime


class Task:

    def __init__(self):
        self.graph = nx.Graph()

    def load_pickles(self, project, t1):
        self.graph.clear()
        PROCESS_METRIC_ORDER = [
            "comm", "adev", "ddev", "add", "del", "own", "minor", "sctr", "ncomm", "nadev", "nddev", "nsctr", "oexp",
            "exp"
        ]

        PRODUCT_METRICS_ORDER = [
            "CountDeclMethodPrivate", "AvgLineCode", "CountLine", "MaxCyclomatic",
            "CountDeclMethodDefault", "AvgEssential", "CountDeclClassVariable",
            "SumCyclomaticStrict", "AvgCyclomatic", "AvgLine", "CountDeclClassMethod",
            "AvgLineComment", "AvgCyclomaticModified", "CountDeclFunction",
            "CountLineComment", "CountDeclClass", "CountDeclMethod",
            "SumCyclomaticModified", "CountLineCodeDecl", "CountDeclMethodProtected",
            "CountDeclInstanceVariable", "MaxCyclomaticStrict", "CountDeclMethodPublic",
            "CountLineCodeExe", "SumCyclomatic", "SumEssential", "CountStmtDecl",
            "CountLineCode", "CountStmtExe", "RatioCommentToCode", "CountLineBlank",
            "CountStmt", "MaxCyclomaticModified", "CountSemicolon", "AvgLineBlank",
            "CountDeclInstanceMethod", "AvgCyclomaticStrict", "PercentLackOfCohesion",
            "MaxInheritanceTree", "CountClassDerived", "CountClassCoupled",
            "CountClassBase", "CountInput_Max", "CountInput_Mean", "CountInput_Min",
            "CountOutput_Max", "CountOutput_Mean", "CountOutput_Min", "CountPath_Max",
            "CountPath_Mean", "CountPath_Min", "MaxNesting_Max", "MaxNesting_Mean",
            "MaxNesting_Min"
        ]
        pickle_dir1 = os.path.join(f"{project}_{t1}_pickles")
        pickle_dir = ''

        bug_count = pickle.load(open(os.path.join(pickle_dir1, "file_bugs.pkl"), "rb"))
        product_metrics = pickle.load(open(os.path.join(pickle_dir1, "file_product_metrics.pkl"), "rb"))
        process_metrics = pickle.load(open(os.path.join(pickle_dir, f"{project}_process_metrics_git_data_fatty.pkl"), "rb"))
        centrality_betweenness = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_sorted_betweeness_git_data_fatty.pkl"), "rb"))
        centrality_closeness = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_sorted_closeness_git_data_fatty.pkl"), "rb"))
        centrality_degree = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_sorted_degree_git_data_fatty.pkl"), "rb"))
        centrality_eigenvector = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_sorted_eigenvector_git_data_fatty.pkl"), "rb"))
        vc_betweenness = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_vector_centrality_betweeness_git_data_fatty.pkl"), "rb"))
        vc_closeness = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_vector_centrality_closeness_git_data_fatty.pkl"), "rb"))
        vc_degree = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_vector_centrality_degree_git_data_fatty.pkl"), "rb"))
        vc_eigenvector = pickle.load(
            open(os.path.join(pickle_dir, f"{project}_{t1}_vector_centrality_git_data_fatty.pkl"), "rb"))

        file_sets = [
            set(centrality_betweenness.keys()),
            set(centrality_closeness.keys()),
            set(centrality_degree.keys()),
            set(centrality_eigenvector.keys()),
            set(vc_betweenness.keys()),
            set(vc_closeness.keys()),
            set(vc_degree.keys()),
            set(vc_eigenvector.keys()),
            set(bug_count.keys())
        ]

        pm_files = set()
        for metric_name in PROCESS_METRIC_ORDER:
            pm_files |= set(process_metrics.get(metric_name, {}).get(t1, {}).keys())

        file_sets.append(pm_files)

        common_files = sorted(set.intersection(*file_sets))

        all_files = common_files

        rows = []
        for file in all_files:
            row = {"file": file}

            for metric_name in PRODUCT_METRICS_ORDER:
                row[metric_name] = product_metrics.get(file, {}).get(metric_name, None)
            for metric_name in PROCESS_METRIC_ORDER:
                row[metric_name] = process_metrics.get(metric_name, {}).get(t1, {}).get(file, None)

            row["Betweenness"] = centrality_betweenness.get(file, None)
            row["Closeness"] = centrality_closeness.get(file, None)
            row["Degree"] = centrality_degree.get(file, None)
            row["Eigenvector"] = centrality_eigenvector.get(file, None)

            for idx, val in enumerate(vc_betweenness.get(file, []), 1):
                row[f"VCB{idx}"] = val
            for idx, val in enumerate(vc_closeness.get(file, []), 1):
                row[f"VCC{idx}"] = val
            for idx, val in enumerate(vc_degree.get(file, []), 1):
                row[f"VCD{idx}"] = val
            for idx, val in enumerate(vc_eigenvector.get(file, []), 1):
                row[f"VCE{idx}"] = val

            bc = bug_count.get(file, 0)
            row["BugCount"] = bc
            row["BugPresence"] = 1 if bc > 0 else 0

            rows.append(row)

        OUTPUT_DIR = os.path.join('..', "csv_outputs")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.DataFrame(rows)
        safe_tag = t1.replace("/", "_")
        df.to_csv(os.path.join(OUTPUT_DIR, f"{project}_{safe_tag}_fatty_product.csv"), index=False)

        print("CSVs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task()
    all_tags = {
        'activemq': ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
        'camel': ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
        'groovy': ["GROOVY_1_5_7", "GROOVY_1_6_BETA_1", "GROOVY_1_6_BETA_2"],
        'hbase': ["0.94.0", "0.95.0", "0.95.2"],
        'hive': ["release-0.9.0", "release-0.10.0", "release-0.12.0"],
        'jruby': ["1.1", "1.4.0", "1.5.0", "1.7.0.preview1"],
        'lucene': ["releases/lucene/2.3.0", "releases/lucene/2.9.0", "releases/lucene/3.0.0",
                   "releases/lucene-solr/3.1"],
        'wicket': ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
        'derby': ["10.2.1.6", "10.3.1.4", "10.5.1.1"]
    }
    for project in all_tags:
        for idx, t1 in enumerate(all_tags[project]):
            print(f"project = {project} | tag = {t1}")
            t.load_pickles(project, t1)

    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
