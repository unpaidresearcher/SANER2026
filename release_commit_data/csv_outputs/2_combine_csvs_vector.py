import os
import pandas as pd
import re

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

DATA_DIR = ""

vc_pattern = re.compile(r'^(VC[B|C|D|E])(\d+)$')
pm_pattern = re.compile(r'^(comm|adev|ddev|add|del|own|minor|sctr|ncomm|nadev|nddev|nsctr|oexp|exp)(\d+)$')

PROCESS_METRIC_ORDER = [
    "comm", "adev", "ddev", "add", "del", "own", "minor", "sctr",
    "ncomm", "nadev", "nddev", "nsctr", "oexp", "exp"
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

for project, tags in all_tags.items():
    dfs = []
    max_counts_vc = {"VCB": 0, "VCC": 0, "VCD": 0, "VCE": 0}
    max_counts_pm = {pm: 0 for pm in PROCESS_METRIC_ORDER}
    tc_cols = ["Betweeness", "Closeness", "Degree", "Eigenvector"]

    for tag in tags:
        tag = tag.replace("/", "_")
        file_path = os.path.join(DATA_DIR, f"{project}_{tag}_vector_fatty_product.csv")
        df = pd.read_csv(file_path)

        for col in df.columns:
            m = vc_pattern.match(col)
            if m:
                prefix, idx = m.groups()
                idx = int(idx)
                if idx > max_counts_vc[prefix]:
                    max_counts_vc[prefix] = idx

        for col in df.columns:
            m = pm_pattern.match(col)
            if m:
                prefix, idx = m.groups()
                idx = int(idx)
                if idx > max_counts_pm[prefix]:
                    max_counts_pm[prefix] = idx

    for tag in tags:
        tag = tag.replace("/", "_")
        file_path = os.path.join(DATA_DIR, f"{project}_{tag}_vector_fatty_product.csv")
        df = pd.read_csv(file_path)

        for pm, max_idx in max_counts_pm.items():
            for i in range(1, max_idx + 1):
                col_name = f"{pm}{i}"
                if col_name not in df.columns:
                    df[col_name] = 0

        for prefix, max_idx in max_counts_vc.items():
            for i in range(1, max_idx + 1):
                col_name = f"{prefix}{i}"
                if col_name not in df.columns:
                    df[col_name] = 0

        bug_cols = ["BugCount", "BugPresence"]

        pm_cols = [f"{pm}{i}" for pm in PROCESS_METRIC_ORDER for i in range(1, max_counts_pm[pm] + 1)]

        vc_cols = [f"{prefix}{i}" for prefix in ["VCB", "VCC", "VCD", "VCE"]
                   for i in range(1, max_counts_vc[prefix] + 1)]

        other_cols = [c for c in df.columns if c not in pm_cols + vc_cols + bug_cols]

        df = df[other_cols + pm_cols + vc_cols + bug_cols]

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, f"{project}_combined_vector_fatty_product.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

for project in all_tags:
    file_path = os.path.join(DATA_DIR, f"{project}_combined_vector_fatty_product.csv")
    df = pd.read_csv(file_path)

    file_col = ["file"]
    pm_cols1 = ["comm", "adev", "ddev", "add", "del", "own", "minor",
                "sctr", "ncomm", "nadev", "nddev", "nsctr", "oexp", "exp"]

    pr_cols = [
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

    pm_final = []
    for base in pm_cols1:
        cols = [
            c for c in df.columns
            if c.startswith(base)
               and (c == base or c[len(base):].isdigit())
               and (c == base or int(c[len(base):]) <= 100)
        ]
        cols = sorted(
            cols,
            key=lambda x: int(x[len(base):]) if x != base else 0
        )
        pm_final.extend(cols)
    bug_cols = ["BugCount", "BugPresence"]

    vc_cols = []
    vc_suffixes = ['B', 'C', 'D', 'E']

    for suffix in vc_suffixes:
        cols = [
            c for c in df.columns
            if c.startswith(f"VC{suffix}")
            and c[len(f"VC{suffix}") :].isdigit()
            and int(c[len(f"VC{suffix}") :]) <= 100
        ]
        cols = sorted(cols, key=lambda x: int(x[len(f"VC{suffix}") :]))
        vc_cols.extend(cols)

    df_vc = df[file_col + pr_cols + pm_final + vc_cols + bug_cols]
    df_vp = df[file_col + pr_cols + pm_final + bug_cols]

    out_path = os.path.join(DATA_DIR, f"{project}_combined_hyper_vector_fatty_product.csv")
    df_vc.to_csv(out_path, index=False)
    out_path1 = os.path.join(DATA_DIR, f"{project}_combined_process_vector_fatty_product.csv")
    df_vp.to_csv(out_path1, index=False)
    print(f"Saved {out_path}\n{out_path1}")
    