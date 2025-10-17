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

for project, tags in all_tags.items():
    dfs = []
    max_counts = {"VCB": 0, "VCC": 0, "VCD": 0, "VCE": 0}

    for tag in tags:
        tag = tag.replace("/", "_")
        file_path = os.path.join(DATA_DIR, f"{project}_{tag}_fatty_product.csv")
        df = pd.read_csv(file_path)
        for col in df.columns:
            m = vc_pattern.match(col)
            if m:
                prefix, idx = m.groups()
                idx = int(idx)
                if idx > max_counts[prefix]:
                    max_counts[prefix] = idx

    for tag in tags:
        tag = tag.replace("/", "_")
        file_path = os.path.join(DATA_DIR, f"{project}_{tag}_fatty_product.csv")
        df = pd.read_csv(file_path)

        for prefix, max_idx in max_counts.items():
            for i in range(1, max_idx + 1):
                col_name = f"{prefix}{i}"
                if col_name not in df.columns:
                    df[col_name] = 0

        bug_cols = ["BugCount", "BugPresence"]
        vc_cols = [f"{prefix}{i}" for prefix in ["VCB", "VCC", "VCD", "VCE"]
                   for i in range(1, max_counts[prefix] + 1)]
        other_cols = [c for c in df.columns if c not in vc_cols + bug_cols]

        df = df[other_cols + vc_cols + bug_cols]

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, f"{project}_combined_fatty_product.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

