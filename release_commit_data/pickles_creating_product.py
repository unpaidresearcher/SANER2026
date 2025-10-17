import pandas as pd
import pickle
import os

projects = [
            'activemq', 'camel', 'groovy', 'hbase', 'hive', 'jruby',
            'lucene',
            'wicket',
            'derby'
            ]
all_tags = {
    'activemq': ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    'camel': ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    'groovy': ["GROOVY_1_5_7", "GROOVY_1_6_BETA_1", "GROOVY_1_6_BETA_2"],
    'hbase': ["0.94.0", "0.95.0", "0.95.2"],
    'hive': ["release-0.9.0", "release-0.10.0", "release-0.12.0"],
    'jruby': ["1.1", "1.4.0", "1.5.0", "1.7.0.preview1"],
    'lucene': ["releases/lucene/2.3.0", "releases/lucene/2.9.0", "releases/lucene/3.0.0", "releases/lucene-solr/3.1"],
    'wicket': ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
    'derby': ["10.2.1.6", "10.3.1.4", "10.5.1.1"]
}

all_tags_their = {
    'activemq': ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    'camel': ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    'groovy': ["groovy-1_5_7", "groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
    'hbase': ["hbase-0.94.0", "hbase-0.95.0", "hbase-0.95.2"],
    'hive': ["hive-0.9.0", "hive-0.10.0", "hive-0.12.0"],
    'jruby': ["jruby-1.1", "jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
    'lucene': ["lucene-2.3.0", "lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
    'wicket': ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
    'derby': ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"]
}

output_dir = "comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

metric_cols = ["COMM", "ADEV", "DDEV", "Added_lines", "Del_lines"]

product_metrics = [
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

for project in projects:
    print(f"\nProcessing {project}...")

    our_csv = pd.read_csv(f"{project}_tag_commits_diff.csv")
    all_common_rows = []

    for id, tag_their in enumerate(all_tags_their[project]):
        tag_our = all_tags[project][id]

        their_csv = pd.read_csv(f"{tag_their}.csv")

        non_zero_theirs_df = their_csv[(their_csv[metric_cols] != 0).all(axis=1)]
        non_zero_theirs_files = set(non_zero_theirs_df["File"].unique())

        our_tag_df = our_csv[our_csv["Tag"] == tag_our]
        our_tag_files = set(our_tag_df["File"].unique())

        common_files = non_zero_theirs_files & our_tag_files

        file_bugs = {}
        commits = set()
        commit_set_of_files = {}
        file_commits = {}
        commit_authors = {}
        file_authors = {}
        file_authors_distinct = {}
        file_lines_added = {}
        file_lines_deleted = {}
        author_lines_added = {}
        author_lines_deleted = {}
        author_entire_lines = {}
        commit_lines_added = {}
        commit_lines_deleted = {}
        file_commits_lines = {} 
        file_commits_lines_added = {}
        file_commits_lines_deleted = {}
        file_authors_lines = {} 
        file_commit_authors = {} 
        file_commit_author_lines = {}
        file_product_metrics = {}

        for file in common_files:
            bugs = non_zero_theirs_df.loc[non_zero_theirs_df["File"] == file, "RealBugCount"].sum()
            file_bugs[file] = bugs

            file_df_product = non_zero_theirs_df.loc[non_zero_theirs_df["File"] == file, product_metrics]

            metrics_dict = file_df_product.sum().to_dict()

            file_product_metrics[file] = metrics_dict

            file_df = our_tag_df[our_tag_df["File"] == file]

            file_lines_added[file] = file_df["Lines Added"].sum()
            file_lines_deleted[file] = file_df["Lines Deleted"].sum()
            file_commits[file] = set(file_df["Commit ID"])
            file_authors[file] = set(file_df["Author"])
            file_authors_distinct[file] = set(file_df["Author"])

            file_commits_lines[file] = {}
            file_commits_lines_added[file] = {}
            file_commits_lines_deleted[file] = {}
            file_authors_lines[file] = {}
            file_commit_authors[file] = {}
            file_commit_author_lines[file] = {}

            for commit_id in file_df["Commit ID"].unique():
                commit_df = file_df[file_df["Commit ID"] == commit_id]

                commits.add(commit_id)
                commit_set_of_files.setdefault(commit_id, set()).add(file)
                commit_authors.setdefault(commit_id, set()).update(commit_df["Author"])

                commit_lines_added[commit_id] = commit_lines_added.get(commit_id, 0) + commit_df["Lines Added"].sum()
                commit_lines_deleted[commit_id] = commit_lines_deleted.get(commit_id, 0) + commit_df[
                    "Lines Deleted"].sum()

                file_commits_lines[file][commit_id] = commit_df["Lines Added"].sum() + commit_df["Lines Deleted"].sum()
                file_commits_lines_added[file][commit_id] = commit_df["Lines Added"].sum()
                file_commits_lines_deleted[file][commit_id] = commit_df["Lines Deleted"].sum()

                file_commit_authors[file].setdefault(commit_id, set()).update(commit_df["Author"])

                if commit_id not in file_commit_author_lines[file]:
                    file_commit_author_lines[file][commit_id] = {}
                for author, group_df in commit_df.groupby("Author"):
                    file_commit_author_lines[file][commit_id][author] = (
                            group_df["Lines Added"].sum() + group_df["Lines Deleted"].sum()
                    )

            for author, group_df in file_df.groupby("Author"):
                author_lines_added[author] = author_lines_added.get(author, 0) + group_df["Lines Added"].sum()
                author_lines_deleted[author] = author_lines_deleted.get(author, 0) + group_df["Lines Deleted"].sum()

                file_authors_lines[file][author] = group_df["Lines Added"].sum() + group_df["Lines Deleted"].sum()

            all_common_rows.append({
                "Tag": tag_our,
                "File": file,
                "Bugs": bugs,
                "Commits": ";".join(sorted(file_commits[file])),
                "Authors": ";".join(sorted(file_authors[file])),
                "Authors Distinct": ";".join(sorted(file_authors_distinct[file])),
                "Lines Added": file_lines_added[file],
                "Lines Deleted": file_lines_deleted[file]
            })

        pickle_dir = os.path.join(output_dir, f"{project}_{tag_our}_pickles")
        os.makedirs(pickle_dir, exist_ok=True)

        pickle.dump(file_bugs, open(os.path.join(pickle_dir, "file_bugs.pkl"), "wb"))
        pickle.dump(commits, open(os.path.join(pickle_dir, "commits.pkl"), "wb"))
        pickle.dump(commit_set_of_files, open(os.path.join(pickle_dir, "commit_set_of_files.pkl"), "wb"))
        pickle.dump(file_commits, open(os.path.join(pickle_dir, "file_commits.pkl"), "wb"))
        pickle.dump(commit_authors, open(os.path.join(pickle_dir, "commit_authors.pkl"), "wb"))
        pickle.dump(file_authors, open(os.path.join(pickle_dir, "file_authors.pkl"), "wb"))
        pickle.dump(file_authors_distinct, open(os.path.join(pickle_dir, "file_authors_distinct.pkl"), "wb"))
        pickle.dump(file_lines_added, open(os.path.join(pickle_dir, "file_lines_added.pkl"), "wb"))
        pickle.dump(file_lines_deleted, open(os.path.join(pickle_dir, "file_lines_deleted.pkl"), "wb"))
        pickle.dump(author_lines_added, open(os.path.join(pickle_dir, "author_lines_added.pkl"), "wb"))
        pickle.dump(author_lines_deleted, open(os.path.join(pickle_dir, "author_lines_deleted.pkl"), "wb"))
        pickle.dump(commit_lines_added, open(os.path.join(pickle_dir, "commit_lines_added.pkl"), "wb"))
        pickle.dump(commit_lines_deleted, open(os.path.join(pickle_dir, "commit_lines_deleted.pkl"), "wb"))
        pickle.dump(file_commits_lines, open(os.path.join(pickle_dir, "file_commits_lines.pkl"), "wb"))
        pickle.dump(file_commits_lines_added, open(os.path.join(pickle_dir, "file_commits_lines_added.pkl"), "wb"))
        pickle.dump(file_commits_lines_deleted, open(os.path.join(pickle_dir, "file_commits_lines_deleted.pkl"), "wb"))
        pickle.dump(file_authors_lines, open(os.path.join(pickle_dir, "file_authors_lines.pkl"), "wb"))
        pickle.dump(file_commit_authors, open(os.path.join(pickle_dir, "file_commit_authors.pkl"), "wb"))
        pickle.dump(file_commit_author_lines, open(os.path.join(pickle_dir, "file_commit_author_lines.pkl"), "wb"))
        pickle.dump(file_product_metrics, open(os.path.join(pickle_dir, "file_product_metrics.pkl"), "wb"))

        print(f"Pickles saved for {project} | {tag_our}")

    csv_path = os.path.join(output_dir, f"{project}_common_files.csv")
    pd.DataFrame(all_common_rows).to_csv(csv_path, index=False)
    print(f"CSV saved for {project} → {csv_path}")
