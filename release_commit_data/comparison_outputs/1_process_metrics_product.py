import itertools
import math
import os
import pickle
import statistics
from pymongo import MongoClient
import networkx as nx
import datetime
from datetime import datetime


class Task:

    def __init__(self):
        self.graph = nx.Graph()
        self.weighted_graph = nx.Graph()
        self.process_metrics = {
            "comm": {},
            "adev": {},
            "ddev": {},
            "add": {},
            "del": {},
            "own": {},
            "minor": {},
            "sctr": {},
            "ncomm": {},
            "nadev": {},
            "nddev": {},
            "nsctr": {},
            "oexp": {},
            "exp": {}
        }

    def load_pickles(self, project, t1):
        self.graph.clear()
        self.weighted_graph.clear()
        pickle_dir = os.path.join(f"{project}_{t1}_pickles")

        file_bugs = pickle.load(open(os.path.join(pickle_dir, "file_bugs.pkl"), "rb"))
        commits = pickle.load(open(os.path.join(pickle_dir, "commits.pkl"), "rb"))
        commit_set_of_files = pickle.load(open(os.path.join(pickle_dir, "commit_set_of_files.pkl"), "rb"))
        file_commits = pickle.load(open(os.path.join(pickle_dir, "file_commits.pkl"), "rb"))
        commit_authors = pickle.load(open(os.path.join(pickle_dir, "commit_authors.pkl"), "rb"))
        file_authors = pickle.load(open(os.path.join(pickle_dir, "file_authors.pkl"), "rb"))
        file_authors_distinct = pickle.load(open(os.path.join(pickle_dir, "file_authors_distinct_total.pkl"), "rb"))
        file_lines_added = pickle.load(open(os.path.join(pickle_dir, "file_lines_added.pkl"), "rb"))
        file_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "file_lines_deleted.pkl"), "rb"))
        author_lines_added = pickle.load(open(os.path.join(pickle_dir, "author_lines_added.pkl"), "rb"))
        author_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "author_lines_deleted.pkl"), "rb"))
        author_lines_added_total = pickle.load(open(os.path.join(pickle_dir, "author_lines_added_total.pkl"), "rb"))
        author_lines_deleted_total = pickle.load(open(os.path.join(pickle_dir, "author_lines_deleted_total.pkl"), "rb"))
        commit_lines_added = pickle.load(open(os.path.join(pickle_dir, "commit_lines_added.pkl"), "rb"))
        commit_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "commit_lines_deleted.pkl"), "rb"))
        file_commits_lines = pickle.load(open(os.path.join(pickle_dir, "file_commits_lines.pkl"), "rb"))
        file_authors_lines = pickle.load(open(os.path.join(pickle_dir, "file_authors_lines.pkl"), "rb"))
        file_commit_authors = pickle.load(open(os.path.join(pickle_dir, "file_commit_authors.pkl"), "rb"))
        file_commit_author_lines = pickle.load(open(os.path.join(pickle_dir, "file_commit_author_lines.pkl"), "rb"))

        for commit in commits:
            files = commit_set_of_files[commit]
            ordered_pairs = {(x, y) for x in files for y in files if x != y}
            self.graph.add_edges_from(ordered_pairs)
            for x, y in ordered_pairs:
                if self.weighted_graph.has_edge(x, y):
                    self.weighted_graph[x][y]["weight"] += 1
                else:
                    self.weighted_graph.add_edge(x, y, weight=1)

        print(f"nodes = {len(self.graph.nodes())}, edges = {len(self.graph.edges())}")
        print(f"file_commits = {len(file_commits)}")
        print(f"file_authors = {len(file_authors)}")
        print(f"file_authors_distinct = {len(file_authors_distinct)}")
        HCPF2 = dict()
        file_commit_counts = {file: len(commits) for file, commits in file_commits.items()}
        total_changes = sum(file_commit_counts.values())
        probabilities = [count / total_changes for count in file_commit_counts.values()]
        entropy_Hi = -sum(p * math.log2(p) for p in probabilities if p > 0)
        HCPF2 = {file: (count / total_changes) * entropy_Hi for file, count in file_commit_counts.items()}
        total_lines_added = 0
        total_lines_deleted = 0
        for f in file_lines_added:
            total_lines_added = total_lines_added + file_lines_added[f]
        for f in file_lines_deleted:
            total_lines_deleted = total_lines_deleted + file_lines_deleted[f]
        author_lines = dict()
        for a in author_lines_added:
            if a not in author_lines:
                author_lines[a] = 0
            author_lines[a] = author_lines[a] + author_lines_added[a] + author_lines_deleted[a]
        total_author_lines = 0
        for a in author_lines:
            total_author_lines = total_author_lines + author_lines[a]
        author_lines_total = dict()
        for a in author_lines_added_total:
            if a not in author_lines_total:
                author_lines_total[a] = 0
            author_lines_total[a] = author_lines_total[a] + author_lines_added_total[a] + author_lines_deleted_total[a]
        total_author_lines_total = 0
        for a in author_lines_total:
            total_author_lines_total = total_author_lines_total + author_lines_total[a]

        comm = dict()
        adev = dict()
        ddev = dict()
        sctr = dict()
        ncomm = dict()
        nadev = dict()
        nddev = dict()
        nsctr = dict()
        add = dict()
        dele = dict()
        own = dict()
        minor = dict()
        oexp = dict()
        exp = dict()
        for i, f in enumerate(self.graph.nodes()):
            print(f"release = {t1} | {i + 1}/{len(self.graph.nodes())} - {datetime.now()}")
            comm[f] = len(file_commits[f])
            print(f"comm of {f} = {comm[f]}")
            adev[f] = len(file_authors[f])
            print(f"adev of {f} = {adev[f]}")
            ddev[f] = len(file_authors_distinct[f])
            print(f"ddev of {f} = {ddev[f]}")
            sctr[f] = HCPF2[f]
            print(f"sctr of {f} = {sctr[f]}")

            neighbors = list(self.graph.neighbors(f))
            commn = 0
            adevn = 0
            ddevn = 0
            sctrn = 0
            if len(neighbors) > 0:
                for n in neighbors:
                    if n in file_commits.keys():
                        commn = commn + (self.weighted_graph[f][n]["weight"]) * (len(file_commits[n]))
                        adevn = adevn + (self.weighted_graph[f][n]["weight"]) * (len(file_authors[n]))
                        ddevn = ddevn + (self.weighted_graph[f][n]["weight"]) * (len(file_authors_distinct[n]))
                        sctrn = sctrn + (self.weighted_graph[f][n]["weight"]) * (HCPF2[n])
                    else:
                        print(f"non src file = {n}")
                ncomm[f] = commn
                nadev[f] = adevn
                nddev[f] = ddevn
                nsctr[f] = sctrn
            else:
                ncomm[f] = 0
                nadev[f] = 0
                nddev[f] = 0
                nsctr[f] = 0
            print(f"ncomm of {f} = {ncomm[f]}")
            print(f"nadev of {f} = {nadev[f]}")
            print(f"nddev of {f} = {nddev[f]}")
            print(f"nsctr of {f} = {nsctr[f]}")

            add[f] = file_lines_added[f] / total_lines_added
            print(f"add of {f} = {add[f]}")
            dele[f] = file_lines_deleted[f] / total_lines_deleted
            print(f"del of {f} = {dele[f]}")

            authors = file_authors[f]
            max_lines = 0
            total_lines = 0
            lines_file = file_lines_added[f] + file_lines_deleted[f]
            authors_list = list()
            exp_author_lines = 0
            mean_list = list()
            for a in authors:
                if max_lines < file_authors_lines[f][a]:
                    max_author = a
                    max_lines = file_authors_lines[f][a]

                mean_list.append(author_lines_total[a] / total_author_lines_total)

                if file_authors_lines[f][a] <= (lines_file * 5) / 100:
                    authors_list.append(a)

            own[f] = file_authors_lines[f][max_author] / lines_file
            print(f"own of {f} = {own[f]}")
            minor[f] = len(authors_list)
            print(f"minor of {f} = {minor[f]}")

            oexp[f] = author_lines_total[max_author] / total_author_lines_total
            print(f"oexp of {f} = {oexp[f]}")
            exp[f] = statistics.geometric_mean(mean_list)
            print(f"exp of {f} = {exp[f]}")

        self.process_metrics['comm'][t1] = comm
        self.process_metrics['adev'][t1] = adev
        self.process_metrics['ddev'][t1] = ddev
        self.process_metrics['sctr'][t1] = sctr
        self.process_metrics['ncomm'][t1] = ncomm
        self.process_metrics['nadev'][t1] = nadev
        self.process_metrics['nddev'][t1] = nddev
        self.process_metrics['nsctr'][t1] = nsctr
        self.process_metrics['add'][t1] = add
        self.process_metrics['del'][t1] = dele
        self.process_metrics['own'][t1] = own
        self.process_metrics['minor'][t1] = minor
        self.process_metrics['oexp'][t1] = oexp
        self.process_metrics['exp'][t1] = exp


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
        with open(f'{project}_process_metrics_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(t.process_metrics, f)
        for pr in t.process_metrics:
            for pr1 in t.process_metrics[pr]:
                print(pr1, len(t.process_metrics[pr][pr1]))

    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
