import itertools
import math
import os
import pickle
import statistics
from collections import defaultdict

import numpy as np
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
        commits_release = pickle.load(open(os.path.join(pickle_dir, "commits.pkl"), "rb"))
        commits_total = pickle.load(open(os.path.join(pickle_dir, "commits_total.pkl"), "rb"))
        commit_set_of_files = pickle.load(open(os.path.join(pickle_dir, "commit_set_of_files.pkl"), "rb"))
        commit_set_of_files_total = pickle.load(open(os.path.join(pickle_dir, "commit_set_of_files_total.pkl"), "rb"))
        file_commits = pickle.load(open(os.path.join(pickle_dir, "file_commits.pkl"), "rb"))
        file_commits_total = pickle.load(open(os.path.join(pickle_dir, "file_commits_total.pkl"), "rb"))
        commit_authors = pickle.load(open(os.path.join(pickle_dir, "commit_authors.pkl"), "rb"))
        commit_authors_total = pickle.load(open(os.path.join(pickle_dir, "commit_authors_total.pkl"), "rb"))
        file_authors = pickle.load(open(os.path.join(pickle_dir, "file_authors.pkl"), "rb"))
        file_authors_distinct = pickle.load(open(os.path.join(pickle_dir, "file_authors_distinct_total.pkl"), "rb"))
        file_lines_added = pickle.load(open(os.path.join(pickle_dir, "file_lines_added.pkl"), "rb"))
        file_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "file_lines_deleted.pkl"), "rb"))
        author_lines_added = pickle.load(open(os.path.join(pickle_dir, "author_lines_added.pkl"), "rb"))
        author_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "author_lines_deleted.pkl"), "rb"))
        author_lines_added_total = pickle.load(open(os.path.join(pickle_dir, "author_lines_added_total.pkl"), "rb"))
        author_lines_deleted_total = pickle.load(open(os.path.join(pickle_dir, "author_lines_deleted_total.pkl"), "rb"))
        commit_lines_added = pickle.load(open(os.path.join(pickle_dir, "commit_lines_added.pkl"), "rb"))
        commit_lines_added_total = pickle.load(open(os.path.join(pickle_dir, "commit_lines_added_total.pkl"), "rb"))
        commit_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "commit_lines_deleted.pkl"), "rb"))
        commit_lines_deleted_total = pickle.load(open(os.path.join(pickle_dir, "commit_lines_deleted_total.pkl"), "rb"))
        file_commits_lines = pickle.load(open(os.path.join(pickle_dir, "file_commits_lines.pkl"), "rb"))
        file_commits_lines_added = pickle.load(open(os.path.join(pickle_dir, "file_commits_lines_added.pkl"), "rb"))
        file_commits_lines_deleted = pickle.load(open(os.path.join(pickle_dir, "file_commits_lines_deleted.pkl"), "rb"))
        file_authors_lines = pickle.load(open(os.path.join(pickle_dir, "file_authors_lines.pkl"), "rb"))
        file_commit_authors = pickle.load(open(os.path.join(pickle_dir, "file_commit_authors.pkl"), "rb"))
        file_commit_author_lines = pickle.load(open(os.path.join(pickle_dir, "file_commit_author_lines.pkl"), "rb"))
        commit_author_lines_total = pickle.load(open(os.path.join(pickle_dir, "commit_author_lines_total.pkl"), "rb"))
        file_commit_authors_total = pickle.load(open(os.path.join(pickle_dir, "file_commit_authors_total.pkl"), "rb"))
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
        author_lines1 = dict()
        for a in author_lines_added:
            if a not in author_lines1:
                author_lines1[a] = 0
            author_lines1[a] = author_lines1[a] + author_lines_added[a] + author_lines_deleted[a]
        total_author_lines = 0
        for a in author_lines1:
            total_author_lines = total_author_lines + author_lines1[a]
        author_lines_total1 = dict()
        for a in author_lines_added_total:
            if a not in author_lines_total1:
                author_lines_total1[a] = 0
            author_lines_total1[a] = author_lines_total1[a] + author_lines_added_total[a] + author_lines_deleted_total[a]
        total_author_lines_total = 0
        for a in author_lines_total1:
            total_author_lines_total = total_author_lines_total + author_lines_total1[a]

        for commit in commits_release:
            files = commit_set_of_files[commit]
            ordered_pairs = {(x, y) for x in files for y in files if x != y}
            self.graph.add_edges_from(ordered_pairs)
            for x, y in ordered_pairs:
                if self.weighted_graph.has_edge(x, y):
                    self.weighted_graph[x][y]["weight"] += 1
                else:
                    self.weighted_graph.add_edge(x, y, weight=1)

        file_commit_vector = {}
        file_commit_vector_size = {}
        file_commit_vector_size_norm = {}
        max_commit_size = 0

        # 1. Find maximum commit size across all commits
        for commits in file_commits.values():
            for commit in commits:
                commit_size = len(commit_set_of_files[commit])
                max_commit_size = max(max_commit_size, commit_size)

        print("Maximum commit size across all files:", max_commit_size)

        # 2. Build commit size vectors (counts + commit IDs)
        for file, commits in file_commits.items():
            size_vector_counts = [0] * max_commit_size
            size_vector_commits = [[] for _ in range(max_commit_size)]

            for commit in commits:
                commit_size = len(commit_set_of_files[commit])
                idx = commit_size - 1
                size_vector_counts[idx] += 1
                size_vector_commits[idx].append(commit)

            file_commit_vector_size[file] = size_vector_counts
            file_commit_vector[file] = size_vector_commits

        for file, values in file_commit_vector_size.items():
            count = len(file_commits[file])
            if count == 0:
                divisor = 1
            else:
                divisor = count
            new_values = [val / divisor for val in values]
            file_commit_vector_size_norm[file] = new_values

        # 3. Build author vectors
        file_author_vector = {}
        file_author_count_vector = {}
        file_author_count_vector_norm = {}

        for f, size_vectors in file_commit_vector.items():
            author_sized_frequency = {}
            c = 0
            for commits in size_vectors:
                c = c + 1
                authors_in_size = set()
                for commit in commits:
                    if commit in file_commit_authors[f]:
                        for author in file_commit_authors[f][commit]:
                            authors_in_size.add(author)
                for author in authors_in_size:
                    author_sized_frequency[author] = author_sized_frequency.get(author, 0) + 1

            file_author_vector[f] = author_sized_frequency
            reciprocal_sums = []
            for commits in size_vectors:
                authors_in_size = set()
                for commit in commits:
                    if commit in file_commit_authors[f]:
                        for author in file_commit_authors[f][commit]:
                            authors_in_size.add(author)

                sum_reciprocals = sum(1 / author_sized_frequency[author] for author in authors_in_size)
                reciprocal_sums.append(sum_reciprocals)
            file_author_count_vector[f] = reciprocal_sums

        for file, values in file_author_count_vector.items():
            count = sum(file_author_count_vector[file])
            if count == 0:
                divisor = 1
            else:
                divisor = count
            new_values = [val / divisor for val in values]
            file_author_count_vector_norm[file] = new_values

        # 4. Build distinct author vectors
        file_commit_vector_total = {}
        file_commit_vector_size_total = {}
        max_commit_size_total = 0

        for commits in file_commits_total.values():
            for commit in commits:
                commit_size = len(commit_set_of_files_total[commit])
                max_commit_size_total = max(max_commit_size_total, commit_size)

        print("Maximum commit size across all files:", max_commit_size_total)

        for file, commits in file_commits_total.items():
            size_vector_counts = [0] * max_commit_size
            size_vector_commits = [[] for _ in range(max_commit_size)]

            for commit in commits:
                commit_size = len(commit_set_of_files_total[commit])
                idx = commit_size - 1
                size_vector_counts[idx] += 1
                size_vector_commits[idx].append(commit)

            file_commit_vector_size_total[file] = size_vector_counts
            file_commit_vector_total[file] = size_vector_commits

        file_distinct_author_vector = {}
        file_distinct_author_count_vector = {}
        file_distinct_author_count_vector_norm = {}

        for f, size_vectors in file_commit_vector.items():
            distinct_author_sized_frequency = {}
            c = 0
            for commits in size_vectors:
                c = c + 1
                distinct_authors_in_size = set()
                for commit in commits:
                    if commit in file_commit_authors_total[f]:
                        for author in file_commit_authors_total[f][commit]:
                            distinct_authors_in_size.add(author)
                for author in distinct_authors_in_size:
                    distinct_author_sized_frequency[author] = distinct_author_sized_frequency.get(author, 0) + 1

            file_distinct_author_vector[f] = distinct_author_sized_frequency
            distinct_reciprocal_sums = []
            for commits in size_vectors:
                distinct_authors_in_size = set()
                for commit in commits:
                    if commit in file_commit_authors_total[f]:
                        for author in file_commit_authors_total[f][commit]:
                            distinct_authors_in_size.add(author)

                distinct_sum_reciprocals = sum(1 / distinct_author_sized_frequency[author] for author in distinct_authors_in_size)
                distinct_reciprocal_sums.append(distinct_sum_reciprocals)
            file_distinct_author_count_vector[f] = distinct_reciprocal_sums

        for file, values in file_distinct_author_count_vector.items():
            count = sum(file_distinct_author_count_vector[file])
            if count == 0:
                divisor = 1
            else:
                divisor = count
            new_values = [val / divisor for val in values]
            file_distinct_author_count_vector_norm[file] = new_values

        # 5. Build lines add, del vectors
        file_lines_added_vector = {}
        file_lines_deleted_vector = {}

        for f, size_vectors in file_commit_vector.items():
            added_vector = []
            deleted_vector = []
            for commits in size_vectors:
                size_added = sum(file_commits_lines_added[f].get(c, 0) for c in commits)
                size_deleted = sum(file_commits_lines_deleted[f].get(c, 0) for c in commits)

                added_vector.append(size_added / file_lines_added[f] if file_lines_added[f] > 0 else 0.0)
                deleted_vector.append(size_deleted / file_lines_deleted[f] if file_lines_deleted[f] > 0 else 0.0)

            file_lines_added_vector[f] = added_vector
            file_lines_deleted_vector[f] = deleted_vector

        # 6. Build own vectors
        file_owner_vectors = {}
        file_oexp_vectors = {}
        file_exp_vectors = {}

        max_size = max(len(commit_set_of_files_total[c]) for c in commits_total)
        commit_size_vector = [[] for _ in range(max_size)]
        for commit in commits_total:
            commit_size = len(commit_set_of_files_total[commit])
            idx = commit_size - 1
            commit_size_vector[idx].append(commit)

        author_lines_total = dict()
        for a in author_lines_added_total:
            if a not in author_lines_total:
                author_lines_total[a] = 0
            author_lines_total[a] = author_lines_total[a] + author_lines_added_total[a] + author_lines_deleted_total[a]
        for f, size_vectors in file_commit_vector.items():
            authors_files = file_authors[f]
            max_lines = 0
            for a in authors_files:
                if max_lines < file_authors_lines[f][a]:
                    max_author = a
                    max_lines = file_authors_lines[f][a]

            owner = max_author 

            owner_vector = []
            oexp_vector = []
            exp_vector = []
            for commits in size_vectors:
                owner_lines = 0
                for commit in commits:
                    owner_lines = owner_lines + file_commit_author_lines[f][commit].get(owner, 0)
                if owner in file_authors_lines[f]:
                    if file_authors_lines[f][owner] > 0:
                        owner_vector.append(owner_lines / file_authors_lines[f][owner])
                    else:
                        owner_vector.append(0.0)
                else:
                    owner_vector.append(0.0)

            file_owner_vectors[f] = owner_vector

            total_lines = 0
            for commit in commits_total:
                if owner in commit_author_lines_total[commit]:
                    total_lines = total_lines + commit_author_lines_total[commit][owner]
            for commits in commit_size_vector:
                oexp_lines = 0
                for commit in commits:
                    if owner in commit_author_lines_total[commit]:
                        oexp_lines += commit_author_lines_total[commit][owner]
                if total_lines > 0:
                    oexp_vector.append(oexp_lines / total_lines)
                else:
                    oexp_vector.append(0.0)

            file_oexp_vectors[f] = oexp_vector

            for commits in size_vectors:
                total_lines = 0
                exp_lines = {}
                for commit in commits:
                    authors = file_commit_authors[f][commit]
                    for author in authors:
                        if author not in exp_lines:
                            exp_lines[author] = 0
                        exp_lines[author] += author_lines_total[author]
                mean_list = []
                if total_author_lines_total > 0:
                    for author in exp_lines:
                        mean_list.append(exp_lines[author] / total_author_lines_total)
                if mean_list:
                    exp_vector.append(statistics.geometric_mean(mean_list))
                else:
                    exp_vector.append(0.0)

            file_exp_vectors[f] = exp_vector

        # 6. Build minor vectors
        file_minor_vectors = {}
        file_minor_vectors_norm = {}

        for f, size_vectors in file_commit_vector.items():
            lines_file = file_lines_added[f] + file_lines_deleted[f]
            authors_list = set()
            authors_files = file_authors[f]
            for a in authors_files:
                if file_authors_lines[f][a] <= (lines_file * 5) / 100:
                    authors_list.add(a)

            minor_vector = []
            minor_author_size_frequency = {}
            for commits in size_vectors:
                minor_authors = set()
                for commit in commits:
                    for author in authors_list:
                        if author in file_commit_author_lines[f][commit]:
                            if file_commit_author_lines[f][commit][author] <= (lines_file * 5) / 100:
                                minor_authors.add(author)
                for author in minor_authors:
                    minor_author_size_frequency[author] = minor_author_size_frequency.get(author, 0) + 1

            reciprocal_sums = []
            for commits in size_vectors:
                minor_authors = set()
                for commit in commits:
                    for author in authors_list:
                        if author in file_commit_author_lines[f][commit]:
                            if file_commit_author_lines[f][commit][author] <= (lines_file * 5) / 100:
                                minor_authors.add(author)

                sum_reciprocals = sum(1 / minor_author_size_frequency[author] for author in minor_authors)
                reciprocal_sums.append(sum_reciprocals)

            file_minor_vectors[f] = reciprocal_sums

        for file, values in file_minor_vectors.items():
            count = sum(file_minor_vectors[file])
            if count == 0:
                divisor = 1
            else:
                divisor = count
            new_values = [val / divisor for val in values]
            file_minor_vectors_norm[file] = new_values

        # 7. Build SCTR vectors
        file_sctr_vectors = {}

        max_size = max(len(commit_set_of_files[c]) for c in commits_release)
        commit_size_vector_release = [[] for _ in range(max_size)]
        for commit in commits_release:
            commit_size = len(commit_set_of_files[commit])
            idx = commit_size - 1  # 0-based
            commit_size_vector_release[idx].append(commit)

        prob_file_vector = {}
        for f, size_vectors in file_commit_vector.items():
            prob_list = []
            all_commits = [c for commits in size_vectors for c in commits]
            for idx, commits in enumerate(size_vectors):
                if len(commit_size_vector_release[idx]) > 0:
                    prob_list.append(len(commits) / len(all_commits))
                else:
                    prob_list.append(0.0)
            prob_file_vector[f] = prob_list

        max_size = len(next(iter(prob_file_vector.values())))
        entropy_vector = [0.0] * max_size
        for k in range(max_size):
            entropy = 0.0
            for probs in prob_file_vector.values():
                p = probs[k]
                if p > 0:
                    entropy -= p * math.log(p, 2)
            entropy_vector[k] = entropy
        for f, prob_list in prob_file_vector.items():
            sctr_list = []
            for k, p in enumerate(prob_list):
                sctr_list.append(p * entropy_vector[k])
            file_sctr_vectors[f] = sctr_list

        # 8 Build NADEV vectors
        file_nadev_vectors = {}
        file_nddev_vectors = {}
        file_ncomm_vectors = {}
        file_nsctr_vectors = {}

        for f in file_author_count_vector:
            vec_len = len(file_author_count_vector[f])
            nadev_vector = np.zeros(vec_len)

            if f in self.weighted_graph:
                for neighbor in self.weighted_graph.neighbors(f):
                    if neighbor in file_author_count_vector:
                        weight = self.weighted_graph[f][neighbor]['weight']
                        neighbor_vec = np.array(file_author_count_vector[neighbor])
                        nadev_vector += weight * neighbor_vec

            file_nadev_vectors[f] = nadev_vector.tolist()

        for f in file_distinct_author_count_vector:
            vec_len = len(file_distinct_author_count_vector[f])
            nddev_vector = np.zeros(vec_len)

            if f in self.weighted_graph:
                for neighbor in self.weighted_graph.neighbors(f):
                    if neighbor in file_distinct_author_count_vector:
                        weight = self.weighted_graph[f][neighbor]['weight']
                        neighbor_vec = np.array(file_distinct_author_count_vector[neighbor])
                        nddev_vector += weight * neighbor_vec

            file_nddev_vectors[f] = nddev_vector.tolist()

        for f in file_commit_vector_size:
            vec_len = len(file_commit_vector_size[f])
            ncomm_vector = np.zeros(vec_len)

            if f in self.weighted_graph:
                for neighbor in self.weighted_graph.neighbors(f):
                    if neighbor in file_commit_vector_size:
                        weight = self.weighted_graph[f][neighbor]['weight']
                        neighbor_vec = np.array(file_commit_vector_size[neighbor])
                        ncomm_vector += weight * neighbor_vec

            file_ncomm_vectors[f] = ncomm_vector.tolist()

        for f in file_sctr_vectors:
            vec_len = len(file_sctr_vectors[f])
            nsctr_vector = np.zeros(vec_len)

            if f in self.weighted_graph:
                for neighbor in self.weighted_graph.neighbors(f):
                    if neighbor in file_sctr_vectors:
                        weight = self.weighted_graph[f][neighbor]['weight'] 
                        neighbor_vec = np.array(file_sctr_vectors[neighbor])
                        nsctr_vector += weight * neighbor_vec

            file_nsctr_vectors[f] = nsctr_vector.tolist()

        for f in file_ncomm_vectors:
            if f in self.weighted_graph:
                neighbors = list(self.weighted_graph.neighbors(f))
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
                    ncomm_count = commn
                    nadev_count = adevn
                    nddev_count = ddevn
                    nsctr_count = sctrn
                else:
                    ncomm_count = 0
                    nadev_count = 0
                    nddev_count = 0
                    nsctr_count = 0

                if ncomm_count == 0:
                    ncomm_divisor = 1 
                else:
                    ncomm_divisor = ncomm_count
                if nadev_count == 0:
                    nadev_divisor = 1  
                else:
                    nadev_divisor = nadev_count
                if nddev_count == 0:
                    nddev_divisor = 1  
                else:
                    nddev_divisor = nddev_count
                if nsctr_count == 0:
                    nsctr_divisor = 1  
                else:
                    nsctr_divisor = nsctr_count

                new_ncomm_values = [val / ncomm_divisor for val in file_ncomm_vectors[f]]
                new_nadev_values = [val / nadev_divisor for val in file_nadev_vectors[f]]
                new_nddev_values = [val / nddev_divisor for val in file_nddev_vectors[f]]
                new_nsctr_values = [val / nsctr_divisor for val in file_nsctr_vectors[f]]
                file_ncomm_vectors[f] = new_ncomm_values
                file_nadev_vectors[f] = new_nadev_values
                file_nddev_vectors[f] = new_nddev_values
                file_nsctr_vectors[f] = new_nsctr_values

        print("\n=== File → Commit Size Vectors (first 5) ===")
        for f, v in list(file_commit_vector_size_norm.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → Author Count Vectors (first 5) ===")
        for f, v in list(file_author_count_vector_norm.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → Distinct Author Count Vectors (first 5) ===")
        for f, v in list(file_distinct_author_count_vector_norm.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → ADD Vectors (first 5) ===")
        for f, v in list(file_lines_added_vector.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → DEL Vectors (first 5) ===")
        for f, v in list(file_lines_deleted_vector.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → OWN Vectors (first 5) ===")
        for f, v in list(file_owner_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → OEXP Vectors (first 5) ===")
        for f, v in list(file_oexp_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → EXP Vectors (first 5) ===")
        for f, v in list(file_exp_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → MINOR Vectors (first 5) ===")
        for f, v in list(file_minor_vectors_norm.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → SCTR Vectors (first 5) ===")
        for f, v in list(file_sctr_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → NADEV Vectors (first 5) ===")
        for f, v in list(file_nadev_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → NDDEV Vectors (first 5) ===")
        for f, v in list(file_nddev_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → NCOMM Vectors (first 5) ===")
        for f, v in list(file_ncomm_vectors.items())[:5]:
            print(f"{f}: {v}")

        print("\n=== File → NSCTR Vectors (first 5) ===")
        for f, v in list(file_nsctr_vectors.items())[:5]:
            print(f"{f}: {v}")

        self.process_metrics['comm'][t1] = file_commit_vector_size_norm
        self.process_metrics['adev'][t1] = file_author_count_vector_norm
        self.process_metrics['ddev'][t1] = file_distinct_author_count_vector_norm
        self.process_metrics['sctr'][t1] = file_sctr_vectors
        self.process_metrics['ncomm'][t1] = file_ncomm_vectors
        self.process_metrics['nadev'][t1] = file_nadev_vectors
        self.process_metrics['nddev'][t1] = file_nddev_vectors
        self.process_metrics['nsctr'][t1] = file_nsctr_vectors
        self.process_metrics['add'][t1] = file_lines_added_vector
        self.process_metrics['del'][t1] = file_lines_deleted_vector
        self.process_metrics['own'][t1] = file_owner_vectors
        self.process_metrics['minor'][t1] = file_minor_vectors_norm
        self.process_metrics['oexp'][t1] = file_oexp_vectors
        self.process_metrics['exp'][t1] = file_exp_vectors


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
        with open(f'{project}_process_metrics_centralities_git_data_vector_norm_fatty.pkl', 'wb') as f:
            pickle.dump(t.process_metrics, f)
        for pr in t.process_metrics:
            for pr1 in t.process_metrics[pr]:
                print(pr1, len(t.process_metrics[pr][pr1]))

    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
