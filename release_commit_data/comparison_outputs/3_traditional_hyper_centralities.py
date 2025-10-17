import itertools
import math
import os
import pickle
import statistics
from collections import defaultdict
import networkx as nx
import datetime
from datetime import datetime


class Task:

    def __init__(self):
        self.graph = nx.Graph()
        self.hyper_graph = dict()
        self.sorted_centrality = dict()
        self.sorted_betweeness = dict()
        self.sorted_closeness = dict()
        self.sorted_degree = dict()
        self.D = dict()

    def load_pickles(self, project, t1):
        self.graph.clear()
        self.hyper_graph.clear()
        pickle_dir = os.path.join(f"{project}_{t1}_pickles")

        file_bugs = pickle.load(open(os.path.join(pickle_dir, "file_bugs.pkl"), "rb"))
        commits = pickle.load(open(os.path.join(pickle_dir, "commits.pkl"), "rb"))
        commit_set_of_files = pickle.load(open(os.path.join(pickle_dir, "commit_set_of_files.pkl"), "rb"))

        edge_weights = defaultdict(int)
        for commit in commits:
            files = commit_set_of_files[commit]
            ordered_pairs = {(x, y) for x in files for y in files if x != y}
            self.graph.add_edges_from(ordered_pairs)
            for x, y in ordered_pairs:
                pair = tuple(sorted((x, y)))
                edge_weights[pair] += 1

        for (x, y), weight in edge_weights.items():
            self.graph.add_edge(x, y, weight=weight)

        for u, v, data in self.graph.edges(data=True):
            data['distance'] = 1 / data['weight']

        with open(f'{project}_{t1}_graph_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(self.graph, f)

        sorted_centrality = {}
        sorted_betweeness = {}
        sorted_closeness = {}
        sorted_degree = {}
        print(f"eigenvector building")
        if self.graph:
            centrality = nx.eigenvector_centrality(self.graph, max_iter=10000, weight='weight')
            sorted_centrality = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True))
            self.sorted_centrality[t1] = sorted_centrality
        print(f"betweeness building")
        if self.graph:
            betweeness = nx.betweenness_centrality(self.graph, weight='weight')
            sorted_betweeness = dict(sorted(betweeness.items(), key=lambda x: x[1], reverse=True))
            self.sorted_betweeness[t1] = sorted_betweeness
        print(f"closeness building")
        if self.graph:
            closeness = nx.closeness_centrality(self.graph, distance='distance')
            sorted_closeness = dict(sorted(closeness.items(), key=lambda x: x[1], reverse=True))
            self.sorted_closeness[t1] = sorted_closeness
        print(f"degree building")

        def degree_cent(G):
            s = 1.0 / (len(G) - 1.0)
            centrality = {n: d * s for n, d in G.degree(weight='weight')}
            return centrality

        if self.graph:
            degree = degree_cent(self.graph)
            sorted_degree = dict(sorted(degree.items(), key=lambda x: x[1], reverse=True))
            self.sorted_degree[t1] = sorted_degree

        print(f"dumping {datetime.now()}")
        with open(f'{project}_{t1}_sorted_eigenvector_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(sorted_centrality, f)
        with open(f'{project}_{t1}_sorted_betweeness_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(sorted_betweeness, f)
        with open(f'{project}_{t1}_sorted_closeness_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(sorted_closeness, f)
        with open(f'{project}_{t1}_sorted_degree_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(sorted_degree, f)

        for commit in commits:
            files = commit_set_of_files[commit]
            if commit not in self.hyper_graph.keys():
                self.hyper_graph[commit] = files

        node_set = set()
        for edge in self.hyper_graph:
            nodes = self.hyper_graph[edge]
            for n in nodes:
                node_set.add(n)

        with open(f'{project}_{t1}_hyper_graph_git_data_fatty.pkl', 'wb') as f:
            pickle.dump(self.hyper_graph, f)

        line_graph = nx.Graph()
        for edge in self.hyper_graph:
            nodes = self.hyper_graph[edge]
            line_graph.add_node(tuple(nodes))

        for node1 in line_graph.nodes():
            for node2 in line_graph.nodes():
                if node1 != node2:
                    if set(node1) & set(node2): 
                        line_graph.add_edge(node1, node2)

        D = 0
        for node in line_graph.nodes:
            if len(node) > D:
                D = len(node)
        print(f"D = {D}")
        self.D[t1] = D

        if D > 1:
            eigenvector_centrality = nx.eigenvector_centrality(line_graph, max_iter=1000)
            betweeness_centrality = nx.betweenness_centrality(line_graph)
            closness_centrality = nx.closeness_centrality(line_graph)
            degree_centrality = nx.degree_centrality(line_graph)

            print(f"calculating hyper vector centrality {datetime.now()}")
            hyper_vector_centrality = dict()
            hyper_betweeness = dict()
            hyper_closeness = dict()
            hyper_degree = dict()
            for i in node_set:
                ev = 0
                bn = 0
                cn = 0
                de = 0
                if i not in hyper_vector_centrality.keys():
                    hyper_vector_centrality[i] = list()
                if i not in hyper_betweeness.keys():
                    hyper_betweeness[i] = list()
                if i not in hyper_closeness.keys():
                    hyper_closeness[i] = list()
                if i not in hyper_degree.keys():
                    hyper_degree[i] = list()
                for k in range(D):
                    if k >= 2:
                        for node in line_graph.nodes():
                            if k == line_graph.degree(node):
                                if i in node:
                                    ev = ev + eigenvector_centrality[node]
                                    bn = bn + betweeness_centrality[node]
                                    cn = cn + closness_centrality[node]
                                    de = de + degree_centrality[node]
                        hyper_vector_centrality[i].append(ev / k)
                        hyper_betweeness[i].append(bn / k)
                        hyper_closeness[i].append(cn / k)
                        hyper_degree[i].append(de / k)

            print(f"it's done {datetime.now()}")

            vector_centrality = hyper_vector_centrality.copy()
            vector_centrality_betweeness = hyper_betweeness.copy()
            vector_centrality_closeness = hyper_closeness.copy()
            vector_centrality_degree = hyper_degree.copy()
            for i in hyper_vector_centrality:
                if i not in self.sorted_centrality[t1].keys():
                    vector_centrality.pop(i)
                if i not in self.sorted_betweeness[t1].keys():
                    vector_centrality_betweeness.pop(i)
                if i not in self.sorted_closeness[t1].keys():
                    vector_centrality_closeness.pop(i)
                if i not in self.sorted_degree[t1].keys():
                    vector_centrality_degree.pop(i)

            print(f"dumping {datetime.now()}")
            with open(f'{project}_{t1}_vector_centrality_git_data_fatty.pkl', 'wb') as f:
                pickle.dump(vector_centrality, f)
            with open(f'{project}_{t1}_vector_centrality_betweeness_git_data_fatty.pkl', 'wb') as f:
                pickle.dump(vector_centrality_betweeness, f)
            with open(f'{project}_{t1}_vector_centrality_closeness_git_data_fatty.pkl', 'wb') as f:
                pickle.dump(vector_centrality_closeness, f)
            with open(f'{project}_{t1}_vector_centrality_degree_git_data_fatty.pkl', 'wb') as f:
                pickle.dump(vector_centrality_degree, f)


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
