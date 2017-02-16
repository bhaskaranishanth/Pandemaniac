import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import sim
import numpy as np
import collections

def run(file_name):
    f = open(file_name, 'r')
    data = json.loads(f.read())

    graph_data = {}
    for k, v in data.iteritems():
        graph_data[int(k)] = [int(x) for x in v]

    # pprint(graph_data)

    G = nx.Graph()
    for k, v in graph_data.iteritems():
        for n in v:
            G.add_edge(k, n)


    # pprint(graph_data)
    # print G.nodes()
    # nx.draw(G)
    # plt.show()

    # make_histogram(G)
    # make_ccdf_clustering(G)

    triangle_info = nx.triangles(G)

    # Strategy
    win = 0
    total = 0
    for num_nodes in range(5, 30):
        print 'Num nodes; ', num_nodes
        total += 1
        # s1_nodes = get_highest_degree_nodes(num_nodes, G)

        s1_nodes = strategy_3(num_nodes, G)
        s2_nodes = strategy_2(num_nodes, G)
        strategies = {'s1' : s1_nodes, 's2' : s2_nodes}
        output = sim.run(graph_data, strategies)

        if output['s2'] > output['s1']:
            win += 1
        print output
        print 'Win: %d  Total: %d Perc: %f' % (win, total, float(win) / total)
    # Make the degree histogram for G
    return G


def make_histogram(G):
    # Histogram
    degree_sequence=sorted([d for n, d in G.degree().iteritems()], reverse=True)
    degreeCount=collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")

    plt.show()


def make_ccdf_clustering(G):
    dict_c_coeff = nx.clustering(G)
    coeff_seq = sorted(dict_c_coeff.values(),reverse = True)
    hist, bins = np.histogram(coeff_seq, bins = 50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Histogram Of Node Clustering Coefficients')
    plt.ylabel("Count")
    plt.xlabel("Clustering Coefficients")
    plt.show()
    cdf = np.cumsum(hist)
    yvals = np.arange(len(coeff_seq))/float(len(coeff_seq))
    yvals = yvals
    plt.plot(coeff_seq, yvals)
    plt.title('CCDF Of Node Clustering Coefficients')
    plt.show()
    
def get_highest_degree_nodes(n, G):
    D = nx.degree(G)
    x = sorted(D, key = D.get, reverse = True)[:n]
    return x


''' Methods attempted: Degree Centrality, Closeness centrality,
Choosing top n nodes with least degree, betweenness centrality
'''
def strategy_2(n, G):
    triangle_info = nx.triangles(G)
    node_order = sorted(triangle_info, key=lambda x:triangle_info[x], reverse=True)
    return node_order[:n]

def strategy_3(n, G):
    # Perform betweenness principle
    D = nx.closeness_centrality(G)
    x = sorted(D, key = D.get, reverse=True)[:n]
    return x


run('testgraph1.json')
