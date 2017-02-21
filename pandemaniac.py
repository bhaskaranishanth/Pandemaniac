import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import sim
import numpy as np
import collections
import random
import itertools

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

    # # Output the two strategies
    # output_file(10, G, file_name)

    # return



    # pprint(graph_data)
    # print G.nodes()
    # nx.draw(G)
    # plt.show()

    # make_histogram(G)
    # make_ccdf_clustering(G)

    # Strategy
    win = 0
    total = 0
    for num_nodes in range(10, 11):
        print 'Num nodes; ', num_nodes
        total += 1
        s1_nodes = get_highest_degree_nodes(num_nodes, G)


        # s1_nodes = strategy_3(num_nodes, G)
        # s1_nodes = strategy_4(num_nodes, G)
        # s2_nodes = strategy_4(num_nodes, G)
        # s2_nodes = strategy_5(num_nodes, G) # Best strategy so far
        # s2_nodes = strategy_6(num_nodes, G)
        # s2_nodes = strategy_7(num_nodes, G)
        # s2_nodes = strategy_8(num_nodes, G)
        # s2_nodes = strategy_2(num_nodes, G)
        s2_nodes = strategy_9(num_nodes, G)


        # print 's1 nodes: ', s1_nodes
        print 's2 nodes: ', s2_nodes
        # assert len(s2_nodes) == num_nodes
        # assert not set(s2_nodes).issubset(s1_nodes)
        # printIntersection(s1_nodes, s2_nodes)
        # s1_nodes = [57, 54, 53, 18]
        # s2_nodes = [60, 80]
        # strategies = {'s1' : s1_nodes, 's2' : s2_nodes}
        # output = sim.run(graph_data, strategies)
        # print output

        # if output['s2'] > output['s1']:
        #     win += 1
        # print 'Win: %d  Total: %d Perc: %f' % (win, total, float(win) / total)
    # Make the degree histogram for G
    return G


def printIntersection(s1_nodes, s2_nodes):
    print "Getting intersection "
    s2_nodes_diff = [val for val in s2_nodes if val not in s1_nodes]
    s1_nodes_diff = [val for val in s1_nodes if val not in s2_nodes]
    print "S1 Nodes: ", s1_nodes_diff
    print "S2 Nodes: ", s2_nodes_diff


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


def brute_force(s1_nodes, other_nodes, graph_data):
    invalid = [11, 108, 75, 2, 17, 111, 115, 113, 57, 54, 53, 18]
    valid = [val for val in other_nodes if val not in invalid]
    print "valid length: ", len(valid)

    attempt = list(itertools.combinations(valid, 2))
    print len(attempt)
    s1_nodes = [75, 2, 17, 111, 115, 113, 57, 54, 53, 18]


    for elem in attempt:
        s2_nodes = elem
        strategies = {'s1' : s1_nodes, 's2' : s2_nodes}
        output = sim.run(graph_data, strategies)
        print output, "\t", elem
        if output['s2'] > output['s1']:
            print "We did it"
            break


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

def strategy_4(n, G):
    triangle_info = nx.triangles(G)
    node_order = sorted(triangle_info, key=lambda x:triangle_info[x], reverse=True)

    val = []
    for x in node_order:
        val.append(x)
        neighbors = G[x]
        neigh_order = sorted(neighbors, key=lambda x:triangle_info[x], reverse=True)

        # val.extend(G[x])
        val.append(neigh_order[0])
        # val.append(neigh_order[-1])

    # print val
    # D = nx.closeness_centrality(G)
    # x = sorted(D, key = D.get, reverse=True)

    ret_set = set()
    counter = 0
    while len(ret_set) != n and counter < len(val):
        ret_set.add(val[counter])
        counter += 1

    # print ret_set
    assert len(ret_set) == n
    return ret_set
    # return node_order[:n]



def strategy_5(n, G):
    data_info = nx.number_of_cliques(G)
    node_order = sorted(data_info, key=lambda x:data_info[x], reverse=True)
    return node_order[:n]


def strategy_6(n, G):
    # data_info = nx.strongly_connected_components_recursive(nx.DiGraph(G))
    # Get largest connected component
    node_order = max(nx.strongly_connected_components_recursive(nx.DiGraph(G)), key=len)
    # sorted(data_info, key=lambda x:data_info[x], reverse=True)
    # print node_order

    return node_order[:n]

def strategy_7(n, G):
    data_info = nx.pagerank(G)
    node_order = sorted(data_info, key=lambda x:data_info[x], reverse=True)

    return node_order[:n]

def strategy_8(n, G):
    # Perform betweenness principle
    D = nx.betweenness_centrality(G)
    x = sorted(D, key = D.get, reverse=True)[:n]
    return x


def strategy_9(n, G):
    # Get top 120% nodes
    D = nx.degree(G)
    x = sorted(D, key = D.get, reverse = True)
    top_strat = x[:n + (n / 5)]
    print top_strat


    # Look at the top nodes neighbors and see the one list
    # where there are not many of the top neighbors
    # for elem in top_strat:
    #     print "elem: ", elem, " and ", G.neighbors(elem)
    #     [val for val in G.neighbors(elem) if val not in invalid]
    

    # Look at nodes whose neighbors are not in this list
    # print nx.degree_histogram(G)
    # # Start with betweenness
    # betD = nx.betweenness_centrality(G)
    # betX = sorted(D, key = D.get, reverse=True)
    # for elem in betX:

    data_info = nx.number_of_cliques(G)
    node_order = sorted(data_info, key=lambda x:data_info[x], reverse=True)
    new_list = [val for val in node_order if val not in top_strat]
    return new_list[:n]


    # intersect = G.neighbors(x[0])
    # for i in range(1, 8):
    #     neighbors2 = G.neighbors(x[i])
    #     # print "neighbors1: ", intersect
    #     print "neighbors2: ", neighbors2
    #     intersect = [val for val in intersect if val in neighbors2 ]
    #     # and val not in top_strat]
    #     print "Intersect: ", intersect


def output_file(n, G, file_name):
    n1 = strategy_2(n, G)
    n2 = strategy_3(n, G)
    ret_lst = []
    for i in range(50):
        if i % 2:
            ret_lst.extend(n1)
        else:
            ret_lst.extend(n2)
    f = open(file_name[:-5] + '_output.txt', 'w')
    assert len(ret_lst) == n * 50
    for i in ret_lst:
        f.write(str(i) + '\n')
    f.close()



run('2.10.31.json')

# f = open('8.35.2.json')
# data = json.loads(f.read())
# f.close()
# graph_data = {}
# for k, v in data.iteritems():
#     graph_data[int(k)] = [int(x) for x in v]

# g = open('dup.txt', 'w')
# for y in range(35):
#     for i in range(50):
#         x = random.choice(graph_data.keys())
#         g.write(str(x) + '\n')

# g.close()

