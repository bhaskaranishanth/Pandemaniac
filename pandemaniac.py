import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import sim
import numpy as np

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


    triangle_info = nx.triangles(G)
    


    # Strategy
    strategies = {'s1' : [0, 1], 's2' : [5, 6]}
    output = sim.run(nx.to_dict_of_dicts(G), strategies)
    print output
    return G

G = run('testgraph1.json')
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

