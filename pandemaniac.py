import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import sim

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

    triangle_info = nx.traingles(G)
    


    # Strategy
    strategies = {'s1' : [0, 1], 's2' : [5, 6]}
    output = sim.run(nx.to_dict_of_dicts(G), strategies)
    print output



run('testgraph1.json')
