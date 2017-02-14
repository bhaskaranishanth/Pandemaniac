import json
from pprint import pprint
import networkx as nx

def run(file_name):
    f = open(file_name, 'r')
    data = json.loads(f.read())

    graph_data = {}
    for k, v in data.iteritems():
        graph_data[int(k)] = [int(x) for x in v]

    pprint(graph_data)

    # G = nx.Graph()
    # for k, v in graph_data.iteritems():
    #     for n in v:
    #         G.add_edge(k, v)


    # pprint(graph_data)



run('testgraph1.json')
