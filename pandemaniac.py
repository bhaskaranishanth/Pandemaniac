import json
from pprint import pprint

def run(file_name):
    f = open(file_name, 'r')
    graph_data = json.loads(f.read())


    pprint(graph_data)



run('testgraph1.json')
