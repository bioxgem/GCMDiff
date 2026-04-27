import numpy as np
#有問題!!
def normalize_graph_0_1(graph):
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            beginAtom=(graph[i][j][0]-1)/21
            endAtom=(graph[i][j][1]-1)/21
            bond=(graph[i][j][2])/3
            graph[i][j]=[beginAtom,endAtom,bond]
    return graph

def normalize_graph_m1_1(graph):
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            beginAtom=(2*(graph[i][j][0]-1)/21)-1
            endAtom=(2*(graph[i][j][1]-1)/21)-1
            bond=(graph[i][j][2])/3
            graph[i][j]=[beginAtom,endAtom,bond]
    return graph