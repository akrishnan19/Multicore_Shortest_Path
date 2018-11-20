#!/usr/bin/python

import sys
from random import randint

class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                      for row in range(vertices)] 
  
    def printSolution(self, dist, src): 
        print "Vertex \tDistance from Vertex " + str(src)
        for node in range(self.V): 
            print node,"\t\t",dist[node]
        print "\n"
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minDistance(self, dist, sptSet): 
  
        # Initilaize minimum distance for next node 
        min = sys.maxint 
  
        # Search not nearest vertex not in the  
        # shortest path tree 
        for v in range(self.V): 
            if dist[v] < min and sptSet[v] == False: 
                min = dist[v] 
                min_index = v 
  
        return min_index 
  
    # Funtion that implements Dijkstra's single source  
    # shortest path algorithm for a graph represented  
    # using adjacency matrix representation 
    def dijkstra(self, src): 
  
        dist = [sys.maxint] * self.V 
        dist[src] = 0
        sptSet = [False] * self.V 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minDistance(dist, sptSet) 
  
            # Put the minimum distance vertex in the  
            # shotest path tree 
            sptSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]: 
                    dist[v] = dist[u] + self.graph[u][v] 
  
        self.printSolution(dist, src)

if(len(sys.argv) != 3):
	print "Need number of nodes and maximum weight"
	print "Sample usage: ./test_gen.py 10 10"
	exit()

G = [[-1 for x in range(int(sys.argv[1]))] for y in range(int(sys.argv[1]))]

for i in range(int(sys.argv[1])):
	for j in range(int(sys.argv[1])):
		if(G[j][i] != -1):
			G[i][j] = G[j][i]
		else:
			if(i == j):
				G[i][j] = 0
			else:
				G[i][j] = randint(0, int(sys.argv[2]))
f = open("gen_test.txt", "w+")
f.write(str(int(sys.argv[1])) + "\n")
for i in range(int(sys.argv[1])):
	f.write(str(G[i][0]))
	for j in range(1, int(sys.argv[1])):
		f.write(" " + str(G[i][j]))
	f.write(" \n");

graph = Graph(int(sys.argv[1]))
graph.graph = G
sys.stdout = open("golden_standard.txt", "w+")
for i in range(int(sys.argv[1])):
	graph.dijkstra(i)
