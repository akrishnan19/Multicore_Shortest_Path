#!/usr/bin/python

import sys
from random import randint

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
