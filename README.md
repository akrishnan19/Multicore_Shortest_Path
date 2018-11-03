# Multicore_Shortest_Path
Instructions for use:
	1) run make
	2) ./dijkstra path_to_file source_vertex

Structure of input file:
	1) Line 1 contains number of vertices
	2) Incidence matrix, with a new row in a new line and values in a row separated by a space and ","
		e.g:
		0, 4, 0, 0, 0, 0, 0, 8, 0
		4, 0, 8, 0, 0, 0, 0, 11, 0
		.
		.
		.
		(so on and so forth) 