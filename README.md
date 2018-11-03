# Multicore_Shortest_Path
Instructions for use:
	1) run make
	2) ./dijkstra.out path_to_file source_vertex

Structure of input file:
	1) Line 1 contains number of vertices
	2) Incidence matrix with the following properties:
		a) All new rows are in a new line
		b) There is a space after each value including the last value of the row
		c) An empty line at the end of the file to signify EOF
		
		e.g:
		0 4 0 0 0 0 0 8 0 
		4 0 8 0 0 0 0 11 0 
		.
		.
		.
		(so on and so forth) 
		