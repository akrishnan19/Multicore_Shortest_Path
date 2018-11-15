# Multicore_Shortest_Path
### Instructions for use:
1. run make [dijkstra] [fords]
2. ./dijkstra path_to_file source_vertex or ./dijkstra path_to_file
3. ./fords path_to_file source_vertex or ./fords path_to_file

### Structure of input file:
1. Line 1 contains number of vertices

2. Incidence matrix with the following properties:
    1. All new rows are in a new line
    2. There is a space after each value including the last value of the row
    3. An empty line at the end of the file to signify EOF
        
        ```e.g:
        9
        0 4 0 0 0 0 0 8 0 
        4 0 8 0 0 0 0 11 0 
        .
        .
        .
        (so on and so forth) 
        
        ```

3. Algorithms Implemented:
    1. Dijkstra's Algorithm
        1. By Akaash
        2. TODO: time/work complexity
    2. Bellman Ford's Algorithm
        1. By Beathan
        2. TODO: time/work complexity