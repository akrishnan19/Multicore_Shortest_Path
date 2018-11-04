// A C++ program for Bellman Ford's single source shortest path algorithm. 
// The program is for adjacency matrix representation of the graph 

#include <stdio.h> 
#include <limits.h> 
#include <omp.h>

void minDistance(int *dist, int *used, int *min_index) {
    int min = INT_MAX; 
   
   for (int v = 0; v < V; v++)
     if (used[v] == false && dist[v] <= min) 
         min = dist[v], *min_index = v;
}

void ford(int** incidence_matrix, int n, int src) {
    // Initialize nxn matrix for distance from each node to each other node
    // Dist to itself is 0, else is INT_MAX
    int* dist = (int*) malloc(sizeof(int)*n);
    dist[0] = 0;
    for(int i = 1; i < n; i++) {
        dist[i] = INT_MAX;
    }

    // Find distance for each pair of vertecies
    for(int i = 0; i < n; i ++) {
        
    }
}

int** read_file(char *file_name, int *vertices) {
  FILE *f;
  int **incidence_matrix;
  int v;

  f = fopen(file_name, "r");
  if(f == NULL) {
    printf("Error reading file\n");
    exit(1);
  }

  fscanf(f, "%d\n", vertices);
  
  v = *vertices; // this is meant for readability later, not any optimizations
  incidence_matrix = (int**) malloc(sizeof(int*) * v);
  #pragma omp parallel for
  for(int iii = 0; iii < v; iii++) incidence_matrix[iii] = (int*)malloc(v * sizeof(int));

  for(int iii = 0; iii < v; iii++) {
    for(int jjj = 0; jjj < v; jjj++) {
      fscanf(f, "%d", &incidence_matrix[iii][jjj]);
    }
  }

  return incidence_matrix;
}

int main(int argc, char *argv[]) {
    int **incidence_matrix;
    int vertices;

    if(argc != 3) {
        printf("Incorrect usage\n"); // sanity check
        printf("Correct usage: ./dijktra path_to_file source_vertex");
        exit(1);
    }
    
    incidence_matrix = read_file(argv[1], &vertices);

    ford(incidence_matrix, vertices, argv[2]);

    free(incidence_matrix);

    return 0;
}