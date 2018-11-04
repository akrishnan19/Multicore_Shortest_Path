// A C++ program for Bellman Ford's single source shortest path algorithm. 
// The program is for adjacency matrix representation of the graph 

#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <limits.h> 
#include <malloc.h>
#include <omp.h>

#define DEBUG 1

using namespace std;

void ford(int** incidence_matrix, int n, int src) {
    int* dist = (int*) malloc(sizeof(int)*n);
    for(int v = 0; v < n; v++) {
        dist[v] = INT_MAX;
    }
    dist[src] = 0;

    for(int v = 0; v < n; v++) {
        for (int e = 0; e < n; e++) { 
            if(incidence_matrix[v][e]) {
                int tempDistance = dist[v] + incidence_matrix[v][e];
                if(tempDistance < dist[e]) {
                    dist[e] = tempDistance;
                }
            }
        }
    }

    for(int i = 0; i < n; i++) {
        cout << dist[i] << endl;
    }

    free(dist);
}

/* Sudo code used
function bellmanFord(G, S)
  for each vertex V in G
      distance[V] <- infinite
      previous[V] <- NULL
  distance[S] <- 0
  for each vertex V in G        
    for each edge (U,V) in G
      tempDistance <- distance[U] + edge_weight(U, V)
      if tempDistance < distance[V]
         distance[V] <- tempDistance
         previous[V] <- U

  for each edge (U,V) in G
    If distance[U] + edge_weight(U, V) < distance[V}
      Error: Negative Cycle Exists

  return distance[], previous[]
  */

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
        printf("Correct usage: ./ford path_to_file source_vertex");
        exit(1);
    }
    
    incidence_matrix = read_file(argv[1], &vertices);

    ford(incidence_matrix, vertices, atoi(argv[2]));

    free(incidence_matrix);

    return 0;
}