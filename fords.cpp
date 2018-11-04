// A C++ program for Bellman Ford's single source shortest path algorithm. 
// The program is for adjacency matrix representation of the graph 

#include <stdio.h> 
#include <limits.h> 
#include <omp.h>


void ford(int graph[V][V], int n) {
    vector <int> v [2000 + 10];
    int dis [1000 + 10];

    for(int i = 0; i < m + 2; i++){

        v[i].clear();
        dis[i] = 2e9;
    }

   for(int i = 0; i < m; i++){

        scanf("%d%d%d", &from , &next , &weight);

        v[i].push_back(from);
        v[i].push_back(next);
        v[i].push_back(weight);
   }

    dis[0] = 0;
    for(int i = 0; i < n - 1; i++){
        int j = 0;
        while(v[j].size() != 0){

            if(dis[ v[j][0]  ] + v[j][2] < dis[ v[j][1] ] ){
                dis[ v[j][1] ] = dis[ v[j][0]  ] + v[j][2];
            }
            j++;
        }
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

    dijktra(incidence_matrix, vertices, atoi(argv[2]));

    free(incidence_matrix);

    return 0;
}