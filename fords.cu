#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <limits.h> 
#include <malloc.h>
#include <omp.h>

__global__ void fords_kernel(int n, int u, int *mat, int *dist) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if(index < n) {
    for(int v = index; v < n; v += stride) {
      int weight = mat[u * n + v];
      if(weight) {
        int temp = dist[u] + weight;
        if(temp < dist[v]) {
          dist[v] = temp;
        }
      }
    }
  }
}

void printResults(int dist[], int n, int source) { 
  printf("Vertex\t\tDistance from Source Vertex %d\n", source);
  for (int i = 0; i < n; i++)
    printf("%d\t\t%d\n", i, dist[i]);
  printf("\n\n");
} 

void fords(int *incidence_matrix, int n, int src) {
  int *h_dist;
  int *d_mat, *d_dist;

  h_dist = (int*) malloc(sizeof(int) * n);
  cudaMalloc((void**) &d_mat, sizeof(int) * n *n);
  cudaMalloc((void**) &d_dist, sizeof(int) * n);

  #pragma omp parallel for
  for(int i = 0 ; i < n; i ++){
    h_dist[i] = INT_MAX;
  }

  h_dist[src] = 0;

  cudaMemcpy(d_mat, incidence_matrix, sizeof(int) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dist, h_dist, sizeof(int) * n, cudaMemcpyHostToDevice);

  dim3 THREAD_SIZE = 256; // can't use variable size so everything is hard-coded
  dim3 BLOCK_SIZE = 256;

  for(int iii = 0; iii < n; iii++) {
    fords_kernel<<< BLOCK_SIZE, THREAD_SIZE >>>(n, iii, d_mat, d_dist);
  }
  
  cudaMemcpy(h_dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);

  printResults(h_dist, n, src);

  cudaFree(d_mat);
  cudaFree(d_dist);
}

int* read_file(char *file_name, int *vertices) {
  FILE *f;
  int *incidence_matrix;
  int v;

  f = fopen(file_name, "r");
  if(f == NULL) {
    printf("Error reading file\n");
    exit(1);
  }

  fscanf(f, "%d\n", vertices);
  
  v = *vertices; // this is meant for readability later, not any optimizations
  incidence_matrix = (int*) malloc(sizeof(int) * v * v);

  for(int iii = 0; iii < v; iii++) {
    for(int jjj = 0; jjj < v; jjj++) {
      fscanf(f, "%d", &incidence_matrix[iii * v + jjj]);
      if(incidence_matrix[iii* v + jjj] < 0) {
        printf("Error: negative edge weight found at %d, %d\n", iii, jjj);
        exit(-1);
      }
    }
  }

  return incidence_matrix;
}

int main(int argc, char *argv[]) {
    int *incidence_matrix;
    int vertices;

    if(argc != 2) {
        printf("Incorrect usage\n"); // sanity check
        printf("Correct usage: ./ford path_to_file\n");
        exit(1);
    }
    
    incidence_matrix = read_file(argv[1], &vertices);

    for(int iii = 0; iii < vertices; iii++)
      fords(incidence_matrix, vertices, iii);

    free(incidence_matrix);

    return 0;
}
