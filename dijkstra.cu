#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define DEBUG 0

// probably wont need this after some updates
void minDistance(int *dist, int *used, int *min_index) {
	int min = INT_MAX; 
   
   for (int v = 0; v < V; v++)
     if (used[v] == false && dist[v] <= min) 
         min = dist[v], *min_index = v;
}

void dijktra(int **graph, int size, int src) {
	// TODO fix this shit
	bool *h_used;
	bool *d_used;
	int *h_dist;
	int *d_dist;
	int *h_mutex;
	int *d_mutex;
	int *h_min;
	int *d_min;

	// allocate stuff
	h_used = (bool*) malloc(sizeof(bool) * size);
	h_dist = (int*) malloc(sizeof(int) * size);
	h_mutex = (int*) malloc(sizeof(int));
	h_min = (int*) malloc(sizeof(int));

	cudaMalloc((void**) &d_used, sizeof(int) * size);
	cudaMalloc((void**) &d_min, sizeof(int));
	cudaMalloc((void**) &d_mutex, sizeof(int));
	cudaMalloc((void**) &d_dist, sizeof(int) * size);

	// computations
	#pragma omp parallel for
	for(int iii = 0; iii < size; iii++){
		used[iii] = false; h_dist[iii] = INT_MAX;
	}

	h_dist[src] = 0;
	dim3 thread_size = 256; // can't use variable size so everything is hard-coded
	dim3 block_size = 256; // 

	for(int iii = 0; iii < size = 1; iii++) {
		minDistance(h_dist, h_used, h_min);
		h_used[*h_min] = true;
	}

	// free later
	free(h_used);
	free(h_dist);
	free(h_mutex);
	free(h_min);
	cudaFree(d_used);
	cudaFree(d_dist);
	cudaFree(d_mutex);
	cudaFree(d_min);

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
