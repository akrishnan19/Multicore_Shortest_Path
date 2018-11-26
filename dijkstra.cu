#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include <time.h>

__global__ void find_minimum_kernel(int *dist, bool* used, int* min, int* min_index, int *mutex, unsigned int n) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;
	int temp_min = INT_MAX;
	int temp_min_index = -1;

	__shared__ int cache[256];
	__shared__ int cache2[256];

	while(index + offset < n) {
		if((used[index+offset] == false)) {
			int temp_temp_min = temp_min;
			temp_min = fminf(temp_min, dist[index + offset]);
			if(temp_temp_min != temp_min){
				temp_min_index = index + offset;
			}
		}
		offset += stride;
	}

	cache[threadIdx.x] = temp_min;
	cache2[threadIdx.x] = temp_min_index;

	__syncthreads();

	unsigned int iii = blockDim.x / 2;
	while(iii > 0) {
		if(threadIdx.x < iii) {
			int temp_temp_min = cache[threadIdx.x];
			cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + iii]);
			if(temp_temp_min != cache[threadIdx.x]){
				cache2[threadIdx.x] = cache2[threadIdx.x + iii];
			}
		}
		__syncthreads();
		iii >>= 1;
	}

	if(threadIdx.x == 0) {
		while(atomicCAS(mutex, 0, 1) != 0); // acquire mutex
		int temp_temp_min = *min;
		*min = fminf(*min, cache[0]);
		if(temp_temp_min != *min){
			*min_index = cache2[0];
		}
		atomicExch(mutex, 0); // unlock mutex
	}
}

__global__ void update_dist_kernel(int *dist, bool *used, int *graph, int u, int n) {
	unsigned int jjj = threadIdx.x + blockIdx.x * blockDim.x;

	if(jjj < n) {
		if(!used[jjj] &&
		(dist[u] != INT_MAX) &&
		(dist[u] + graph[jjj] < dist[jjj]) &&
		graph[jjj]) {
			dist[jjj] = dist[u] + graph[jjj];
		}
	}
}

void printResults(int dist[], int n, int source) { 
	printf("Vertex\t\tDistance from Source Vertex %d\n", source);
	for (int i = 0; i < n; i++)
		printf("%d\t\t%d\n", i, dist[i]);
	printf("\n\n");
} 

void dijktra(int **graph, int size, int src) {
	bool *h_used;
	bool *d_used;
	int *h_dist;
	int *d_dist;
	int *h_mutex;
	int *d_mutex;
	int *d_min;
	int *d_min_index;
	int *d_graph;
	int MAX = INT_MAX;

	// allocate stuff
	h_used = (bool*) malloc(sizeof(bool) * size);
	h_dist = (int*) malloc(sizeof(int) * size);
	h_mutex = (int*) malloc(sizeof(int));

	cudaMalloc((void**) &d_used, sizeof(int) * size);
	cudaMalloc((void**) &d_min, sizeof(int));
	cudaMalloc((void**) &d_mutex, sizeof(int));
	cudaMalloc((void**) &d_dist, sizeof(int) * size);
	cudaMalloc((void**) &d_min_index, sizeof(int));
	cudaMalloc((void**) &d_graph, sizeof(int) * size);

	// computations
	#pragma omp parallel for
	for(int iii = 0; iii < size; iii++){
		h_used[iii] = false;
		h_dist[iii] = INT_MAX;
	}

	h_dist[src] = 0;
	dim3 THREAD_SIZE = 256; // can't use variable size so everything is hard-coded
	dim3 BLOCK_SIZE = 256;

	cudaMemcpy(d_dist, h_dist, sizeof(int)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_used, h_used, sizeof(bool)*size, cudaMemcpyHostToDevice);

	for(int iii = 0; iii < size - 1; iii++) {
		int min_calculated = -1;

		cudaMemcpy(d_min, &MAX, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_min_index, -1, sizeof(int));

		find_minimum_kernel<<< BLOCK_SIZE, THREAD_SIZE >>>(d_dist, d_used, d_min, d_min_index, d_mutex, size);

		cudaMemcpy(&min_calculated, d_min_index, sizeof(int), cudaMemcpyDeviceToHost);

		if(min_calculated == -1) {
			printf("No min calculated. This is bad.\n");
			exit(-1);
		}
		h_used[min_calculated] = true;

		cudaMemcpy(&d_used[min_calculated], &h_used[min_calculated], sizeof(bool), cudaMemcpyHostToDevice);
		
		cudaMemcpy(d_graph, graph[min_calculated], sizeof(int) * size, cudaMemcpyHostToDevice);
		update_dist_kernel<<< BLOCK_SIZE, THREAD_SIZE >>>(d_dist, d_used, d_graph, min_calculated, size);
		
		/*
		#pragma omp parallel for
		for(int jjj = 0; jjj < size; jjj++) {
			if(!h_used[jjj] &&
			(h_dist[min_calculated] != INT_MAX) &&
			(h_dist[min_calculated] + graph[min_calculated][jjj] < h_dist[jjj]) &&
			graph[min_calculated][jjj]) {
				h_dist[jjj] = h_dist[min_calculated] + graph[min_calculated][jjj];
				printf("got in\n");
			}
		}
		*/
	}
	cudaMemcpy(h_dist, d_dist, sizeof(int) * size, cudaMemcpyDeviceToHost);
	printResults(h_dist, size, src);

	// free later
	free(h_used);
	free(h_dist);
	free(h_mutex);
	cudaFree(d_used);
	cudaFree(d_dist);
	cudaFree(d_mutex);
	cudaFree(d_min);
	cudaFree(d_min_index);
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
	for(int iii = 0; iii < v; iii++){
		incidence_matrix[iii] = (int*)malloc(v * sizeof(int));
	}

	for(int iii = 0; iii < v; iii++) {
		for(int jjj = 0; jjj < v; jjj++) {
			fscanf(f, "%d", &incidence_matrix[iii][jjj]);
			if(incidence_matrix[iii][jjj] < 0) {
				printf("Error: negative edge weight found at %d, %d\n", iii, jjj);
				exit(-1);
			}
		}
	}

	return incidence_matrix;
}

int main(int argc, char *argv[]) {
	int **incidence_matrix;
	int vertices;
	// clock_t start, end;

	if(argc < 2 || argc > 3) {
		printf("Incorrect usage\n"); // sanity check
		printf("Correct usage: ./dijktra path_to_file");
		exit(1);
	}
	
	incidence_matrix = read_file(argv[1], &vertices);
	
	// start = clock();
	if(argc == 2)
		for(int iii = 0; iii < vertices; iii++)
			dijktra(incidence_matrix, vertices, iii);
	else
		dijktra(incidence_matrix, vertices, atoi(argv[2]));
	// end = clock();
	// printf("Time taken: %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);
	free(incidence_matrix);

	return 0;
}
