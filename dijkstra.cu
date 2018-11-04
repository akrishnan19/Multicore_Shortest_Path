#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define DEBUG 0

// probably wont need this after some updates
void minDistance(int *dist, int *used, int *min_index, int numV) {
	int min = INT_MAX; 
   
   for (int v = 0; v < numV; v++)
     if (used[v] == false && dist[v] <= min && dist[v] != 0) 
         min = dist[v], *min_index = v;
}

__global__ void find_minimum_kernel(int *dist, bool* used, int* min, int *mutex, unsigned int n) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;
	int temp_min = INT_MAX;

	__shared__ int cache[256];

	while(index + offset < n) {
		if((dist[index + offset] != 0) && (d_used[index+offset] == false)){
			temp_min = fminf(temp_min, dist[index + offset]);
		}
		offset += stride;
	}

	cache[threadIdx.x] = temp_min;

	__syncthreads();

	unsigned int iii = blockDim.x / 2;
	while(iii > 0) {
		if(threadIdx.x < iii) {
			cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + iii]);
		}
		__syncthreads();
		iii >>= 1;
	}

	if(threadIdx.x == 0) {
		while(atomicCAS(mutex, 0, 1) != 0); // acquire mutex
		*min = fminf(*min, cache[0]);
		atomicExch(mutex, 0); // unlock mutex
	}
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
		h_used[iii] = false;
		h_dist[iii] = INT_MAX;
	}

	h_dist[src] = 0;
	dim3 THREAD_SIZE = 256; // can't use variable size so everything is hard-coded
	dim3 BLOCK_SIZE = 256; // 

	for(int iii = 0; iii < size - 1; iii++) {
		int min_calculated = INT_MAX;
		printf("Starting execution %d of %d\n", iii, size - 1);
		// minDistance(h_dist, h_used, h_min, *size);// more efficient to run on CPU than offloading to GPU
		cudaMemcpy(d_used, h_used, sizeof(bool)*size, cudaMemcpyHostToDevice);
		find_minimum_kernel<<< BLOCK_SIZE, THREAD_SIZE >>>(d_dist, d_used, d_min, d_mutex, size);// GPU implementation anyway - could we check the entire graph
		cudaMemcpy(&min_calculated, d_min, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dist, d_dist, sizeof(int)*size, cudaMemcpyDeviceToHost);
		printf("Successfully read min at %d\n", min_calculated);
		h_used[min_calculated] = true;

		for(int jjj = 0; jjj < size; jjj++){
			if(!h_used[iii] &&
				(h_dist[min_calculated] != INT_MAX) &&
				(h_dist[min_calculated] + graph[min_calculated][jjj] < h_dist[jjj]) &&
				graph[min_calculated][jjj]){
					h_dist[jjj] = h_dist[min_calculated] + graph[min_calculated][jjj];

			}
		}
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
	for(int iii = 0; iii < v; iii++){
		incidence_matrix[iii] = (int*)malloc(v * sizeof(int));
	}

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
