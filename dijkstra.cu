#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define DEBUG 0

__global__ void find_minimum_kernel(int *array, int *min, int *mutex, unsigned int n) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;
	int temp_min = 1000;

	__shared__ int cache[256];

	while(index + offset < n) {
		temp_min = fminf(temp_min, array[index + offset]);
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
	for(int iii = 0; iii < size; iii++) {
		for(int jjj = 0; jjj < size; jjj++) {
			printf("%d ", graph[iii][jjj]);
		}
		printf("\n");
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
	for(int iii = 0; iii < v; iii++) incidence_matrix[iii] = (int*)malloc(v * sizeof(int));
	for(int iii = 0; iii < v; iii++) {
		printf("iii is %d\n", iii);
		for(int jjj = 0; jjj < v; jjj++) {
			fscanf(f, "%d", &incidence_matrix[iii][jjj]);
			printf("%d\n", incidence_matrix[iii][jjj]);
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

	printf("calling dijktra\n");

	dijktra(incidence_matrix, vertices, atoi(argv[2]));

	return 0;
}
