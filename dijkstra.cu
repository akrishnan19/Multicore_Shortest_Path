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

__global__ void find_last_digit_kernel(int *arrayA, int *arrayB, unsigned int n) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	while(index + offset < n) {
		arrayB[index + offset] = arrayA[index + offset] % 10;
		offset += stride;
	}

	__syncthreads();

}

typedef struct {
  int *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
  a->array = (int *)malloc(initialSize * sizeof(int));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Array *a, int element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = (int *)realloc(a->array, a->size * sizeof(int));
  }
  a->array[a->used++] = element;
}

void freeArray(Array *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

void find_max(Array *a, int* h_min) { // passing the dynamic array
	unsigned int size = a->used;
	int *h_array;
	int *d_array;
	int *d_min;
	int *d_mutex;
	int *h_mutex;

	// allocate to memory
	h_array = (int*) malloc(sizeof(int) * size);
	h_mutex = (int*) malloc(sizeof(int));
	cudaMalloc((void**) &d_array, sizeof(int) * size);
	cudaMalloc((void**) &d_min, sizeof(int));
	cudaMalloc((void**) &d_mutex, sizeof(int));
	*h_mutex = 0;

	// create a copy of the dynamic array to pass to the gpu
	for(int iii = 0; iii < size; iii++) {
		h_array[iii] = a->array[iii];
	}

	// copy from host to device
	cudaMemcpy(d_array, h_array, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_min, h_min, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mutex, h_mutex, sizeof(int), cudaMemcpyHostToDevice);

	// call to gpu kernel
	dim3 thread_size = 256; // can't use variable size so everything is hard-coded
	dim3 block_size = 256; // 
	find_minimum_kernel<<< block_size, thread_size >>>(d_array, d_min, d_mutex, size);

	// copy from device back to host
	cudaMemcpy(h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

	// free memory
	free(h_array);
	free(h_mutex);
	cudaFree(d_array);
	cudaFree(d_min);
	cudaFree(d_mutex);
}

void find_last_digit(Array *a, int* h_arrayB) { // passing the dynamic array
	unsigned int size = a->used;
	int *h_arrayA;
	int *d_arrayA;
	int *d_arrayB;

	// allocate to memory
	h_arrayA = (int*) malloc(sizeof(int) * size);
	cudaMalloc((void**) &d_arrayA, sizeof(int) * size);
	cudaMalloc((void**) &d_arrayB, sizeof(int) * size);

	// create a copy of the dynamic array to pass to the gpu
	for(int iii = 0; iii < size; iii++) {
		h_arrayA[iii] = a->array[iii];
	}

	// copy from host to device
	cudaMemcpy(d_arrayA, h_arrayA, sizeof(int) * size, cudaMemcpyHostToDevice);

	// call to gpu kernel
	dim3 thread_size = 256; // can't use variable size so everything is hard-coded
	dim3 block_size = 256; // 
	find_last_digit_kernel<<< block_size, thread_size >>>(d_arrayA, d_arrayB, size);

	// copy from device back to host
	cudaMemcpy(h_arrayB, d_arrayB, sizeof(int) * size, cudaMemcpyDeviceToHost);

	// free memory
	free(h_arrayA);
	cudaFree(d_arrayA);
	cudaFree(d_arrayB);
}

void part_a(Array a) {
	int *h_min;
	FILE *dest;

	dest = fopen("q1a.txt", "w+");

	h_min = (int*) malloc(sizeof(int));
	*h_min = 1000;
	find_max(&a, h_min);

	fprintf(dest, "%d", *h_min);

	#if DEBUG
	*h_min = 1000;
	for(int iii = 0; iii < a.used; iii++) {
		if(a.array[iii] < *h_min) *h_min = a.array[iii];
	}
	printf("minimum value from cpu: %d\n", *h_min);
	#endif
	fclose(dest);
	free(h_min);
}

void part_b(Array a) {
	int *h_arrayB;
	FILE *dest;

	dest = fopen("q1b.txt", "w+");

	h_arrayB = (int*) malloc(sizeof(int) * a.used);
	find_last_digit(&a, h_arrayB);

	// printf("Calculations complete:\n");
	for(int iii = 0; iii < a.used - 1; iii++) {
		fprintf(dest, "%d, ", h_arrayB[iii]);
	}
	fprintf(dest, "%d", h_arrayB[a.used - 1]);

	#if DEBUG
	printf("CPU calculations:\n");
	for(int iii = 0; iii < a.used; iii++) {
		printf("%d, ", a.array[iii] % 10);
	}
	printf("\n");
	#endif
	fclose(dest);
	free(h_arrayB);
}

int main(int argc, char *argv[]) {
	char *file_path;
	FILE *f1;
	int ins_elem;
	
	Array a; // dynamically growing array

	if(argc < 2) {
		printf("Incorrect usage\n"); // sanity check
		exit(1);
	}

	file_path = argv[1];
	f1 = fopen(file_path, "r");
	
	initArray(&a, 1);
	
	fscanf(f1, "%d", &ins_elem);
	insertArray(&a, ins_elem);

	while(fscanf(f1, ", %d", &ins_elem) == 1) {
		insertArray(&a, ins_elem);
	}

	part_a(a);
	part_b(a);

	freeArray(&a);
	fclose(f1);
	
	return 0;
}
