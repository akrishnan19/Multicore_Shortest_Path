#Simple makefile for dijkstra.cu
#'make' to build unspecial default
#'make clean' to clean up the place

default:
	@echo Please specify make target: dijkstra, fords, clean
dijkstra:
	nvcc -o dijkstra dijkstra.cu
fords:
	nvcc -o fords fords.cu
clean:
	rm -f dijkstra fords
