#Simple makefile for dijkstra.cu
#'make' to build unspecial default
#'make clean' to clean up the place

default:
	@echo Please specify make target: dijkstra, clean
dijkstra:
	nvcc -o dijkstra.out dijkstra.cu
clean:
	rm -f dijkstra.out 
