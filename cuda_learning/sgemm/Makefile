TARGET=sgemm
INCLUDES=-I../include/
FLAGS=-g -G  -lcublas -O2
src=$(wildcard *.cu)
all:
	nvcc $(INCLUDES) $(FLAGS) $(src) -o t1