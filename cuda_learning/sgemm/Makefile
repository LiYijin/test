TARGET=sgemm
INCLUDES=-I../include/
FLAGS=-g -G  -lcublas -O2
src=$(wildcard *.cu)

all:t1
prof:t1
	sudo /usr/local/cuda/bin/ncu --export proflog --force-overwrite --target-processes application-only \
	 --kernel-regex-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis \
	 --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
	 --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables  \
	 --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight \
	 --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto \
	 --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 \
	 --cache-control all --clock-control base --apply-rules yes  t1

t1:$(src)
	nvcc $(INCLUDES) $(FLAGS) $(src) -o t1