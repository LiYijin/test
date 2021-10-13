TARGET=sgemm
INCLUDES=-I../include/
FLAGS=  -lcublas  -O2
src=$(wildcard *.cu)

all: t1
prof:t1
	sudo /usr/local/cuda/bin/ncu --export proflog --force-overwrite --target-processes application-only \
	 --kernel-regex-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis \
	 --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
	 --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables  \
	 --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight \
	 --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto \
	 --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 \
	 --cache-control all --clock-control base --apply-rules yes  main_test

	 sudo /usr/local/cuda/bin/ncu  --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum main_test

t1:$(src)
	nvcc $(INCLUDES) $(FLAGS) $(src) -o t1

debug:
	nvcc $(INCLUDES) -lcublas -g -G $(src) -o t1

clean:
	rm t1

test:
	nvcc  $(INCLUDES) -gencode arch=compute_75,code=sm_75 sharedmm.cu test1.cu -O2 -o t1