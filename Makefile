# using cublasLT
all:
	nvcc Source.cu -lcublasLt -o a.out && ./a.out && rm a.out