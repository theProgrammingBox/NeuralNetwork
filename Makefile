all:
	# nvcc Source.cu -lcublasLt -o a.out && ./a.out && rm a.out
	nvcc Source2.cu -lcublasLt -o a.out && ./a.out && rm a.out
	# nvcc Source3.cu -lcublasLt -o a.out && ./a.out && rm a.out