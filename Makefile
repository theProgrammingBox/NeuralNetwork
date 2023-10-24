all:
	nvcc Source.cu -lcublas -o a.out && ./a.out && rm -f a.out