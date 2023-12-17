all:
	nvcc Source4.cu -lcublas -o a.out && ./a.out && rm a.out