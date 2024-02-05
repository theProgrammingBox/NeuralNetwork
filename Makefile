all:
	# nvcc test3.cu -lcublas -I/usr/lib/cuda/include -L/usr/lib/cuda/lib64 -lcudnn -o a.out && ./a.out && rm a.out
	nvcc test3.cu -lcublas -o a.out && ./a.out && rm a.out