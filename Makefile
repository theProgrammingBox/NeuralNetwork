all:
	nvcc Main.cu -lcublas -I/usr/lib/cuda/include -L/usr/lib/cuda/lib64 -lcudnn -o a.out && ./a.out && rm a.out