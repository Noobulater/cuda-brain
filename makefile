all:
	nvcc kernel.cu -o kernel.a `sdl2-config --cflags --libs`

run:
	./kernel.a
