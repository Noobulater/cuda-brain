#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SDL.h"
#undef main

__global__
void addChange(int *a, int n) {
	int index = threadIdx.x;

	curandState_t state;

	curand_init(index + a[index], 0, a[index], &state);

	if (index < n) {
		a[index] = curand(&state) % 255;
	}
}

int main(int argc, char *args[])
{
	// Adds numbers and stuff
	int SIZE = 170;
	int *a;
	int *d_a;

	a = (int*)malloc(SIZE*sizeof(int));

	//zero a
	for (int i = 0; i < SIZE; i++) {
		a[i] = 0;
	}

	cudaMalloc(&d_a, SIZE*sizeof(int));

	int screenWidth = 640;
	int screenHeight = 480;

	SDL_Window* pWindow = NULL;
	pWindow = SDL_CreateWindow("Brain Cancer Start", SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		screenWidth,
		screenHeight,
		SDL_WINDOW_SHOWN);

	SDL_Renderer* pRender = SDL_CreateRenderer(pWindow, -1, 0);

	SDL_Event evt;
	bool alive = true;
	while (alive) {
		while (SDL_PollEvent(&evt)) {
			if (evt.type == SDL_QUIT) {
				alive = false;
				break;
			}
			if (evt.type == SDL_KEYDOWN && evt.key.keysym.sym == SDLK_ESCAPE) {
				alive = false;
				break;
			}
		}
		int square = 40;
		int width = (screenWidth / square);
		int height = (screenHeight / square);

		cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);

		addChange << <1, SIZE >> >(d_a, SIZE);

		cudaMemcpy(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

		for (int core = 0; core < SIZE; core++) {
			int i = core % width;
			
			int j = 0;
			if (core >= width) {
				j = core / width;
			}
			for (int x = 0; x < square; x++) {
				for (int y = 0; y < square; y++) {
					SDL_SetRenderDrawColor(pRender, a[core], 105, 180, 255);
					SDL_RenderDrawPoint(pRender, i * square + x, j * square + y);
				}
			}

		}
		SDL_RenderPresent(pRender);
	}

	SDL_DestroyWindow(pWindow);
	SDL_Quit();

	free(a);
	cudaFree(d_a);
	return 0;
}

