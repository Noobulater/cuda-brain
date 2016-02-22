#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "SDL.h"
#undef main

__global__
void addChange(int *a, int n) {
	int index = threadIdx.x;

	if (index < n) {
		a[index]++;
		if (a[index] > 255) {
			a[index] = 255;
		}
	}
}

int main(int argc, char *args[])
{
	// Adds numbers and stuff
	int SIZE = 10;
	int *a;
	int *d_a;

	a = (int*)malloc(SIZE*sizeof(int));

	//zero a
	for (int i = 0; i < SIZE; i++) {
		a[i] = 0;
	}

	cudaMalloc(&d_a, SIZE*sizeof(int));

	cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);

	addChange<<<1, SIZE>>>(d_a, SIZE);

	cudaMemcpy(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

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
		for (int core = 0; core < SIZE; core++) {
			for (int i = 0; i < screenWidth / 40; i++) {
				for (int j = 0; j < screenHeight / 40; j++) {
					SDL_SetRenderDrawColor(pRender, a[core], 105, 180, 255);
					SDL_RenderDrawPoint(pRender, core * (screenWidth / 40) + i, j + 40);
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

