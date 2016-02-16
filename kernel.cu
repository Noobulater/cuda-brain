#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "SDL.h"
#undef main

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char *args[])
{
	int N = 1 << 20;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	saxpy << <(N + 255) / 256, 256 >> >(N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));
	printf("Max error: %fn", maxError);


	SDL_Window* pWindow = NULL;
	pWindow = SDL_CreateWindow("Brain Cancer Start", SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		640,
		480,
		SDL_WINDOW_SHOWN);

	SDL_Surface* pSurface = NULL;
	pSurface = SDL_GetWindowSurface(pWindow);


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
		SDL_FillRect(pSurface, NULL, 0xFFF000);
		SDL_UpdateWindowSurface(pWindow);
	}

	SDL_FreeSurface(pSurface);
	SDL_DestroyWindow(pWindow);
	SDL_Quit();
	return 0;
}

