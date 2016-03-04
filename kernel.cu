#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <direct.h>
#include <math.h>

#include "simpleini\simpleIni.h"
#include "SDL.h"

#undef main

FILE* c_output;
int timeStep;

int N = 100;
int S = 1; // STEPS
bool logStamp;
bool logInit;
bool logChanges;

void loadOptions() {
	// Loads up the config.ini file, and extracts parameters
	printf("Loading Options... \n", N);

	logStamp = true;
	logInit = true;
	logChanges = true;

	CSimpleIniA ini;
	ini.SetUnicode();
	ini.LoadFile("config.ini");
	printf("Completed! \n\nParameters \n", N);
	// Size paramater
	const char * value = ini.GetValue("General", "n", "0");

	// error checking
	if (!atoi(value)) {
		printf("Error Parsing Size, using default value (N = 100) \n");
	}
	else {
		N = atoi(value);
		printf(" N = %d \n", N);
	}

	// Size paramater
	const char * steps = ini.GetValue("General", "s", "1");

	// error checking
	if (!atoi(steps)) {
		printf("Error Parsing Steps, using default value (S = 1) \n");
	}
	else {
		S = atoi(steps);
		printf(" TimeSteps = %d \n", S);
	}

	const char *sizestamp = ini.GetValue("Logs", "sizestamp", "yes");
	if (sizestamp[0] == 'n' || sizestamp[0] == '0' || sizestamp[0] == 'f' || sizestamp[0] == 'F')
		logStamp = false;
	const char *loginit = ini.GetValue("Logs", "loginit", "yes");
	if (loginit[0] == 'n' || loginit[0] == '0' || loginit[0] == 'f' || loginit[0] == 'F')
		logInit = false;
	const char *logchanges = ini.GetValue("Logs", "logchanges", "yes");
	if (logchanges[0] == 'n' || logchanges[0] == '0' || logchanges[0] == 'f' || logchanges[0] == 'F')
		logChanges = false;
}

void pullSpecs(int *blocks, int *threads) {
	int deviceCount, device;
	int gpuDeviceCount = 0;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999)
			if (device == 0)
			{
				*blocks = properties.multiProcessorCount;
				*threads = properties.maxThreadsPerMultiProcessor;
				printf("Success!\n\n");
			}
	}
}

__global__ void processChunk(char *value, char* oldFrame, int bm, int br, int tm, int tr, int N, int seed) {
	//int sIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int b = bm;
	if (br > 0 && blockIdx.x < br) {
		b++;
	}

	int t = tm;
	if (tr > 0 && threadIdx.x < tr) {
		t++;
	}

	curandState_t state;

	curand_init((threadIdx.x + blockIdx.x) * seed / blockIdx.x, 0, 0, &state);

	for (int z = 0; z < b; z++) {
		int y = 0;
		for (int x = 0; x < t; x++) {
			for (int y = 0; y < t; y++) {
				int index = (blockDim.x * x + threadIdx.x % N) + (blockDim.x * y + threadIdx.x / N) * N + (gridDim.x * z + blockIdx.x) * N * N;
				if (index < N*N*N) { 
					// sanity check
					int newValue = curand(&state) % 3;
					//Use old values to determine probablility of mutation
					//if (value[index] != newValue) {
						//value[index] = newValue;
						value[index] = 5;
					//}
					//value[index] = blockIdx.x + 5;
				}
			}
		}
	}
}

// EACH CHARACTER POSSIBLY CONTAINS DATA FOR 4 CELLS
// BYTE IS BROKEN AS FOLLOWS :
// 00 / 01 / 10 / 11
int main(int argc, char *args[])
{
	// Load Specs
	int blocks, threads;
	printf("Retreiveing Hardware Specifications... \n");

	pullSpecs(&blocks, &threads);
	printf("multiProcessorCount %d\n", blocks);
	printf("maxThreadsPerMultiProcessor %d\n", threads);
	printf("\n");
	
	// Load Options
	loadOptions();
	printf("\n");

	// Adds numbers and stuff
	int SIZE = N*N*N; // N^3 Size
	char *a;
	char *n_a;
	char *d_a;
	char *d_init;

	a = (char*)malloc(SIZE*sizeof(char));
	n_a = (char*)malloc(SIZE*sizeof(char));

	int check = mkdir("outputs");

	char buff[256];
	if (logStamp)
		sprintf(buff, "outputs/InitState_%d.csv", N);
	else
		sprintf(buff, "outputs/InitState.csv", N);

	FILE* output;
	if (logInit) {
		output = fopen(buff, "w");
		fputs("X,Y,Z,CELLTYPE\n", output);
	}
	srand(time(NULL));

	printf("Generating Initial Population\n");
	//a contains all value
	for (int i = 0; i < SIZE; i++) {
		// if healthy, glia = 10, normal = 00, cancer glia = 11, cancer = 01
		int g = rand() % 2;
		a[i] = (rand() % 2) * (g+g); // everything starts as a healthy cell
		// This is where cell structure will be defined

		if (logInit)
			fprintf(output, "%d,%d,%d,%d\n", i % N, (i / N) % N, i / (N * N), a[i]);
	}
	// write the initial state
	if (logInit)
		fclose(output);
	 
	printf("Success\n\n");

	char c_buff[256];
	if (logStamp)
		sprintf(c_buff, "outputs/ChangeState_%d.csv", N);
	else 
		sprintf(c_buff, "outputs/ChangeState.csv", N);
	if (logChanges) {
		c_output = fopen(c_buff, "w");
		fputs("X,Y,Z,CELLTYPE,TIMESTEP\n", c_output);
	}

	printf("Transferring Data to GPU\n");
	cudaMalloc(&d_a, SIZE*sizeof(char));
	cudaMalloc(&d_init, SIZE*sizeof(char));
	// have to load initial state, there is no way around it
	cudaMemcpy(d_a, a, SIZE*sizeof(char), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_init, a, SIZE*sizeof(char), cudaMemcpyHostToDevice);
	printf("Success\n\n");

	int blockCount = min(N, blocks);
	int threadCount = min(N*N, threads * 13 / 19);

	int bm = N / blockCount;
	int br = N % blockCount;

	int tm = N*N / threadCount;
	int tr = N*N % threadCount;

	printf("Crunching Data\n", tm, tr);
	for (int timeStep = 0; timeStep < S; timeStep++) {
		int seed = rand();
		processChunk <<<blockCount, threadCount>> >(d_a, d_init, bm, br, tm, tr, N, seed);
		cudaDeviceSynchronize(); // Force Kernels to complete

		cudaMemcpy(n_a, d_a, SIZE*sizeof(char), cudaMemcpyDeviceToHost);

		for (int i = 0; i < SIZE; i++) {
			// if healthy, glia = 10, normal = 00, cancer glia = 11, cancer = 01
			if (a[i] != n_a[i])  {
				// only record differences, also update a[i]
				a[i] = n_a[i];
				// Send changes back to the GPU
				if (logChanges)
					fprintf(c_output, "%d,%d,%d,%d,%d\n", i % N, (i / N) % N, i / (N * N), n_a[i], timeStep);
				cudaMemcpy(&d_init[i], &a[i], sizeof(char), cudaMemcpyHostToDevice);
			}
		}
	}

	if (logChanges)
		fclose(c_output);
	
	printf("Simulation Completed\n");

	while (true) {

	}
	/*
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

		cudaMemcpy(d_a, a, SIZE*sizeof(char), cudaMemcpyHostToDevice);

		//addChange << <1, SIZE >> >(d_a, SIZE);

		cudaMemcpy(a, d_a, SIZE*sizeof(char), cudaMemcpyDeviceToHost);

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
	*/
	free(a);
	cudaFree(d_a);
	return 0;
}

