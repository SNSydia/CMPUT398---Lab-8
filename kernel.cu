#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this


#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
// TODO: write kernel to uniform add each aux array value to corresponding block output
__global__ void blockadd(float* g_aux, float* g_odata, int n){
	int id = blockIdx.x*blockDim.x + threadIdx.x; //Id of the thread within the block

	if (blockIdx.x > 0 && id < n){
		g_odata[id] += g_aux[blockIdx.x];
	}

}

// TODO: write a simple transpose kernel here

// TODO: write 1D scan kernel here

// TODO: write recursive scan wrapper on CPU here

int main(int argc, char **argv) {

	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;  // device input
	float *deviceTmpOutput;  // temporary output
	float *deviceOutput;  // ouput
	int numInputRows, numInputCols; // dimensions of the array

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numInputRows, &numInputCols);
	cudaHostAlloc(&hostOutput, numInputRows * numInputCols * sizeof(float),
		cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of input are ",
		numInputRows, "x", numInputCols);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numInputRows * numInputCols * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numInputRows * numInputCols * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceTmpOutput, numInputRows * numInputCols * sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numInputRows * numInputCols * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numInputRows * numInputCols * sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");
	//TODO: Modify this to complete the functionality of the scan on the deivce
	for (int i = 0; i < numInputRows; ++i) {
		// TODO: call your 1d scan kernel for each row here

		wbCheck(cudaDeviceSynchronize());
	}

	// You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
	dim3 transposeBlockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
	dim3 transposeGridDim(ceil(numInputCols / (float)TRANSPOSE_TILE_DIM), ceil(numInputRows / (float)TRANSPOSE_TILE_DIM));
	// TODO: call your transpose kernel here

	wbCheck(cudaDeviceSynchronize());

	for (int i = 0; i < numInputCols; ++i) {
		// TODO: call your 1d scan kernel for each row of the tranposed matrix here

		wbCheck(cudaDeviceSynchronize());
	}

	// You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
	transposeBlockDim = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
	transposeGridDim = dim3(ceil(numInputRows / (float)TRANSPOSE_TILE_DIM), ceil(numInputCols / (float)TRANSPOSE_TILE_DIM));
	// TODO: call your transpose kernel to get the final result here

	wbCheck(cudaDeviceSynchronize());

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numInputRows * numInputCols * sizeof(float),
		cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceTmpOutput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numInputRows, numInputCols);

	free(hostInput);
	cudaFreeHost(hostOutput);

	wbCheck(cudaDeviceSynchronize());

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
