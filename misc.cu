#include "misc.h"

#include <stdlib.h>
#include <stdio.h>

__inline__ __host__ __device__
bool contains(int3 s, int e){
   return (s.x==e || s.y==e || s.z==e);
}

__global__
void inc(float * data, int N){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < N) data[idx]++;
}

__global__
void subset_G(float * data, int N, int3 subset){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < N && contains(subset, idx)) data[idx]++;
}

__global__
void subset_G_Rand(float * in, float * out, int N, float thresh){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < N){
		float val = in[idx];
		if (val < thresh) out[idx] = val;
	}
}

void AllGPUAllCpu(int N, int nIt){
	float * h_Data(0), * d_Data(0);
	int size = sizeof(float)*N;

	h_Data = (float *)malloc(size);
	cudaMalloc((void **)&d_Data, size);

	for (int i=0; i<nIt; i++){
		cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);
		inc<<<N/1024, 1024>>>(d_Data, N);
		cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);
		for (int j=0; j<N; j++)
			h_Data[j]++;
	}

	free(h_Data);
	cudaFree(d_Data);
}

void SubGpuAllCpu(int N, int nIt){
	float * h_Data(0), * d_Data(0);
	int size = sizeof(float)*N;

	srand(1);

	int3 subset;
	subset.x = (int)(((float)rand()/(float)RAND_MAX) * N);
	subset.y = (int)(((float)rand()/(float)RAND_MAX) * N);
	subset.z = (int)(((float)rand()/(float)RAND_MAX) * N);

	h_Data = (float *)malloc(size);
	cudaMalloc((void **)&d_Data, size);

	for (int i=0; i<nIt; i++){
		cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);
		subset_G<<<N/1024, 1024>>>(d_Data, N, subset);
		cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);
		for (int j=0; j<N; j++)
			h_Data[j]++;
	}

	free(h_Data);
	cudaFree(d_Data);
}

void SubGpuAllCpu_R(int N, int nIt, float thresh/* = -1.f*/){
	float * h_In(0), * h_Out(0), * d_In(0), * d_Out(0);
	int size = sizeof(float)*N;

	srand(1);

	if (thresh < 0)
		thresh = (float)rand()/(float)RAND_MAX;

	h_In = (float *)malloc(size);
	h_Out = (float *)malloc(size);
	cudaMalloc((void **)&d_In, size);
	cudaMalloc((void **)&d_Out, size);

	for (int j=0; j<N; j++){
		h_In[j] = (float)rand()/(float)RAND_MAX;
		h_Out[j] = (float)rand()/(float)RAND_MAX;
	}

	for (int i=0; i<nIt; i++){
		cudaMemcpy(d_In, h_In, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Out, h_Out, size, cudaMemcpyHostToDevice);
		subset_G_Rand<<<N/1024, 1024>>>(d_In, d_Out, N, thresh);
		cudaMemcpy(h_In, d_In, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Out, d_Out, size, cudaMemcpyDeviceToHost);
		for (int j=0; j<N; j++){
			h_In[j]++;
			h_Out[j]++;
		}
	}

	free(h_In);
	free(h_Out);
	cudaFree(d_In);
	cudaFree(d_Out);
}

void AllGpuSubCpu(int N, int nIt){
	float * h_Data(0), * d_Data(0);
	int size = sizeof(float)*N;

	srand(1);

	int3 subset;
	subset.x = (int)(((float)rand()/(float)RAND_MAX) * N);
	subset.y = (int)(((float)rand()/(float)RAND_MAX) * N);
	subset.z = (int)(((float)rand()/(float)RAND_MAX) * N);

	h_Data = (float *)malloc(size);
	cudaMalloc((void **)&d_Data, size);

	for (int i=0; i<nIt; i++){
		cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);
		inc<<<N/1024, 1024>>>(d_Data, N);
		cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);
		for (int j=0; j<N; j++)
			if (contains(subset, j))
				h_Data[j]++;
	}

	free(h_Data);
	cudaFree(d_Data);
}
