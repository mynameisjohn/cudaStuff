#include "misc.h"
#include "gsRelax.h"

//Fix block size for N <= 1024

int main(){
	const uint32_t N(2048), nIt(1000);
	const float minRes(0.1f), thresh = -1.f;
	int dim = 2;

	AllGPUAllCpu(N, nIt);
	SubGpuAllCpu(N, nIt);
	SubGpuAllCpu_R(N, nIt,thresh);
	AllGpuSubCpu(N, nIt);

	relax(N, dim, nIt, minRes);
	
	return 0;
}
