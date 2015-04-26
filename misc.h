#ifndef MISC_H
#define MISC_H

void AllGPUAllCpu(int N, int nIt);
void SubGpuAllCpu(int N, int nIt);
void SubGpuAllCpu_R(int N, int nIt, float thresh = -1.f);
void AllGpuSubCpu(int N, int nIt);

//put this here eventually
//void relax(uint32_t N, uint32_t dim, uint32_t nIt, float minRes);

#endif
