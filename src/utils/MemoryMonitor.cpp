/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#include "includes.h"

/* constructor */
MemoryMonitor::MemoryMonitor()
{
    cpuMemory = 0;
#if GPU == 1
    gpuMemory = 0;
#endif
}

/* malloc cpu memory */
void MemoryMonitor::cpuMalloc(void **hostPtr, int size)
{
    cpuMemory += size;
    *hostPtr = (void *)malloc(size);
    memset(*hostPtr, 0, size);
    cpuPoint[*hostPtr] = size;
}

/* free cpu memory */
void MemoryMonitor::freeCpuMemory(void *ptr)
{
    if (cpuPoint.find(ptr) != cpuPoint.end())
    {
        cpuMemory -= cpuPoint[ptr];
        free(ptr);
        cpuPoint.erase(ptr);
    }
}

/* print total malloc cpu memory */
void MemoryMonitor::printCpuMemory()
{
    printf("total malloc cpu memory %fMb\n", cpuMemory / 1024.0f / 1024.0f);
}

#if GPU == 1
/* malloc gpu memory */
void MemoryMonitor::gpuMalloc(void **devPtr, int size)
{
    gpuMemory += size;
    cudaError_t error = cudaMalloc(devPtr, size);
    Assert(error == cudaSuccess, "Device memory allocation failed.");
    gpuPoint[*devPtr] = size;
}

/* free gpu memory */
void MemoryMonitor::freeGpuMemory(void *ptr)
{
    if (gpuPoint.find(ptr) != gpuPoint.end())
    {
        gpuMemory -= gpuPoint[ptr];
        cudaFree(ptr);
        gpuPoint.erase(ptr);
    }
}

/* print total malloc gpu memory */
void MemoryMonitor::printGpuMemory()
{
    printf("total malloc gpu memory %fMb\n", gpuMemory / 1024.0f / 1024.0f);
}
#endif
