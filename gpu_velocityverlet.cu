//nvcc -o gpu_velocityverlet gpu_velocityverlet.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector_types.h>
#include "book.h"
//#include "cuPrintf.cu"

#define SIGMA 1
#define RC 2.5*SIGMA
#define EPS 0.01
#define ZERO 4*EPS*(pow(SIGMA/RC, 12) - pow(SIGMA/RC, 6))

//double OFF = 4*EPS*(pow(2.5, -12) - powf(2.5, -6))

int N, l;
double S, step;

__constant__ int N_dev, l_dev;
__constant__ double S_dev;
__constant__ double step_dev;

// these exist on the gpu side
//texture<double> tex_r, tex_rn;


int blocksPerGrid;
const int threadsPerBlock = 256;

//GPU lock-free synchronization function
//modification: volatile it is.
//modification: now it also finishes summing K and V
__device__ void __gpu_sync(int goalVal, volatile int *Arrayin,
                                        volatile int *Arrayout,
                           float mK, float mV,
                           float *Kb, float *Vb,
                           float *K, float *V)
{
 // thread ID in a block
 int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
 int nBlockNum = gridDim.x * gridDim.y;
 int bid = blockIdx.x * gridDim.y + blockIdx.y;
 // only thread 0 is used for synchronization
 if (tid_in_block == 0)
 {
  Kb[bid] = mK; Vb[bid] = mV;
  Arrayin[bid] = goalVal;
 }
 if (bid == 1)
 {
  if (tid_in_block < nBlockNum)
  {
    while (Arrayin[tid_in_block] != goalVal);
  }
  __shared__ float2 cache[threadsPerBlock];
  if (tid_in_block < nBlockNum) {cache[tid_in_block].x = Kb[tid_in_block];
                                 cache[tid_in_block].y = Vb[tid_in_block];}
  int i = nBlockNum/2;
  __syncthreads();
  while(i != 0)
  {
   if(tid_in_block < i) {cache[tid_in_block].x += cache[tid_in_block+i].x;
                         cache[tid_in_block].y += cache[tid_in_block+i].y;}
   __syncthreads();
   i /= 2;
  }
  if (tid_in_block == 0) {K[goalVal] = cache[0].x; V[goalVal] = cache[0].y;}
  if (tid_in_block < nBlockNum)
  {
   Arrayout[tid_in_block] = goalVal;
  }
 }
 if (tid_in_block == 0)
 {
  while (Arrayout[bid] != goalVal);
 }
 __syncthreads();
}


/*it seems, that if we want one kernel, we need to calculate r at the end. But
 * since we need r from the next step to eventually calculate v for this next
 * step, we need the initialising kernel*/

__global__ void InitVelocityVerlet
           (double *r, double *v, double *a, double *rn)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid == 0) locke = 0;

//calculating a
   double ax_new = 0, ay_new = 0, az_new = 0,
         x_old = r[3*tid],//tex1Dfetch(tex_r, 3*tid),
         y_old = r[3*tid+1],//tex1Dfetch(tex_r, 3*tid+1),
         z_old = r[3*tid+2];//tex1Dfetch(tex_r, 3*tid+2);
   for(int j = 0; j < N_dev; ++j)
   {
      if (tid != j)
      {
         double x = x_old - r[3*j];//tex1Dfetch(tex_r, 3*j);
         if (x > RC) x-= S_dev; else if (x < -RC) x+= S_dev;
         double y = y_old - r[3*j+1];//tex1Dfetch(tex_r, 3*j+1);
         if (y > RC) y-= S_dev; else if (y < -RC) y+= S_dev;
         double z = z_old - r[3*j+2];//tex1Dfetch(tex_r, 3*j+2);
         if (z > RC) z-= S_dev; else if (z < -RC) z+= S_dev;
         double r2 = x*x + y*y + z*z;
         if (r2 < RC*RC)
         {
            double R = pow(SIGMA*SIGMA/r2, 3);
            double part = 4*EPS*R*R, ULJ = part - 4*EPS*R;
            ax_new += 6*x*(part + ULJ)/r2;
            ay_new += 6*y*(part + ULJ)/r2;
            az_new += 6*z*(part + ULJ)/r2;
         }
      }
   }
   a[3*tid] = ax_new; a[3*tid+1] = ay_new; a[3*tid+2] = az_new;

//calculating r
   double x_new = x_old 
                 + step_dev*v[3*tid]//tex1Dfetch(tex_v, 3*tid)
                 + 0.5*ax_new*step_dev*step_dev;
   if (x_new > S_dev) {x_new -= S_dev; }
   else if (x_new < 0) {x_new += S_dev; }
   double y_new = y_old
                 + step_dev*v[3*tid+1]//tex1Dfetch(tex_v, 3*tid+1)
                 + 0.5*ay_new*step_dev*step_dev;
   if (y_new > S_dev) {y_new -= S_dev; }
   else if (y_new < 0) {y_new += S_dev; }
   double z_new = z_old
                 + step_dev*v[3*tid+2]//tex1Dfetch(tex_v, 3*tid+2)
                 + 0.5*az_new*step_dev*step_dev;
   if (z_new > S_dev) {z_new -= S_dev; }
   else if (z_new < 0) {z_new += S_dev; }
   rn[3*tid] = x_new;
   rn[3*tid+1] = y_new;
   rn[3*tid+2] = z_new;

}

__device__ float acalc(double *ax_new, double *ay_new, double *az_new,
     volatile double *x_load, volatile double *y_load, volatile double *z_load,
     volatile double *ro)
{
   int cacheIndex = threadIdx.x;
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   double sum = 0;
//   ax_new = 0; ay_new = 0; az_new = 0;
   for(int ii = blockIdx.x + 1; ii < blockIdx.x+gridDim.x; ++ii)
   {
      int i = (ii%gridDim.x)*blockDim.x;
      x_load[cacheIndex+i] = ro[3*(cacheIndex+i)];
      y_load[cacheIndex+i] = ro[3*(cacheIndex+i)+1];
      z_load[cacheIndex+i] = ro[3*(cacheIndex+i)+2];
   }
      __syncthreads();
      for(int j = 0; j < N_dev; ++j)
         if(j != tid)
         {
            double x, y, z;
            x = x_load[tid] - x_load[j];//tex1Dfetch(tex_rn, 3*j);
            y = y_load[tid] - y_load[j];//tex1Dfetch(tex_rn, 3*j+1);
            z = z_load[tid] - z_load[j];//tex1Dfetch(tex_rn, 3*j+2);}
            if (x > RC) x-= S_dev; else if (x < -RC) x+= S_dev;
            if (y > RC) y-= S_dev; else if (y < -RC) y+= S_dev;
            if (z > RC) z-= S_dev; else if (z < -RC) z+= S_dev;
            double r2 = x*x + y*y + z*z;
            if (r2 < RC*RC)
            {
               double R = pow(SIGMA*SIGMA/r2, 3);
               double part = 4*EPS*R*R, ULJ = part - 4*EPS*R;
               *ax_new += 6*x*(part + ULJ)/r2;
               *ay_new += 6*y*(part + ULJ)/r2;
               *az_new += 6*z*(part + ULJ)/r2;
               sum += ULJ - ZERO;
            }

         }
//      for(int j = tid+1; j < N_dev; ++j)
//         sum += subacalc(j, ax_new, ay_new, az_new, x_load, y_load, z_load);
   return static_cast<float>(sum);
}

//This version calls a thread for every particle.
__global__ void VelocityVerlet
                (double *r, double *v, double *a, 
                 double *rn, float *Kb, float *Vb,
                 float *K, float *V,
                 int *Arrayin, int *Arrayout)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int cacheIndex = threadIdx.x;
//   volatile int bid = blockIdx.x;
   __shared__ float cache[threadsPerBlock];
   __shared__ float cache2[threadsPerBlock];
//   __shared__ double cache3[threadsPerBlock];
   extern __shared__ double array[];
   volatile double *x_load = array;
   volatile double *y_load = &x_load[N_dev];
   volatile double *z_load = &y_load[N_dev];
   
   if (cacheIndex == 0) Arrayin[blockIdx.x] = -1;
   else if (cacheIndex == 1) Arrayout[blockIdx.x] = -1;

//   bool whichTexture = true;
   double *temp = r; r = rn; rn = temp;
   volatile double ax_old = a[3*tid],//tex1Dfetch(tex_a, 3*tid),
         ay_old = a[3*tid+1],//tex1Dfetch(tex_a, 3*tid+1),
         az_old = a[3*tid+2];//tex1Dfetch(tex_a, 3*tid+2);
   double vx = v[3*tid],
          vy = v[3*tid+1],
          vz = v[3*tid+2];
   int old = cacheIndex + blockIdx.x*blockDim.x;
   x_load[old] = r[3*tid];//tex1Dfetch(tex_r, 3*tid),
   y_load[old] = r[3*tid+1],//tex1Dfetch(tex_r, 3*tid+1),
   z_load[old] = r[3*tid+2];//tex1Dfetch(tex_r, 3*tid+2);

for (int it = 0; it < l_dev; ++it)
{
//calculating a
   double ax_new = 0, ay_new = 0, az_new = 0; 
   cache2[cacheIndex] = acalc(&ax_new, &ay_new, &az_new,
                              x_load, y_load, z_load, r);

//calculating v
   vx = vx//tex1Dfetch(tex_v, 3*tid)
        + 0.5*step_dev*(ax_old + ax_new),
   vy = vy//tex1Dfetch(tex_v, 3*tid + 1)
        + 0.5*step_dev*(ay_old + ay_new),
   vz = vz//tex1Dfetch(tex_v, 3*tid + 2)
        + 0.5*step_dev*(az_old + az_new);
//   v[3*tid] = vx; v[3*tid+1] = vy; v[3*tid+2] = vz;

//measurement of temperature
   cache[cacheIndex] = static_cast<float>(vx*vx + vy*vy + vz*vz);
   __syncthreads();
   int i = blockDim.x/2;
   while (i != 0)
   {
      if (cacheIndex < i)
      {
         cache[cacheIndex] += cache[cacheIndex + i];
         cache2[cacheIndex] += cache2[(cacheIndex+i)];
      }
      __syncthreads();
      i /= 2;
   }

/*   if (cacheIndex == 0)
      K[it*gridDim.x + blockIdx.x] = cache[0]/2;
   else if (cacheIndex == 1)
      V[it*gridDim.x + blockIdx.x] = cache2[0]/2;*/

//calculating r
   double x_new = x_load[old]
                 + vx*step_dev
                 + 0.5*ax_new*step_dev*step_dev;
   if (x_new > S_dev) {x_new -= S_dev; }
   else if (x_new < 0) {x_new += S_dev; }
   double y_new = y_load[old]
                 + vy*step_dev
                 + 0.5*ay_new*step_dev*step_dev;
   if (y_new > S_dev) {y_new -= S_dev; }
   else if (y_new < 0) {y_new += S_dev; }
   double z_new = z_load[old]
                 + vz*step_dev
                 + 0.5*az_new*step_dev*step_dev;
   if (z_new > S_dev) {z_new -= S_dev; }
   else if (z_new < 0) {z_new += S_dev; }
   rn[3*tid] = x_new;
   rn[3*tid+1] = y_new;
   rn[3*tid+2] = z_new;

   temp = r; r = rn; rn = temp;
//   whichTexture = !whichTexture;
   ax_old = ax_new; ay_old = ay_new; az_old = az_new;
   x_load[old] = x_new; y_load[old] = y_new; z_load[old] = z_new;
   __gpu_sync(it, Arrayin, Arrayout,
              cache[0]/2, cache2[0]/2, Kb, Vb, K, V);
}
}

int setAtoms(int n, double b, double *r)
{
   S = n*b;
   if (S < 3*RC) {l = 0; return 1; }
   for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
         for (int x = 0; x < n; ++x)
   {
      r[3*(z*n*n + y*n + x)] = ((double) x)*b + 0.5*b;
      r[3*(z*n*n + y*n + x) + 1] = y*b + 0.5*b;
      r[3*(z*n*n + y*n + x) + 2] = z*b + 0.5*b;
   }
   return 0;
}
void Write(float *K, float *V, FILE *ofp)
{
   //sumall(obs);
   for(int i = 0; i < l; ++i)
   {
      double t = i*step;
      double a = K[i], b = V[i];
      fprintf(ofp, "%f %f %f %f\n", 
                   t, a, b, a+b);
   }

}



int main(int argc, char **argv)
{
   char *iname = argv[1], *oname = argv[2];
   double b = atof(argv[3]);
   step = atof(argv[4]);
   l = atoi(argv[5]);

   FILE *fp; fp = fopen(iname, "r");
   int n;
   fscanf(fp, "%i", &n);
   N = n*n*n;
   double *r = (double*) malloc(sizeof(double) * 3*N), 
         *v = (double*) malloc(sizeof(double) * 3*N);
   setAtoms(n, b, r);
   blocksPerGrid = N / threadsPerBlock;
   for(int i = 0; i < 3*N; ++i)
   {
      float vi;
      if (fscanf(fp, "%f", &vi) == EOF) l = 0;
      v[i] = static_cast<double>(vi);
   }
   fclose(fp);
   cudaEvent_t start, stop;
   HANDLE_ERROR( cudaEventCreate( &start ) );
   HANDLE_ERROR( cudaEventCreate( &stop ) );
   HANDLE_ERROR( cudaEventRecord( start, 0 ) );

   cudaMemcpyToSymbol(S_dev,
                   &S,
                   1*sizeof(double),
                   0,
                   cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(step_dev,
                   &step,
                   1*sizeof(double),
                   0,
                   cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(N_dev,
                   &N,
                   1*sizeof(int),
                   0,
                   cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(l_dev,
                   &l,
                   1*sizeof(int),
                   0,
                   cudaMemcpyHostToDevice);

   double *dev_r, *dev_v, *dev_a, *dev_rn /**dev_an*/;
   HANDLE_ERROR( cudaMalloc( (void**)&dev_r,
                              3*N*sizeof(double) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_v,
                              3*N*sizeof(double) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
                              3*N*sizeof(double) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_rn,
                              3*N*sizeof(double) ) );
//   HANDLE_ERROR( cudaBindTexture( NULL, tex_r,
//                                  dev_r, 3*N*sizeof(float)));
//   HANDLE_ERROR( cudaBindTexture( NULL, tex_rn,
//                                  dev_rn, 3*N*sizeof(float)));
   
   HANDLE_ERROR( cudaMemcpy( dev_r, r, 3*N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy( dev_v, v, 3*N*sizeof(double),
                              cudaMemcpyHostToDevice ) );
   
   float *dev_Kb, *dev_Vb;
   HANDLE_ERROR( cudaMalloc( (void**)&dev_Kb,
                              blocksPerGrid*sizeof(float) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_Vb,
                              blocksPerGrid*sizeof(float) ) );
   float *dev_K, *dev_V;
   HANDLE_ERROR( cudaMalloc( (void**)&dev_K,
                              l*sizeof(float) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_V,
                              l*sizeof(float) ) );
   int *dev_Arrayin, *dev_Arrayout;
   HANDLE_ERROR( cudaMalloc( (void**)&dev_Arrayin,
                              blocksPerGrid*sizeof(int) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_Arrayout,
                              blocksPerGrid*sizeof(int) ) );
   
//   cudaPrintfInit();
   InitVelocityVerlet<<<blocksPerGrid, threadsPerBlock>>>
                    (dev_r, dev_v, dev_a, dev_rn);

   VelocityVerlet<<<blocksPerGrid, threadsPerBlock, 3*N*sizeof(double)>>>
                (dev_r, dev_v, dev_a, dev_rn,
                 dev_Kb, dev_Vb, dev_K, dev_V,
                 dev_Arrayin, dev_Arrayout);
   
/*   cudaFree(dev_r);
   cudaFree(dev_v);
   cudaFree(dev_a);
   cudaFree(dev_rn);
   double *dev_K, *dev_V;
   HANDLE_ERROR( cudaMalloc( (void**)&dev_K,
                              l*sizeof(double) ) );
   HANDLE_ERROR( cudaMalloc( (void**)&dev_V,
                              l*sizeof(double) ) );
   Finish<<<l, blocksPerGrid>>>
        (dev_Kb, dev_Vb, dev_K, dev_V);*/
   float *K = (float*) malloc(sizeof(float) * l),
         *V = (float*) malloc(sizeof(float) * l);
   HANDLE_ERROR( cudaMemcpy( K, dev_K, l*sizeof(float),
                              cudaMemcpyDeviceToHost ) );
   HANDLE_ERROR( cudaMemcpy( V, dev_V, l*sizeof(float),
                              cudaMemcpyDeviceToHost ) );
//   cudaPrintfDisplay();
//   cudaUnbindTexture(tex_r);
//   cudaUnbindTexture(tex_rn);

   cudaFree(dev_r);
   cudaFree(dev_v);
   cudaFree(dev_a);
   cudaFree(dev_rn);

   cudaFree(dev_Kb);
   cudaFree(dev_Vb);
   cudaFree(dev_K);
   cudaFree(dev_V);
   cudaFree(dev_Arrayin);
   cudaFree(dev_Arrayout);

   HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
   HANDLE_ERROR( cudaEventSynchronize( stop ) );
   float   t;
   HANDLE_ERROR( cudaEventElapsedTime( &t, start, stop ) );
   HANDLE_ERROR( cudaEventDestroy(start));
   HANDLE_ERROR( cudaEventDestroy(stop));
   

   fp = fopen(oname, "w");
   fprintf(fp, "Beg. distance: %f; Steps: %d; Time to generate:  %3.1f ms\n", 
            b, l, t);
   Write(K, V, fp);
   fclose(fp);

   free(r);
   free(v);
   free(K);
   free(V);
//   cudaUnbindTexture(tex_r);
//   cudaUnbindTexture(tex_v);
//   cudaUnbindTexture(tex_a);
//   cudaFree(dev_an);
}
