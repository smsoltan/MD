//gcc -std=c99 -o randv randv.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
   srand (time(NULL));
   
   int n = atoi(argv[2]);
   int N = n*n*n;
   float vx = 0, vy = 0, vz = 0;
   float *v = malloc(sizeof(float) * 3*N);
   for(int i = 0; i < N; ++i)
   {
      v[3*i] = (float)rand()/(float)(RAND_MAX);
      v[3*i+1] = (float)rand()/(float)(RAND_MAX);
      v[3*i+2] = (float)rand()/(float)(RAND_MAX);
      vx += v[3*i]; vy += v[3*i+1]; vz += v[3*i+2];
   }
   vx = vx/N; vy = vy/N; vz = vz/N;
   for(int i = 0; i < N; ++i)
   {
      v[3*i] -= vx;
      v[3*i+1] -= vy;
      v[3*i+2] -= vz;
   }

   FILE * fp = fopen(argv[1], "w");

   fprintf(fp, "%d\n", n);
   for(int i = 0; i < 3*N; ++i)
      fprintf(fp, "%f\n", v[i]);

   fclose(fp);
}
