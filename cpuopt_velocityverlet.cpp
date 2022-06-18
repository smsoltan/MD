//g++ -std=c++0x -o cpuopt_velocityverlet cpuopt_velocityverlet.cpp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <list>
//#include <vector_types.h>

#define SIGMA 1
#define RC 2.5*SIGMA
#define RM 3.3*SIGMA
#define EPS 0.01
#define ZERO 4*EPS*(pow(SIGMA/RC, 12) - pow(SIGMA/RC, 6))

//double OFF = 4*EPS*(pow(2.5, -12) - powf(2.5, -6))

typedef std::list<int> List;
typedef std::list<List> Lists;

int l, N, m;
double S;
double step;
double interval;

//This thing stores output
struct Observable
{
   double *macro;
   int o;
   Observable()
   {
      o = 0;
      macro = (double*) malloc(sizeof(double) * l);
   }
   ~Observable()
   {
      free(macro);
   }
   void detect(double measurement)
   {
      macro[o++] = measurement;
   }
};

//this function constructs a Verlet list and calculates accelerations. It uses third Newton's law, obviously.
//requires zeroed 'a' list
void aVerletListConstruct
     (double *r, double *a, Lists *lists)
{
   int i = 0;
   for(auto it = lists->begin(); it != lists->end(); ++it)
   {
      a[3*i] = 0; a[3*i+1] = 0; a[3*i+2] = 0;
      double xo = r[3*i], yo = r[3*i + 1], zo = r[3*i + 2],
            ax = 0, ay = 0, az = 0;
      it->clear();
      for(int j = 0; j < i; ++j)
      {
            double x = xo - r[3*j],
                  y = yo - r[3*j+1],
                  z = zo - r[3*j+2];
            if (x > RM) x-= S; else if (x < -RM) x+= S;
            if (y > RM) y-= S; else if (y < -RM) y+= S;
            if (z > RM) z-= S; else if (z < -RM) z+= S;
            double r2 = x*x + y*y + z*z;
            if (r2 < RM*RM)
            {
               it->push_back(j);
               if (r2 < RC*RC)
               {
                  double R = pow(SIGMA*SIGMA/r2, 3);
                  double part = 4*EPS*R*R, ULJ = part - 4*EPS*R,
                        da = 6*(part + ULJ)/r2,
                        dax = x*da,
                        day = y*da,
                        daz = z*da;
                  ax += dax;
                  ay += day;
                  az += daz;
                  a[3*j] -= dax;
                  a[3*j+1] -= day;
                  a[3*j+2] -= daz;
               }
            }
      }
      a[3*i] = ax; a[3*i+1] = ay; a[3*i+2] = az;
   ++i;
   }
}

// In addition to the above, this function also saves potential energy
//requires zeroed 'a' list
double aVVerletListConstruct
     (double *r, double *a, Lists *lists)
{
   static const double zero = ZERO;
   double sum = 0;
   int i = 0;
   for(auto it = lists->begin(); it != lists->end(); ++it)
   {
      a[3*i] = 0; a[3*i+1] = 0; a[3*i+2] = 0;
      double xo = r[3*i], yo = r[3*i + 1], zo = r[3*i + 2],
            ax = 0, ay = 0, az = 0;
      it->clear();
      for(int j = 0; j < i; ++j)
      {
            double x = xo - r[3*j],
                  y = yo - r[3*j+1],
                  z = zo - r[3*j+2];
            if (x > RM) x-= S; else if (x < -RM) x+= S;
            if (y > RM) y-= S; else if (y < -RM) y+= S;
            if (z > RM) z-= S; else if (z < -RM) z+= S;
            double r2 = x*x + y*y + z*z;
            if (r2 < RM*RM)
            {
               it->push_back(j);
               if (r2 < RC*RC)
               {
                  double R = pow(SIGMA*SIGMA/r2, 3);
                  double part = 4*EPS*R*R, ULJ = part - 4*EPS*R,
                        da = 6*(part + ULJ)/r2,
                        dax = x*da,
                        day = y*da,
                        daz = z*da;
                  ax += dax;
                  ay += day;
                  az += daz;
                  a[3*j] -= dax;
                  a[3*j+1] -= day;
                  a[3*j+2] -= daz;
                  sum += ULJ - zero;
               }
            }
      }
      a[3*i] = ax; a[3*i+1] = ay; a[3*i+2] = az;
   ++i;
   }
   return sum;
}

//this function calculates accelerations of particles based on mutual interactions. It uses Verlet lists and Newton's third law, obviously.
//requires a zeroed 'a' list:
void a_calc(double *r, double *a, Lists *lists)
{
   int i = 0;
   for(auto it = lists->begin(); it != lists->end(); ++it)
   {
      double xo = r[3*i], yo = r[3*i + 1], zo = r[3*i + 2],
            ax = 0, ay = 0, az = 0;
      for(auto jt = it->begin(); jt != it->end(); ++jt)
      {
         int j = *jt;
         double x = xo - r[3*j],
               y = yo - r[3*j+1],
               z = zo - r[3*j+2];
         if (x > RC) x-= S; else if (x < -RC) x+= S;
         if (y > RC) y-= S; else if (y < -RC) y+= S;
         if (z > RC) z-= S; else if (z < -RC) z+= S;
         double r2 = x*x + y*y + z*z;
         if (r2 < RC*RC)
         {
            double R = pow(SIGMA*SIGMA/r2, 3);
            double part = 4*EPS*R*R, ULJ = part - 4*EPS*R,
                  da = 6*(part + ULJ)/r2,
                  dax = x*da,
                  day = y*da,
                  daz = z*da;
            ax += dax;
            ay += day;
            az += daz;
            a[3*j] -= dax;
            a[3*j+1] -= day;
            a[3*j+2] -= daz;
         }
      }
   a[3*i] = ax; a[3*i+1] = ay; a[3*i+2] = az;
   ++i;
   }
}

// In addition to the above, this function also saves potential energy
//requires a zeroed 'a' list:
double aV_calc(double *r, double *a, Lists *lists)
{
//   for(int i = 0; i < 3*N; ++i) a[i] = 0;
   static const double zero = ZERO;
   int i = 0;
   double sum = 0;
   for(auto it = lists->begin(); it != lists->end(); ++it)
   {
      a[3*i] = 0; a[3*i+1] = 0; a[3*i+2] = 0;
      double xo = r[3*i], yo = r[3*i + 1], zo = r[3*i + 2],
            ax = 0, ay = 0, az = 0;
      for(auto jt = it->begin(); jt != it->end(); ++jt)
      {
         int j = *jt;
         double x = xo - r[3*j],
               y = yo - r[3*j+1],
               z = zo - r[3*j+2];
         if (x > RC) x-= S; else if (x < -RC) x+= S;
         if (y > RC) y-= S; else if (y < -RC) y+= S;
         if (z > RC) z-= S; else if (z < -RC) z+= S;
         double r2 = x*x + y*y + z*z;
         if (r2 < RC*RC)
         {
            double R = pow(SIGMA*SIGMA/r2, 3);
            double part = 4*EPS*R*R, ULJ = part - 4*EPS*R,
                  da = 6*(part + ULJ)/r2,
                  dax = x*da,
                  day = y*da,
                  daz = z*da;
            ax += dax;
            ay += day;
            az += daz;
            a[3*j] -= dax;
            a[3*j+1] -= day;
            a[3*j+2] -= daz;
            sum += ULJ - zero;
         }
      }
      a[3*i] = ax; a[3*i+1] = ay; a[3*i+2] = az;
      ++i;
   }
   return sum;
}


//This function initiates the procedure and calculates first step. It saves the kinetic energy
void InitVelocityVerlet
     (double *r, double *v, double *a, double *a_old, Lists *lists, Observable *K)
{
   double sum = 0;
   for(int i = 0; i < N; ++i)
      lists->push_back(List());
   aVerletListConstruct(r, a, lists);
   for(int i = 0; i < 3*N; ++i)
   {
      a_old[i] = a[i];
      double r_new = r[i] + v[i]*step + 0.5*a[i]*step*step;
      if (r_new > S) {r_new -= S; }
      else if (r_new < 0) {r_new += S; }
      r[i] = r_new;
      double vi = v[i];
      sum += vi*vi/2;
   }
   K->detect(sum);
}

//Like the above, but it also saves potential energy as well
void InitVelocityVerlet
     (double *r, double *v, double *a, double *a_old, Lists *lists,
      Observable *K, Observable *V)
{
   double sum = 0;
   for(int i = 0; i < N; ++i)
      lists->push_back(List());
   V->detect(aVVerletListConstruct(r, a, lists));
   for(int i = 0; i < 3*N; ++i)
   {
      a_old[i] = a[i];
      double r_new = r[i] + v[i]*step + 0.5*a[i]*step*step;
      if (r_new > S) {r_new -= S; }
      else if (r_new < 0) {r_new += S; }
      r[i] = r_new;
      double vi = v[i];
      sum += vi*vi/2;
   }
   K->detect(sum);
}

//This function performs single step of a simulation and saves kinetic energy. It makes decisions concerning when the verlet list needs to be redone.
void VelocityVerlet
     (double *r, double *v, double *a, double *a_old, Lists *lists, Observable *K)
{
   static int o = 0, little_l = 1;
   double sum = 0, threev = 0;
   static double kum;
   
   if (o++ < little_l) a_calc(r, a, lists);
   else {aVerletListConstruct(r, a, lists); o = 0; }
   for(int i = 0; i < 3*N; ++i)
   {
      double ai = a[i], vi = v[i] + 0.5*step*(ai + a_old[i]),
            r_new = r[i] + vi*step + 0.5*ai*step*step;
      v[i] = vi;
      a_old[i] = ai;
      if (r_new > S) {r_new -= S; }
      else if (r_new < 0) {r_new += S; }
      r[i] = r_new;
      threev += vi*vi;
      if(i%3 == 2)
      {sum += threev/2; 
       threev = sqrt(threev);
       if (kum < threev) {kum = threev; little_l = 1; }
       threev = 0;}
   }
   if(little_l == 1) little_l = static_cast<int>((RM-RC)/(4*kum*step));
   K->detect(sum);
}

//Like the above, but it also saves potential energy as well
void VelocityVerlet
     (double *r, double *v, double *a, double *a_old, Lists *lists, Observable *K, Observable *V)
{
   static int o = 0, little_l = 1;
   double sum, threev = 0;
   static double kum;
   if (o++ < little_l) sum = aV_calc(r, a, lists);
   else {sum = aVVerletListConstruct(r, a, lists); o = 0; kum = 0;}
   V->detect(sum);
   sum = 0;
   for(int i = 0; i < 3*N; ++i)
   {
      double ai = a[i], vi = v[i] + 0.5*step*(ai + a_old[i]),
            r_new = r[i] + vi*step + 0.5*ai*step*step;
      v[i] = vi;
//      a_old[i] = ai;
      if (r_new > S) {r_new -= S; }
      else if (r_new < 0) {r_new += S; }
      r[i] = r_new;
      threev += vi*vi;
      if(i%3 == 2)
      {sum += threev/2; 
       threev = sqrt(threev);
       if (kum < threev) {kum = threev; little_l = 1;}
       threev = 0; }
   }
   if(little_l == 1) little_l = static_cast<int>((RM-RC)/(2*kum*step));
//   printf("%d\n", little_l);
   K->detect(sum);
}

//This sets atoms on its initial positions on a grid
int setAtoms(int n, double b, double *r)
{
   S = n*b;
   if (S < 3*RC) {l = 0; return 0; }
   for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
         for (int x = 0; x < n; ++x)
   {
      r[3*(z*n*n + y*n + x)] = ((double) x)*b + 0.5*b;
      r[3*(z*n*n + y*n + x) + 1] = y*b + 0.5*b;
      r[3*(z*n*n + y*n + x) + 2] = z*b + 0.5*b;
   }
   return 1;
}

//Writes the obtained values of one observable
void Write(Observable *obs, FILE *ofp)
{
   for(int i = 0; i < l; ++i)
   {
      double t = i*step;
      fprintf(ofp, "%f %f\n", t, (obs->macro[i]));
   }

}

//Writes the obtained values of both observables, and a total energy as well
void Write(Observable *K, Observable *V, FILE *ofp)
{
   for(int i = 0; i < l; ++i)
   {
      double t = i*step;
      double a = K->macro[i], b = V->macro[i];
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
   double *r = (double*) malloc((sizeof *r) * 3*N), 
         *v = (double*) malloc((sizeof *v) * 3*N), 
         *a = (double*) malloc((sizeof *a) * 3*N);
   setAtoms(n, b, r);
   for(int i = 0; i < 3*N; ++i)
   {
      float vi;
      if (fscanf(fp, "%f", &vi) == EOF) l = 0;
      v[i] = static_cast<double>(vi);
   }
   fclose(fp);

   Observable temperature, potential;

   clock_t t;
   t = clock();
   Lists lists;
   double *a_old = (double*) malloc((sizeof *a_old) * 3*N);
   InitVelocityVerlet(r, v, a, a_old, &lists, &temperature, &potential);
   for(int i = 0; i < l; ++i)
   {
      VelocityVerlet(r, v, a, a_old, &lists, &temperature, &potential);
      double *temp = a_old; a_old = a; a = temp;
   }
   free(a_old);
   t = clock() - t;

   fp = fopen(oname, "w");
   fprintf(fp, "Beg. distance: %f; Steps: %d; It took me %d clicks (%f seconds).\n",b,l,(int)t,((double)t)/CLOCKS_PER_SEC);
   Write(&temperature, &potential, fp);
   fclose(fp);

   free(r);
   free(v);
   free(a);
}
