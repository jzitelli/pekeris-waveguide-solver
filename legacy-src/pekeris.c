#include <stdio.h>
#include <math.h>
#include "gsl/gsl_roots.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf_bessel.h"

double *eigenValues = 0;
double K1, K2, 
       RHO1, RHO2, Z_0, H,
       C1, C2, K_SQRD, K2_SQRD,
       C1_SQRD, C2_SQRD,
       M, C1_C2_SQRD;

gsl_root_fsolver   *BrentSolver;
gsl_function       F;

int nMax, n;

double branchStart;
double R, Z;

// The eigenvalue equation to be solved by Brent solver
// DEPENDS on global variable n
double my_f(double x, void *params) {
  return sqrt(K_SQRD-x)*H - atan(sqrt(x/K_SQRD - C1_C2_SQRD)/M*K1/sqrt(K_SQRD-x)) - (n-0.5)*M_PI;
}


// Initiates the Brent solver from GSL
void initSolver() {
  F.function = my_f;
  F.params = 0;

  BrentSolver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

  return;
}


// Calculate all of the eigenvalues for the discrete spectrum,
// store them in the global array.
void calcEigenVals() {
  double x, x_lo, x_hi;
  int status, iter, maxIter;

  maxIter = 500;

  nMax = (int) (K1/M_PI*H*sqrt(1.0 - C1*C1/(C2*C2)) + 0.5);
  
  if (eigenValues != 0)
    free(eigenValues);
  eigenValues = (double*)malloc(nMax*sizeof(double));

  for (n = 1; n <= nMax; ++n) {
    // Solve for the nth eigenvalue k_n^2
    x_lo = K2_SQRD + 0.00001;
    x_hi = K_SQRD - 0.00001;
    gsl_root_fsolver_set (BrentSolver, &F, x_lo, x_hi);

    iter = 0;
    do {
      iter++;
      gsl_root_fsolver_iterate(BrentSolver);
      x = gsl_root_fsolver_root(BrentSolver);
      status = gsl_root_test_interval(x_lo, x_hi, 0, 1e-14);
    } while (status == GSL_CONTINUE && iter < maxIter);

    eigenValues[n-1] = x;
  }

  return;
}


// Sets the parameters for evaluating pekeris field with this C module,
// outputting nVals (number of discrete eigenvals).
void setpekerisparams_(const double *omega,
                       const double *rho1, const double *rho2,
                       const double *z_0,  const double *h,
                       const double *c1,   const double *c2,
                       int *nVals)  {

  C1 = *c1;
  C2 = *c2;
  K1 = *omega/C1;
  K2 = *omega/C2;
  RHO1 = *rho1;
  RHO2 = *rho2;
  Z_0 = *z_0;
  H = *h;

  K_SQRD = K1*K1;
  K2_SQRD = K2*K2;
  C1_SQRD = C1*C1;
  C2_SQRD = C2*C2;
  M = RHO2/RHO1;
  C1_C2_SQRD = C1_SQRD/C2_SQRD;

  branchStart = sqrt(K_SQRD - K2*K2);

  initSolver();
  calcEigenVals();

  *nVals = nMax;

  return;
}


// Load the values into the fortran module data (it should have been allocated)
void geteigenvalues_(double *eigenvals) {
  int i;

  for(i = 0; i < nMax; ++i) {
    eigenvals[i] = eigenValues[i];
  }

  return;
}


/*
int main() {
  double w = 200.0;
  C1 = 1500.0;
  C2 = 1800.0;
  K1 = w/C1;
  K2 = w/C2;
  RHO1 = 1000.0;
  RHO2 = 1800.0;
  Z_0 = 30.0;
  H = 150.0;

  K_SQRD = K1*K1;
  C1_SQRD = C1*C1;
  C2_SQRD = C2*C2;
  M = RHO2/RHO1;
  C1_C2_SQRD = C1_SQRD/C2_SQRD;

  initSolver();
  calcEigenVals();

  for (int i = 0; i < nMax; ++i) {
    std::cout << eigenValues[i] << std::endl;
  }

  return 0;
}
*/

