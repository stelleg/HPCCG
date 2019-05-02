// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <random>

#include "taco.h"

using namespace taco;
using namespace std;

Tensor<double> HPCGA(int nx, int ny, int nz){
  cout << "generating matrix" << endl;
  int n = nx*ny*nz;
  Tensor<double> A({n, n}, {Dense, Sparse});
  for(int x=0; x<nx; x++)
    for(int y=0; y<ny; y++)
      for(int z=0; z<nz; z++){
        int row = z*nx*ny + y*nx + x;
        for(int sx=-1; sx<=1 && x+sx>=0 && x+sx<nx; sx++)
          for(int sy=-1; sy<=1 && y+sy>=0 && y+sy<ny; sy++)
            for(int sz=-1; sz<=1 && z+sz>=0 && z+sz<nz; sz++){
              int col = row + sz*nx*ny + sy*nx + sx;
              if(row == col) A.insert({row, col}, 27.0);
              else A.insert({row, col}, -1.0);
            }
      }
  A.pack(); 
  cout << "done" << endl;
  return A;
}

int main(int argc, char* argv[]) {
  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  int nx = argc > 1 ? atoi(argv[1]) : 4;
  int ny = argc > 2 ? atoi(argv[2]) : 4;
  int nz = argc > 3 ? atoi(argv[3]) : 4;
  int n = nx * ny * nz;
  int max_iter = 150; 

  // Solving Ax = b for x
  Tensor<double> A = HPCGA(nx, ny, nz); 
  Tensor<double> xexact({n}, {Dense}), x({n}, {Dense}), xold({n}, {Dense}); 
  Tensor<double> p({n}, {Dense}), pold({n}, {Dense}), Ap({n}, {Dense}); 
  Tensor<double> b({n}, {Dense}), r({n}, {Dense}), rold({n}, {Dense}); 
  Tensor<double> beta(0), rtrans(0), oldrtrans(0), alpha(0); 

  for(int i=0; i<n; i++){
    xexact.insert({i}, 1.0);
    b.insert({i}, 27.0); // todo
    p.insert({i}, 0.0); 
    r.insert({i}, 0.0); 
    xold.insert({i}, 0.0); 
    pold.insert({i}, 0.0); 
    rold.insert({i}, 0.0); 
  }

  IndexVar i, j;  
  Ap(i) = A(i,j) * p(j); 
  rtrans = r(i) * r(i); 
  oldrtrans = rtrans(); 
  beta = rtrans() / oldrtrans();
  p(i) = r(i) + beta() * pold(i); 
  x(i) = xold(i) + alpha() * p(i); 
  alpha = p(i) * Ap(i); 
  r(i) = Ap(i) + alpha() * rold(i); 
  xold(i) = x(i); 
  pold(i) = p(i); 
  rold(i) = r(i); 

  // Ordering important, computed in order
  auto kernels = {Ap, rtrans, oldrtrans, beta, p, x, alpha, r, xold, pold, rold}; 

  for(auto k : kernels) k.compile(); 

  for(int iter = 0; iter < max_iter; iter++){ 
    for(auto k : kernels) k.compute(); 
  }

  std::cout << x << std::endl;
}
