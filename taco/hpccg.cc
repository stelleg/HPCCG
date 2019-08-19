// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <random>
#include <taco.h>

using namespace taco;
using namespace std;

Tensor<double> HPCGA(int nx, int ny, int nz){
  cout << "generating matrix" << endl;
  int n = nx*ny*nz;
  Tensor<double> A("A", {n, n}, {Dense, Sparse});
  for(int x=0; x<nx; x++)
    for(int y=0; y<ny; y++)
      for(int z=0; z<nz; z++){
        int row = z*nx*ny + y*nx + x;
        for(int sx=-1; sx<=1; sx++){
          for(int sy=-1; sy<=1; sy++){
            for(int sz=-1; sz<=1; sz++){
              if(x+sx>=0 && x+sx<nx && y+sy>=0 && y+sy<ny && z+sz>=0 && z+sz<nz){
                int col = row + sz*nx*ny + sy*nx + sx;
                if(row == col) A.insert({row, col}, 27.0);
                else A.insert({row, col}, -1.0);
              }
            }
          }
        }
      }
  A.pack(); 
  write("A.tns", A); 
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
  Tensor<double> xexact("xexact", {n}, {Dense}), x("x", {n}, {Dense}), xold("xold", {n}, {Dense}); 
  Tensor<double> p("p", {n}, {Dense}), pold("pold", {n}, {Dense}), Ap("Ap", {n}, {Dense}); 
  Tensor<double> b("b", {n}, {Dense}), r("r", {n}, {Dense}), rold("rold", {n}, {Dense}); 
  Tensor<double> beta(1), rtrans(1), oldrtrans(1), alpha(0); 

  cout << beta << endl;  

  for(int i=0; i<n; i++){
    xexact.insert({i}, 1.0);
    p.insert({i}, 1.0); 
    r.insert({i}, 1.0); 
    xold.insert({i}, 1.0); 
    pold.insert({i}, 0.0); 
    rold.insert({i}, 1.0); 
  }

  auto initials = {xexact, p, r, xold, pold, rold}; 
  for(auto t : initials) t.pack(); 

  IndexVar i, j;  
  b(i) = A(i,j) * xexact(j); 
  b.compile(); b.assemble(); b.compute(); 
   
  Ap(i) = A(i,j) * p(j); 
  rtrans = r(i) * r(i); 
  beta = rtrans() / oldrtrans();
  p(i) = r(i) + beta() * pold(i); 
  x(i) = xold(i) + alpha() * p(i); 
  alpha = p(i) * Ap(i); 
  r(i) = Ap(i) + alpha() * rold(i); 
  oldrtrans = rtrans(); 
  xold(i) = x(i); 
  pold(i) = p(i); 
  rold(i) = r(i); 

  // Ordering important, computed in order
  auto kernels = {Ap, rtrans, beta, p, x, alpha, r, oldrtrans, xold, pold, rold}; 

  for(auto k : kernels) {
    k.compile(); 
  }

  for(int iter = 0; iter < max_iter; iter++){ 
    cout << "iter " << iter << endl; 
    cout << "alpha " << alpha << endl; 
    cout << x << endl; 
    for(auto k : kernels){
      //cout << k << endl; 
      k.assemble(); 
      k.compute();
    }
  }

  std::cout << x << std::endl;
  write("matrix.mtx", A); 
}
