// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <random>
#include <taco.h>
#include <chrono>

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
                if(row == col) A(row, col) = 27.0;
                else A(row, col) = -1.0;
              }
            }
          }
        }
      }
  A.pack(); 
  //write("A.tns", A); 
  cout << "done" << endl;
  return A;
}

double *val(Tensor <double> t){
  return static_cast<double*>(t.getStorage().getValues().getData());
}
int main(int argc, char* argv[]) {
  //std::default_random_engine gen(0);
  //std::uniform_real_distribution<double> unif(0.0, 1.0);
  int nx = argc > 1 ? atoi(argv[1]) : 4;
  int ny = argc > 2 ? atoi(argv[2]) : 4;
  int nz = argc > 3 ? atoi(argv[3]) : 4;
  int n = nx * ny * nz;
  int max_iter = 152; 
  double tolerance = 0.0; 

  // Solving Ax = b for x
  Tensor<double> A = HPCGA(nx, ny, nz); 
  Tensor<double> xexact("xexact", {n}, {Dense}), x("x", {n}, {Dense}), xold("xold", {n}, {Dense}); 
  Tensor<double> p("p", {n}, {Dense}), pold("pold", {n}, {Dense}), Ap("Ap", {n}, {Dense}); 
  Tensor<double> b("b", {n}, {Dense}), r("r", {n}, {Dense}), rold("rold", {n}, {Dense}), tmp1("tmp1", {n}, {Dense}); 
  Tensor<double> beta(1), rtrans(1), oldrtrans(1), alpha(0), tmp0(0); 

  for(int i=0; i<n; i++){
    xexact.insert({i}, 1.0);
    x.insert({i}, 0.0); 
  }

  auto initials = {xexact, p, x}; 
  for(auto t : initials) t.pack(); 

  IndexVar i, j;  
  b(i) = A(i,j) * xexact(j); 
  p(i) = x(i); 
  Ap(i) = A(i,j) * p(j); 
  r(i) = b(i) - Ap(i); 
  rtrans = r(i) * r(i); 

  auto start = std::chrono::high_resolution_clock::now(); 
  for(int iter = 1; iter <= max_iter && sqrt(*val(rtrans)) > tolerance; iter++){ 
    if(iter == 3) start = std::chrono::high_resolution_clock::now(); 
    cout << "iter: " << iter << endl; 
    cout << " residual: " << sqrt(*val(rtrans)) << endl; 
    cout << " beta: " << *val(beta) << endl; 
    if(iter == 1) p(i) = r(i); 
    else {
      oldrtrans = rtrans(); 
      rtrans = r(i) * r(i); 
      beta = rtrans() / oldrtrans();
      pold(i) = p(i); 
      p(i) = r(i) + beta() * pold(i); 
    }
    Ap(i) = A(i,j) * p(j); 
    tmp0 = p(i) * Ap(i); 
    alpha() = rtrans() / tmp0();  
    //cout << " alpha: " << *val(alpha) << endl; 
    //cout << " rtrans: " << *val(rtrans) << endl;
    //cout << p << endl; 
    //cout << Ap << endl; 
    xold(i) = x(i); 
    x(i) = xold(i) + alpha() * p(i); 
    //cout << x << endl; 
    rold(i) = r(i); 
    tmp1(i) = alpha() * Ap(i); 
    r(i) = rold(i) - tmp1(i); 
    //cout << r << endl; 
  }
  auto finish = std::chrono::high_resolution_clock::now(); 
  std::chrono::duration<double> elapsed = finish - start; 
  cout << "final rtrans " << *val(rtrans) << endl; 
  cout << "Total time: " << elapsed.count() << "s\n"; 
}
