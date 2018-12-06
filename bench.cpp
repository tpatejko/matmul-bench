#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <chrono>
#include <mkldnn.h>
#include <mkl_cblas.h>

struct mat_descriptor {
  int height;
  int width;
  int stride{0};
  int batch_size{0};
  bool transposed;
};

mat_descriptor create_matrix_descriptor(const std::vector<int>& dims, bool transposed) {
  mat_descriptor desc;
  
  if (dims.size() == 2) {
    desc.height = dims[0];
    desc.width = dims[1];
  } else {
    desc.batch_size = 1;

    for (size_t i = 0; i < dims.size() - 2; ++i) {
      desc.batch_size *= dims[i];
    }

    desc.width = *dims.rbegin();
    desc.height = *(dims.rbegin() + 1);
    
    desc.stride = desc.height * desc.width;
  }

  if (transposed) {
    std::swap(desc.height, desc.width);
  }

  desc.transposed = transposed;

  return desc;
}

template<typename T>
using duration_t = std::chrono::duration<double, T>;

duration_t<std::micro> calculate_total_time(const std::vector<duration_t<std::micro>>& measures) {
  return std::accumulate(std::begin(measures),
			 std::end(measures),
			 duration_t<std::micro>::zero(),
			 std::plus<duration_t<std::micro>>());
}

duration_t<std::micro> calculate_average(const std::vector<duration_t<std::micro>>& measures) {
  auto total = calculate_total_time(measures);
  return total / measures.size();
}

class single {
  void mkldnn_matmul(const mat_descriptor& dims_a, const float* a,
		     const mat_descriptor& dims_b, const float* b,
		     float* c,
		     float alpha, float beta) {
    char a_transposed = dims_a.transposed ? 'T' : 'N';
    char b_transposed = dims_b.transposed ? 'T' : 'N';

    auto m = dims_a.height;
    auto n = dims_b.width;
    auto k = dims_a.width;

    auto lda = dims_a.transposed ? m : k;
    auto ldb = dims_b.transposed ? k : n;
    auto ldc = n;
  
    mkldnn_sgemm(&b_transposed, &a_transposed,
		 &n, &m, &k,
		 &alpha,
		 b, &ldb,
		 a, &lda,
		 &beta,
		 c, &ldc);
  }
  
  void mkl_matmul(const mat_descriptor& dims_a, const float* a,
		  const mat_descriptor& dims_b, const float* b,
		  float* c,
		  float alpha, float beta)
  {
    CBLAS_TRANSPOSE a_transposed = dims_a.transposed ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE b_transposed = dims_b.transposed ? CblasTrans : CblasNoTrans;

    auto m = dims_a.height;
    auto n = dims_b.width;
    auto k = dims_a.width;
  
    auto lda = dims_a.transposed ? m : k;
    auto ldb = dims_b.transposed ? k : n;
    auto ldc = n;
  
    cblas_sgemm(CblasRowMajor, a_transposed, b_transposed,
		m, n, k,
		alpha,
		a, lda,
		b, ldb, beta,
		c, ldc);
  }

 public:
  void operator()() {
    constexpr int m = 2048;
    constexpr int k = 2048;
    constexpr int n = 64;
    constexpr int warm_up = 10;
    constexpr int iters = 10000;
  
    auto dims_a = create_matrix_descriptor({m, k}, false);
    auto dims_b = create_matrix_descriptor({k, n}, false);

    std::unique_ptr<float[]> ptr_a{new float[dims_a.height*dims_a.width]};
    std::unique_ptr<float[]> ptr_b{new float[dims_b.height*dims_b.width]};

    std::random_device rd;
    std::mt19937 e{rd()};
    std::uniform_real_distribution<float> dist{0, 1};

    for (size_t i = 0; i < dims_a.height*dims_a.width; i++) {
      ptr_a[i] = dist(e);
    }

    for (size_t i = 0; i < dims_b.height*dims_b.width; i++) {
      ptr_b[i] = dist(e);
    }

    std::unique_ptr<float[]> mkldnn_ptr_c{new float[dims_a.height*dims_b.width]};
    std::unique_ptr<float[]> mkl_ptr_c{new float[dims_a.height*dims_b.width]};

    for (int i = 0; i < warm_up; i++) {
      mkldnn_matmul(dims_a, ptr_a.get(),
		    dims_b, ptr_b.get(),
		    mkldnn_ptr_c.get(),
		    1.0, 0.0);

      mkl_matmul(dims_a, ptr_a.get(),
		 dims_b, ptr_b.get(),
		 mkl_ptr_c.get(),
		 1.0, 0.0);
    }
    
    std::vector<std::chrono::duration<double, std::micro>> mkldnn_durs;
    std::vector<std::chrono::duration<double, std::micro>> mkl_durs;
    
    for (int i = 0; i < iters; i++) {
      auto s = std::chrono::high_resolution_clock::now();
      mkldnn_matmul(dims_a, ptr_a.get(),
		    dims_b, ptr_b.get(),
		    mkldnn_ptr_c.get(),
		    1.0, 0.0);
      auto e = std::chrono::high_resolution_clock::now();
      mkldnn_durs.push_back(e-s);
    }

    for (int i = 0; i < iters; i++) {
      auto s = std::chrono::high_resolution_clock::now();
      mkl_matmul(dims_a, ptr_a.get(),
		 dims_b, ptr_b.get(),
		 mkl_ptr_c.get(),
		 1.0, 0.0);
      auto e = std::chrono::high_resolution_clock::now();
      mkl_durs.push_back(e-s);
    }
    
    std::cout << "Single execution\n";
    std::cout << "Avg. per iteration " << iters << " iterations\n";
    std::cout << "MKLDNN: " << calculate_average(mkldnn_durs).count() << " us" << std::endl;
    std::cout << "MKL: "    << calculate_average(mkl_durs).count()    << " us" << std::endl;  

    for (int i = 0; i < dims_a.height*dims_b.width; i++) {
      if (mkldnn_ptr_c[i] != mkl_ptr_c[i])
	std::cout << "mkldnn " << mkldnn_ptr_c[i] << " "
		  << "mkl " << mkl_ptr_c[i] << std::endl;
    }
  }
};

class batched {
  void mkldnn_matmul(const mat_descriptor& dims_a, const float* a,
		     const mat_descriptor& dims_b, const float* b,
		     float* c,
		     float alpha, float beta) {
    char a_transposed = dims_a.transposed ? 'T' : 'N';
    char b_transposed = dims_b.transposed ? 'T' : 'N';

    auto m = dims_a.height;
    auto n = dims_b.width;
    auto k = dims_a.width;

    auto lda = dims_a.transposed ? m : k;
    auto ldb = dims_b.transposed ? k : n;
    auto ldc = n;

    auto stride_a = dims_a.height * dims_a.width;
    auto stride_b = dims_b.height * dims_b.width;
  
    for (int i = 0; i < dims_a.batch_size; i++) {
      auto ptr_a = a + i * stride_a;
      auto ptr_b = b + i * stride_b;
      auto ptr_c = c + i * m * n;
    
      mkldnn_sgemm(&b_transposed, &a_transposed,
		   &n, &m, &k,
		   &alpha,
		   ptr_b, &ldb,
		   ptr_a, &lda,
		   &beta,
		   ptr_c, &ldc);
    }
  }

  void mkl_matmul(const mat_descriptor& dims_a, const float* a,
		  const mat_descriptor& dims_b, const float* b,
		  float* c,
		  float alpha, float beta) {
    CBLAS_TRANSPOSE a_transposed = dims_a.transposed ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE b_transposed = dims_b.transposed ? CblasTrans : CblasNoTrans;

    auto m = dims_a.height;
    auto n = dims_b.width;
    auto k = dims_a.width;
  
    auto lda = dims_a.transposed ? m : k;
    auto ldb = dims_b.transposed ? k : n;
    auto ldc = n;

    std::vector<const float*> a_arrays(dims_a.batch_size);
    std::vector<const float*> b_arrays(dims_a.batch_size);
    std::vector<float*> c_arrays(dims_a.batch_size);
    
    for (int i = 0; i < dims_a.batch_size; i++) {
      a_arrays[i] = a + i * dims_a.stride;
      b_arrays[i] = b + i * dims_b.stride;
      c_arrays[i] = c + i * m * n;
    }
  
    cblas_sgemm_batch(CblasRowMajor, &a_transposed, &b_transposed,
		      &m, &n, &k,
		      &alpha,
		      a_arrays.data(), &lda,
		      b_arrays.data(), &ldb, &beta,
		      c_arrays.data(), &ldc,
		      1, &dims_a.batch_size);
  }

 public:
  void operator()() {
    constexpr int batch_size = 32;
    constexpr int head = 8;
    constexpr int m = 256;
    constexpr int k = 256;
    constexpr int n = 64;
    constexpr int warm_up = 10;
    constexpr int iters = 1000;
  
    auto dims_a = create_matrix_descriptor({batch_size, head, m, k}, false);
    auto dims_b = create_matrix_descriptor({batch_size, head, k, n}, false);

    std::unique_ptr<float[]> ptr_a{new float[dims_a.batch_size*dims_a.height*dims_a.width]};
    std::unique_ptr<float[]> ptr_b{new float[dims_b.batch_size*dims_b.height*dims_b.width]};

    std::random_device rd;
    std::mt19937 e{rd()};
    std::uniform_real_distribution<float> dist{0, 1};

    for (size_t i = 0; i < dims_a.batch_size*dims_a.height*dims_a.width; i++) {
      ptr_a[i] = dist(e);
    }

    for (size_t i = 0; i < dims_b.batch_size*dims_b.height*dims_b.width; i++) {
      ptr_b[i] = dist(e);
    }

    std::unique_ptr<float[]> mkldnn_ptr_c{new float[dims_a.batch_size*dims_a.height*dims_b.width]};
    std::unique_ptr<float[]> mkl_ptr_c{new float[dims_a.batch_size*dims_a.height*dims_b.width]};

    for (int i = 0; i < warm_up; i++) {
      mkldnn_matmul(dims_a, ptr_a.get(),
		    dims_b, ptr_b.get(),
		    mkldnn_ptr_c.get(),
		    1.0, 0.0);

      mkl_matmul(dims_a, ptr_a.get(),
		 dims_b, ptr_b.get(),
		 mkl_ptr_c.get(),
		 1.0, 0.0);
    }

    std::vector<std::chrono::duration<double, std::micro>> mkldnn_durs;
    std::vector<std::chrono::duration<double, std::micro>> mkl_durs;

    for (int i = 0; i < iters; i++) {
      auto s = std::chrono::high_resolution_clock::now();
      mkldnn_matmul(dims_a, ptr_a.get(),
		    dims_b, ptr_b.get(),
		    mkldnn_ptr_c.get(),
		    1.0, 0.0);
      auto e = std::chrono::high_resolution_clock::now();
      mkldnn_durs.push_back(e-s);
    }
    

    for (int i = 0; i < iters; i++) {
      auto s = std::chrono::high_resolution_clock::now();
      mkl_matmul(dims_a, ptr_a.get(),
		 dims_b, ptr_b.get(),
		 mkl_ptr_c.get(),
		 1.0, 0.0);
      auto e = std::chrono::high_resolution_clock::now();
      mkl_durs.push_back(e-s);
    }

    std::cout << "Batched execution\n";
    std::cout << "Avg. per iteration " << iters << " iterations\n";
    std::cout << "MKLDNN: " << calculate_average(mkldnn_durs).count()   << " us" << std::endl;
    std::cout << "MKL: "    << calculate_average(mkl_durs).count()      << " us" << std::endl;
    
    mkldnn_matmul(dims_a, ptr_a.get(),
		  dims_b, ptr_b.get(),
		  mkldnn_ptr_c.get(),
		  1.0, 0.0);

    mkl_matmul(dims_a, ptr_a.get(),
	       dims_b, ptr_b.get(),
	       mkl_ptr_c.get(),
	       1.0, 0.0);

    for (int i = 0; i < dims_a.batch_size*dims_a.height*dims_b.width; i++) {
      if (mkldnn_ptr_c[i] != mkl_ptr_c[i])
	std::cout << "mkldnn " << mkldnn_ptr_c[i] << " "
		  << "mkl " << mkl_ptr_c[i] << std::endl;
    }
  }
};

int main() {
  single s;
  s();
  
  batched b;
  b();
  
  return 0;
}
