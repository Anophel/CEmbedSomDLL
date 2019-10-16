#pragma once

#include <assert.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <stdexcept>



using Coords2D = std::pair<float, float>;

#define USE_INTRINS // TODO invent a better ifdef
#ifdef USE_INTRINS
#include <xmmintrin.h>
#endif

#define log(x) std::cerr << x << std::endl

#define timed(label, cmd)                                                      \
	{                                                                      \
		auto __begin = std::chrono::high_resolution_clock::now ();     \
		cmd;                                                           \
		auto t =                                                       \
		  std::chrono::duration_cast<std::chrono::nanoseconds> (       \
		    std::chrono::high_resolution_clock::now () - __begin)      \
		    .count ();                                                 \
		log (label << " done in " << t * 1e-9f << "s");                \
	}

// this helps with debugging floating-point overflows and similar nastiness,
// uncomment if needed.
//#define DEBUG_CRASH_ON_FPE

#ifdef DEBUG_CRASH_ON_FPE
#include <fenv.h>
#endif

using namespace std;

// some small numbers first!
static const float min_boost = 0.00001f; // lower limit for the parameter

// this is added before normalizing the distances
static const float zero_avoidance = 0.00000000001f;

// a tiny epsilon for preventing singularities
static const float koho_gravity = 0.00001f;


struct dist_id {
  float dist;
  size_t id;
};

union U {
  __m128 v;    // SSE 4 x float vector
  float a[4];  // scalar array of 4 floats
};

inline static constexpr float vectorGetByIndex(__m128 V, unsigned int i)
{
  U u{0ULL};

  assert(i <= 3);
  u.v = V;
  return u.a[i];
}

inline static constexpr float sqrf(float n)
{
  return n * n;
}

// euclidean distance
inline static float sqrcos(const float* p1, const float* p2, size_t dim)
{
	float h = 0, da = 0, db=0;
	const float *p1e = p1 + dim;
	for (; p1 < p1e; ++p1, ++p2) {
		h += *p1 * *p2;
		da += *p1 * *p1;
		db += *p2 * *p2;
	}
	// icc behaves terribly here
	return 1-(h<0.0000001?0:h/sqrt(da*db));
}
// euclidean distance
inline static float sqreucl(const float* p1, const float* p2, size_t dim)
{
#ifndef USE_INTRINS
  float sqdist = 0;
  for (size_t i = 0; i < dim; ++i) {
    float tmp = p1[i] - p2[i];
    sqdist += tmp * tmp;
  }
  return sqdist;
#else
  const float *p1e = p1 + dim, *p1ie = p1e - 3;

  __m128 s = _mm_setzero_ps();
  for (; p1 < p1ie; p1 += 4, p2 += 4) {
    __m128 tmp = _mm_sub_ps(_mm_loadu_ps(p1), _mm_loadu_ps(p2));
    s = _mm_add_ps(_mm_mul_ps(tmp, tmp), s);
  }
  float sqdist = 0;
  for (; p1 < p1e; ++p1, ++p2) {
    float tmp = *p1 - *p2;
    sqdist += tmp * tmp;
  }
  // icc behaves terribly here
  return sqdist + vectorGetByIndex(s, 0) + vectorGetByIndex(s, 1) + vectorGetByIndex(s, 2) + vectorGetByIndex(s, 3);

#endif
}

inline static void hswap(dist_id& a, dist_id& b)
{
  dist_id c = a;
  a = b;
  b = c;
}

inline static void heap_down(dist_id* heap, size_t start, size_t lim)
{
  for (;;) {
    size_t L = 2 * start + 1;
    size_t R = L + 1;
    if (R < lim) {
      float dl = heap[L].dist;
      float dr = heap[R].dist;

      if (dl > dr) {
        if (heap[start].dist >= dl) break;
        hswap(heap[L], heap[start]);
        start = L;
      }
      else {
        if (heap[start].dist >= dr) break;
        hswap(heap[R], heap[start]);
        start = R;
      }
    }
    else if (L < lim) {
      if (heap[start].dist < heap[L].dist)
        hswap(heap[L], heap[start]);
      break; // exit safely!
    }
    else
      break;
  }
}


struct Position_Mapping {
	Coords2D coords;
	size_t clust_id;
	//float dist_cent;
};

class CEmbedSom
{
public:
  

public:
  CEmbedSom() = delete;
  CEmbedSom(const std::string& dataFilepath, size_t dSize);


  std::vector<Position_Mapping> GetImageEmbeddings(const std::vector<size_t>& imageIds, const std::vector<float>& distribution);

  std::vector<std::pair<Position_Mapping, size_t>> GetCollectionRepresentants(const std::vector<size_t>& imageIds, const std::vector<float>& distribution, size_t xdim = 16, size_t ydim = 16, size_t rlen = 15);

private:
  void embedsom(
    unsigned e_dimension,
    size_t n,
    size_t dim,
    float boost,
    size_t topn,
    float adjust,
    size_t ncodes,
    const vector<float>& points,   // n * dim-size vector
    const vector<float>& koho,     // ncodes * dim-size vector
    const vector<float>& emcoords, // ncodes * e_dimension-size vector
    vector<float>& embedding);

  template<typename RNG>
  void som(size_t n,
    size_t k,
    size_t dim,
    size_t rlen,
    const std::vector<float>& points,
    std::vector<float>& koho,
    const std::vector<float>& nhbrdist,
    float alphasA[2],
    float radiiA[2],
    float alphasB[2],
    float radiiB[2],
    RNG& rng,
	std::vector<float> probabs);
  /* this serves for classification into small clusters */
  void mapPointsToKohos(size_t n,
    size_t k,
    size_t dim,
    const std::vector<float>& points,
    const std::vector<float>& koho,
	std::vector<std::pair<size_t, float>>& mapping);

  void print_help();

  template<class T>
  bool read(const std::string& s, T& out);

private:
  std::string _dataFilepath;
	size_t datasetSize;
};
