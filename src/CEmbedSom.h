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


#define DIMS_SIZE 128

#define EMPTY_CLUSTER std::numeric_limits<size_t>::max()
#define TOTAL_SOM_ITER 100000

using Coords2D = std::pair<double, double>;

struct PointWithId {
	double first;
	double second;
	int clustId;
	int imageId;
	double sigma;
};

//#define USE_INTRINS // TODO invent a better ifdef
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
static const double min_boost = 0.00001f; // lower limit for the parameter

// this is added before normalizing the distances
static const double zero_avoidance = 0.00000000001f;

// a tiny epsilon for preventing singularities
static const double koho_gravity = 0.00001f;


struct dist_id {
  double dist;
  size_t id;
};

#if 0
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
#endif

inline static constexpr float sqrf(float n)
{
  return n * n;
}

//manhattan distance
inline static double
manh(const double *p1, const double *p2, const size_t dim)
{
	double dist = 0;
	for (size_t i = 0; i < dim; ++i) {
		dist += abs(p1[i] - p2[i]);
	}
	return dist;
}

// euclidean distance
inline static double cosdist(const double* p1, const double* p2, size_t dim)
{
	double h = 0, da = 0, db=0;
	const double *p1e = p1 + dim;
	for (; p1 < p1e; ++p1, ++p2) {
		h += *p1 * *p2;
		da += *p1 * *p1;
		db += *p2 * *p2;
	}
	// icc behaves terribly here
	return 1 - (h / sqrt(da*db));
}
// euclidean distance
inline static double sqreucl(const double* p1, const double* p2, size_t dim)
{
#ifndef USE_INTRINS
	double sqdist = 0;
  for (size_t i = 0; i < dim; ++i) {
		double tmp = p1[i] - p2[i];
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
      double dl = heap[L].dist;
      double dr = heap[R].dist;

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
	double sigma;
};

struct CosDist_t
{
	double operator()(const double* p1, const double* p2, size_t dim)
	{
		return cosdist(p1, p2, dim);
	}
};

struct Manh_t
{
	double operator()(const double* p1, const double* p2, size_t dim)
	{
		return manh(p1, p2, dim);
	}
};

class CEmbedSom
{
	std::string _dataFilepath;
	const std::vector<double> _origdata;
	size_t datasetSize;
	size_t dims;

public:
  CEmbedSom() = delete;
  CEmbedSom(double* data, size_t dSize, size_t dims);

#if 0
  std::vector<Position_Mapping> GetImageEmbeddings(const std::vector<size_t>& imageIds, const std::vector<double>& distribution);
#endif


	/* this serves for classification into small clusters */
	template <typename Dist>
	inline void mapPointsToKohos(size_t n,
		size_t k,
		size_t dim,
		const std::vector<double>& points,
		const std::vector<double>& koho,
		std::vector<std::pair<size_t, double>>& mapping,
		std::vector<double>& sigmas,
		Dist dist)
	{
		std::vector<size_t> mappingCount;
		mappingCount.resize(k);
		for (size_t point = 0; point < n; ++point) {
			size_t nearest = 0;
			double nearestd =
				dist(points.data() + dim * point, koho.data(), dim);
			for (size_t i = 1; i < k; ++i) {
				double tmp = dist(points.data() + dim * point,
					koho.data() + dim * i,
					dim);
				if (tmp < nearestd) {
					nearest = i;
					nearestd = tmp;
				}
			}


			mapping[point] = std::make_pair(nearest, nearestd);
			sigmas[nearest] += nearestd;
			mappingCount[nearest]++;
		}

		// Dividing all sums to get mean and thats the sigma
		for (size_t i = 0; i < k; ++i) {
			sigmas[i] /= (double)mappingCount[i];
		}
	}


	template<typename RNG, typename Dist>
	inline void som(size_t n,
		size_t k,
		size_t dim,
		size_t rlen,
		const std::vector<double>& points,
		std::vector<double>& koho,
		const std::vector<double>& nhbrdist,
		double alphasA[2],
		double radiiA[2],
		double alphasB[2],
		double radiiB[2],
		const std::vector<double> &scores,
		RNG& rng,
		Dist dist)
	{
		std::discrete_distribution<size_t> random(scores.begin(), scores.end());

		size_t niter = TOTAL_SOM_ITER;

		double thresholdA0 = radiiA[0], alphaA0 = alphasA[0],
			thresholdADiff = radiiA[1] - radiiA[0],
			alphaADiff = alphasA[1] - alphasA[0], thresholdB0 = radiiB[0],
			alphaB0 = alphasB[0], thresholdBDiff = radiiB[1] - radiiB[0],
			alphaBDiff = alphasB[1] - alphasB[0];

		for (size_t iter = 0; iter < niter; ++iter) {
			size_t point = random(rng);
			double riter = iter / double(niter);

			size_t nearest = 0;
			{
				double nearestd = dist(
					points.data() + dim * point, koho.data(), dim);
				for (size_t i = 1; i < k; ++i) {
					double tmp =
						dist(points.data() + dim * point,
							koho.data() + dim * i,
							dim);
					if (tmp < nearestd) {
						nearest = i;
						nearestd = tmp;
					}
				}
			}

			double thresholdA = thresholdA0 + riter * thresholdADiff,
				thresholdB = thresholdB0 + riter * thresholdBDiff,
				alphaA = alphaA0 + riter * alphaADiff,
				alphaB = alphaB0 + riter * alphaBDiff;

			for (size_t i = 0; i < k; ++i) {
				double d = nhbrdist[i + k * nearest];

				double alpha;

				if (d > thresholdA) {
					if (d > thresholdB)
						continue;
					alpha = alphaB;
				}
				else
					alpha = alphaA;

				for (size_t j = 0; j < dim; ++j)
					koho[j + i * dim] +=
					alpha *
					(points[j + point * dim] - koho[j + i * dim]);
			}
		}
	}

	template <typename Dist>
	inline std::vector<PointWithId> GetCollectionRepresentants(const std::vector<size_t>& imageIds,
		const std::vector<double>& probabilities,
		Dist dist,
		bool forceRepre,
		bool mostProbab,
		size_t xdim,
		size_t ydim,
		size_t rlen)
	{
		// parameters
		size_t n = imageIds.size();
		double adjust = 1;
		double smooth = 0;
		size_t k = 10;
		int seed = 0x31337;
		double alphasA[2] = { 0.05, 0.01 };
		double radiiA[2] = { 8, 0 };
		double negRadius = 0;
		double negAlpha = 0;


		size_t datasize = datasetSize;

		std::mt19937 rng(seed);

		if (!(dims && n && xdim && ydim && k)) {
			log("weird arguments");
			log(dims);
			log(n);
			log(xdim);
			log(ydim);
			log(k);
			throw std::runtime_error("weird arguments");
		}

		const size_t kohos = xdim * ydim;
		std::vector<double> koho;
		koho.resize(kohos * dims, 0);

		std::vector<double> somcoords, nhbrdist;
		somcoords.resize(kohos * 2);
		for (size_t x = 0; x < xdim; ++x)
			for (size_t y = 0; y < ydim; ++y) {
				somcoords[2 * (x + xdim * y) + 0] = x;
				somcoords[2 * (x + xdim * y) + 1] = y;
			}

		nhbrdist.resize(kohos * kohos);
		for (size_t i = 0; i < kohos; ++i)
			for (size_t j = 0; j < kohos; ++j) {
				nhbrdist[i + kohos * j] =
					sqrt(sqreucl(somcoords.data() + 2 * i,
						somcoords.data() + 2 * j,
						2));
			}

		double alphasB[2] = { -alphasA[0] * negAlpha, -alphasA[1] * negAlpha };
		double radiiB[2] = { radiiA[0] * negRadius, radiiA[1] * negRadius };

		log("computing SOM...");

		timed("som",
			som(n,
				kohos,
				dims,
				rlen,
				_origdata,
				koho,
				nhbrdist,
				alphasA,
				radiiA,
				alphasB,
				radiiB,
				probabilities,
				rng,
				dist););

		log("mapping...");
		std::vector<std::pair<size_t, double>> mapping;
		mapping.resize(n);
		std::vector<double> sigmas;
		sigmas.resize(kohos);
		timed(
			"mapping",
			mapPointsToKohos(n, kohos, dims, _origdata, koho, mapping, sigmas, dist));

		std::vector<size_t> repres(kohos, EMPTY_CLUSTER);

		if (mostProbab) {
			std::vector<double> probabRepres;
			probabRepres.resize(kohos);
			for (size_t i = 0; i < kohos; ++i) {
				repres[i] = 0;
				probabRepres[i] = 0;
			}

			for (size_t point = 0; point < n; ++point) {
				auto koMap = mapping[point];
				if (probabRepres[koMap.first] < probabilities[point]) {
					repres[koMap.first] = point;
					probabRepres[koMap.first] = probabilities[point];
				}
			}
		}
		else {
			log("Creating representants");
			std::vector<std::vector<size_t>> clusters;
			std::vector<std::vector<double>> clustersDist;
			clusters.resize(kohos);
			clustersDist.resize(kohos);

			// prepare probabilities
			size_t point = 0;
			for (auto&& m : mapping) {
				clusters[m.first].push_back(point);
				clustersDist[m.first].push_back(probabilities[point++]);
			}

			// compute distribution
			for (size_t clust = 0; clust < clustersDist.size(); ++clust) {
				for (size_t cdi = 1; cdi < clustersDist[clust].size(); ++cdi) {
					clustersDist[clust][cdi] += clustersDist[clust][cdi - 1];
				}
			}

			// create representants
			for (size_t clust = 0; clust < clustersDist.size(); ++clust) {
				if (clustersDist[clust].size() > 0) {
					std::uniform_real_distribution<double> random(0.0, clustersDist[clust][clustersDist[clust].size() - 1]);
					double target = random(rng);

					for (point = 0; point < clustersDist[clust].size() && clustersDist[clust][point] < target; ++point) {}

					repres[clust] = clusters[clust][point];
				}
				else {
					log("Empty clust: " << clust);
					if (forceRepre)
					{
						double minDist = dist(_origdata.data(), koho.data() + clust * dims, dims);
						size_t minPoint = 0;
						for (size_t point = 0; point < n; ++point) {
							if (std::find(repres.begin(), repres.end(), point) == repres.end())
							{
								double tmp = dist(_origdata.data() + point * dims, koho.data() + clust * dims, dims);
								if (tmp < minDist) {
									minDist = tmp;
									minPoint = point;
								}
							}
						}
						repres[clust] = minPoint;
						log("Forced representant " << minPoint);
					}
					else
					{
						repres[clust] = 0;
					}
				}
			}
			log("Representants created");
		}

		//TODO better emcoords

#if 0
		log("layouting...");

		std::vector<double> emcoords;
		emcoords = somcoords;

#if DO_LAYOUT_SOM
		timed("layouting",
			emcoords_som(2, kohos, dims, somcoords, koho, emcoords));
#else //do MST
		timed("layouting",
			emcoords_mst(2, kohos, dims, somcoords, koho, emcoords));
#endif

		log("embedding...");
		std::vector<double> embed;
		embed.resize(2 * kohos);
		timed("embedding",
			embedsom(2,
				kohos,
				dims,
				pow((1 + sqrt(5)) / 2, smooth - 2),
				k,
				adjust,
				kohos,
				_origdata,
				koho,
				emcoords,
				embed));
#endif

		// Create result data structure
		std::vector<PointWithId> result;
		result.reserve(kohos);

		for (size_t i = 0; i < kohos; ++i)
		{
			PointWithId crs;
			crs.first = 0;
			crs.second = 0;
			crs.imageId = imageIds[repres[i]];
			crs.clustId = i;
			crs.sigma = sigmas[i];
			result.emplace_back(crs);
		}

		return result;
	}

	inline std::vector<PointWithId> GetCollectionRepresentantsCos(const std::vector<size_t>& imageIds,
		const std::vector<double>& distribution, bool forceRepre, bool mostProbab = false,
		size_t xdim = 16, size_t ydim = 16, size_t rlen = 15)
	{
		CosDist_t d;
		return GetCollectionRepresentants<CosDist_t>(imageIds, distribution, d, forceRepre, mostProbab, xdim, ydim, rlen);
	}

	inline std::vector<PointWithId> GetCollectionRepresentantsManh(const std::vector<size_t>& imageIds,
		const std::vector<double>& distribution, bool forceRepre, bool mostProbab = false,
		size_t xdim = 16, size_t ydim = 16, size_t rlen = 15)
	{
		Manh_t d;
		return GetCollectionRepresentants<Manh_t>(imageIds, distribution, d, forceRepre, mostProbab, xdim, ydim, rlen);
	}

private:
  void embedsom(
    unsigned e_dimension,
    size_t n,
    size_t dim,
    double boost,
    size_t topn,
    double adjust,
    size_t ncodes,
    const vector<double>& points,   // n * dim-size vector
    const vector<double>& koho,     // ncodes * dim-size vector
    const vector<double>& emcoords, // ncodes * e_dimension-size vector
    vector<double>& embedding);

  void print_help();

  template<class T>
  bool read(const std::string& s, T& out);
};

