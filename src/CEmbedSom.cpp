
#include "CEmbedSom.h"

CEmbedSom::CEmbedSom(const std::string& dataFilepath) :
    _dataFilepath(dataFilepath)
  {}


  std::vector<std::pair<Coords2D, size_t>> CEmbedSom::GetImageEmbeddings(const std::vector<size_t>& imageIds)
  {
    // parameters
    size_t dims = 50;
    size_t n = 0;
    size_t xdim = 16;
    size_t ydim = 16;
    size_t rlen = 15;
    float adjust = 1;
    float smooth = 0;
    size_t k = 10;
    int seed = 0x31337;
    float alphasA[2] = { 0.05, 0.01 };
    float radiiA[2] = { 8, 0 };
    float negRadius = 0;
    float negAlpha = 0;

    size_t datasize = 20000;
    //std::string _dataFilepath = ; //TODO tenhle soubor to musi mit po ruce, cestu mozno libovolne menit

    std::vector<float>origdata;
    origdata.resize(datasize*dims);
    std::ifstream inputFS(_dataFilepath.c_str(), std::ios::binary);
    
    // If input stream is valid and opened successfully
    if (!inputFS || !inputFS.is_open())
    {
      throw std::runtime_error("Opening input file failed!");
    }

    inputFS.read((char*)origdata.data(), sizeof(float)*datasize*dims);

    //TODO: tady je soubor nactenej a je mozny to poustet opakovane pro kazdy data

    std::vector<float>points;

    { //TODO: tady si misto nacitani ze vstupu dej vlastni data. Cisla odpovidaji cislum obrazku 0-19999

      for (auto&& imageId : imageIds)
      {
        auto od = origdata.begin() + imageId * dims;
        points.insert(points.end(), od, od + dims);

        ++n;
      }
    }

    std::mt19937 rng(seed);

    if (!(dims && n && xdim && ydim && rlen && k)) {
      log("weird arguments");
      throw std::runtime_error("weird arguments");
    }

    const size_t kohos = xdim * ydim;
    std::vector<float> koho;
    koho.resize(kohos * dims, 0);

    std::vector<float> somcoords, nhbrdist;
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
          sqrtf(sqreucl(somcoords.data() + 2 * i,
            somcoords.data() + 2 * j,
            2));
      }

    float alphasB[2] = { -alphasA[0] * negAlpha, -alphasA[1] * negAlpha };
    float radiiB[2] = { radiiA[0] * negRadius, radiiA[1] * negRadius };

    log("computing SOM...");

    timed("som",
      som(n,
        kohos,
        dims,
        rlen,
        points,
        koho,
        nhbrdist,
        alphasA,
        radiiA,
        alphasB,
        radiiB,
        rng));

    log("mapping...");
    std::vector<size_t> mapping;
    mapping.resize(n);
    timed(
      "mapping",
      mapPointsToKohos(n, kohos, dims, points, koho, mapping));

    //TODO better emcoords

    log("embedding...");
    std::vector<float> embed;
    embed.resize(2 * n);
    timed("embedding",
      embedsom(2,
        n,
        dims,
        powf((1 + sqrtf(5)) / 2, smooth - 2),
        k,
        adjust,
        kohos,
        points,
        koho,
        somcoords,
        embed));


    // Create result data structure
    std::vector<std::pair<Coords2D, size_t>> result;
    result.reserve(embed.size() / 2);

    for (size_t i = 0; i < n; ++i)
    {
      result.emplace_back(Coords2D{ embed[2 * i + 0] , embed[2 * i + 1] }, mapping[i]);
    }

    return result;
  }

  void CEmbedSom::embedsom(
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
    vector<float>& embedding)      // n * e_dimension-size vector
  {
    size_t i, j, k;

#ifdef DEBUG_CRASH_ON_FPE
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif

    if (topn > ncodes) topn = ncodes;
    if (boost < min_boost) boost = min_boost;

    vector<dist_id> dists;
    dists.resize(topn);

    float mtx[12]; // only 6 in case of pe_dimension=2, but who cares

    const float* point = points.data();
    for (size_t ptid = 0; ptid < n; ++ptid, point += dim) {

      // heap-knn
      for (i = 0; i < topn; ++i) {
        dists[i].dist =
          sqreucl(point, koho.data() + i * dim, dim);
        dists[i].id = i;
      }

      for (i = 0; i < topn; ++i)
        heap_down(dists.data(), topn - i - 1, topn);

      for (i = topn; i < ncodes; ++i) {
        float s = sqreucl(point, koho.data() + i * dim, dim);
        if (dists[0].dist < s) continue;
        dists[0].dist = s;
        dists[0].id = i;
        heap_down(dists.data(), 0, topn);
      }

      // heapsort the result
      for (i = topn - 1; i > 0; --i) {
        hswap(dists[0], dists[i]);
        heap_down(dists.data(), 0, i);
      }

      // compute scores
      float sum = 0, ssum = 0, min = dists[0].dist;
      for (i = 0; i < topn; ++i) {
        dists[i].dist = sqrtf(dists[i].dist);
        sum += dists[i].dist / (i + 1);
        ssum += 1 / float(i + 1);
        if (dists[i].dist < min) min = dists[i].dist;
      }

      sum = -ssum / (zero_avoidance + sum * boost);

      for (i = 0; i < topn; ++i)
        dists[i].dist = expf((dists[i].dist - min) * sum);

      // prepare the matrix for 2- or 3-variable linear eqn
      if (e_dimension == 2)
        for (i = 0; i < 6; ++i)
          mtx[i] = 0; // it's stored by columns!
      if (e_dimension == 3)
        for (i = 0; i < 12; ++i) mtx[i] = 0;

      for (i = 0; i < topn; ++i) {
        // add a really tiny influence of the point to prevent
        // singularities
        size_t idx = dists[i].id;
        float ix, iy, iz;
        if (e_dimension == 2) {
          ix = emcoords[2 * idx + 0];
          iy = emcoords[2 * idx + 1];
        }
        if (e_dimension == 3) {
          ix = emcoords[3 * idx + 0];
          iy = emcoords[3 * idx + 1];
          iz = emcoords[3 * idx + 2];
        }
        float pi = dists[i].dist;
        float gs = koho_gravity * dists[i].dist;
        if (e_dimension == 2) {
          mtx[0] += gs;
          mtx[3] += gs;
          mtx[4] += gs * ix;
          mtx[5] += gs * iy;
        }
        if (e_dimension == 3) {
          mtx[0] += gs;
          mtx[4] += gs;
          mtx[8] += gs;
          mtx[9] += gs * ix;
          mtx[10] += gs * iy;
          mtx[11] += gs * iz;
        }

        for (j = i + 1; j < topn; ++j) {

          size_t jdx = dists[j].id;
          float jx, jy, jz;
          if (e_dimension == 2) {
            jx = emcoords[2 * jdx + 0];
            jy = emcoords[2 * jdx + 1];
          }
          if (e_dimension == 3) {
            jx = emcoords[3 * jdx + 0];
            jy = emcoords[3 * jdx + 1];
            jz = emcoords[3 * jdx + 2];
          }
          float pj = dists[j].dist;

          float scalar = 0, sqdist = 0;
          for (k = 0; k < dim; ++k) {
            float tmp = koho[k + dim * jdx] -
              koho[k + dim * idx];
            sqdist += tmp * tmp;
            scalar += tmp * (point[k] -
              koho[k + dim * idx]);
          }

          if (scalar != 0) {
            if (sqdist == 0)
              continue;
            else
              scalar /= sqdist;
          }

          if (e_dimension == 2) {
            const float hx = jx - ix;
            const float hy = jy - iy;
            const float hpxy = hx * hx + hy * hy;
            const float ihpxy = 1 / hpxy;

            const float s =
              pi * pj / powf(hpxy, adjust);

            const float diag = s * hx * hy * ihpxy;
            const float rhsc =
              s * (scalar +
              (hx * ix + hy * iy) * ihpxy);

            mtx[0] += s * hx * hx * ihpxy;
            mtx[1] += diag;
            mtx[2] += diag;
            mtx[3] += s * hy * hy * ihpxy;
            mtx[4] += hx * rhsc;
            mtx[5] += hy * rhsc;
          }

          if (e_dimension == 3) {
            const float hx = jx - ix;
            const float hy = jy - iy;
            const float hz = jz - iz;
            const float hpxyz =
              hx * hx + hy * hy + hz * hz;
            const float s =
              pi * pj / powf(hpxyz, adjust);
            const float ihpxyz = 1 / hpxyz;
            const float sihpxyz = s * ihpxyz;

            const float rhsc =
              s * (scalar +
              (hx * ix + hy * iy + hz * iz) *
                ihpxyz);

            mtx[0] += sihpxyz * hx * hx;
            mtx[1] += sihpxyz * hx * hy;
            mtx[2] += sihpxyz * hx * hz;
            mtx[3] += sihpxyz * hy * hx;
            mtx[4] += sihpxyz * hy * hy;
            mtx[5] += sihpxyz * hy * hz;
            mtx[6] += sihpxyz * hz * hx;
            mtx[7] += sihpxyz * hz * hy;
            mtx[8] += sihpxyz * hz * hz;
            mtx[9] += hx * rhsc;
            mtx[10] += hy * rhsc;
            mtx[11] += hz * rhsc;
          }
        }
      }

      if (e_dimension == 2) {
        // cramer
        float det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
        // output is stored R-style by columns
        embedding[0 + ptid * 2] =
          (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
        embedding[1 + ptid * 2] =
          (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
      }
      if (e_dimension == 3) {
        float det =
          mtx[0] * mtx[4] * mtx[8] + mtx[1] * mtx[5] * mtx[6] +
          mtx[2] * mtx[3] * mtx[7] - mtx[0] * mtx[5] * mtx[7] -
          mtx[1] * mtx[3] * mtx[8] - mtx[2] * mtx[4] * mtx[6];
        embedding[0 + ptid * 3] = (mtx[9] * mtx[4] * mtx[8] +
          mtx[10] * mtx[5] * mtx[6] +
          mtx[11] * mtx[3] * mtx[7] -
          mtx[9] * mtx[5] * mtx[7] -
          mtx[10] * mtx[3] * mtx[8] -
          mtx[11] * mtx[4] * mtx[6]) /
          det;
        embedding[1 + ptid * 3] = (mtx[0] * mtx[10] * mtx[8] +
          mtx[1] * mtx[11] * mtx[6] +
          mtx[2] * mtx[9] * mtx[7] -
          mtx[0] * mtx[11] * mtx[7] -
          mtx[1] * mtx[9] * mtx[8] -
          mtx[2] * mtx[10] * mtx[6]) /
          det;
        embedding[2 + ptid * 3] = (mtx[0] * mtx[4] * mtx[11] +
          mtx[1] * mtx[5] * mtx[9] +
          mtx[2] * mtx[3] * mtx[10] -
          mtx[0] * mtx[5] * mtx[10] -
          mtx[1] * mtx[3] * mtx[11] -
          mtx[2] * mtx[4] * mtx[9]) /
          det;
      }
    }

#ifdef DEBUG_CRASH_ON_FPE
    fedisableexcept(FE_INVALID | FE_OVERFLOW);
#endif
  }

  template<typename RNG>
  void CEmbedSom::som(size_t n,
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
    RNG& rng)
  {
    std::uniform_int_distribution<size_t> random(0, n - 1);

    size_t niter = rlen * n;

    float thresholdA0 = radiiA[0], alphaA0 = alphasA[0],
      thresholdADiff = radiiA[1] - radiiA[0],
      alphaADiff = alphasA[1] - alphasA[0], thresholdB0 = radiiB[0],
      alphaB0 = alphasB[0], thresholdBDiff = radiiB[1] - radiiB[0],
      alphaBDiff = alphasB[1] - alphasB[0];

    for (size_t iter = 0; iter < niter; ++iter) {
      size_t point = random(rng);
      float riter = iter / (float)niter;

      size_t nearest = 0;
      {
        float nearestd = sqreucl(
          points.data() + dim * point, koho.data(), dim);
        for (size_t i = 1; i < k; ++i) {
          float tmp =
            sqreucl(points.data() + dim * point,
              koho.data() + dim * i,
              dim);
          if (tmp < nearestd) {
            nearest = i;
            nearestd = tmp;
          }
        }
      }

      float thresholdA = thresholdA0 + riter * thresholdADiff,
        thresholdB = thresholdB0 + riter * thresholdBDiff,
        alphaA = alphaA0 + riter * alphaADiff,
        alphaB = alphaB0 + riter * alphaBDiff;


      for (size_t i = 0; i < k; ++i) {
        float d = nhbrdist[i + k * nearest];

        float alpha;

        if (d > thresholdA) {
          if (d > thresholdB) continue;
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

  /* this serves for classification into small clusters */
  void CEmbedSom::mapPointsToKohos(size_t n,
    size_t k,
    size_t dim,
    const std::vector<float>& points,
    const std::vector<float>& koho,
    std::vector<size_t>& mapping)
  {
    for (size_t point = 0; point < n; ++point) {
      size_t nearest = 0;
      float nearestd =
        sqreucl(points.data() + dim * point, koho.data(), dim);
      for (size_t i = 1; i < k; ++i) {
        float tmp = sqreucl(points.data() + dim * point,
          koho.data() + dim * i,
          dim);
        if (tmp < nearestd) {
          nearest = i;
          nearestd = tmp;
        }
      }

      mapping[point] = nearest;
    }
  }

  void CEmbedSom::print_help()
  {
    std::cout << "See source code for command line options." << std::endl;
  }

  template<class T>
  bool CEmbedSom::read(const std::string& s, T& out)
  {
    std::stringstream ss(s);
    ss >> out;
    return !ss.bad() && ss.eof();
  }

