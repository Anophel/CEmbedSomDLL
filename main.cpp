
#include <iostream>
#include <string>

#include "CEmbedSom.h"

#include <cassert>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <random>
#include <vector>

using namespace std;

struct Point {
	float first;
	float second;
	int clustId;
	//float distCent;
};

struct PointWithId {
	float first;
	float second;
	int clustId;
	//float distCent;
	int imageId;
};

const std::string path{ R"ddd(C:\Users\User\source\repos\VideoBrowser\CEmbedSomDLL\data\images-ordered-pca.bin)ddd" };
CEmbedSom* ces;

// ugly AF
#define FRAND (rand() / (float)RAND_MAX)

vector<size_t>
sample_w(const vector<float>& ws, size_t k)
{
	size_t n = ws.size();

	assert(n >= 2);
	assert(k < n);

	size_t branches = n - 1;
	vector<float> tree(branches + n, 0);
	float sum = 0;
	for (size_t i = 0; i < n; ++i)
		sum += tree[branches + i] = ws[i];

	auto upd = [&tree, branches, n](size_t i) {
		const size_t l = 2 * i + 1;
		const size_t r = 2 * i + 2;
		if (i < branches + n)
			tree[i] = ((l < branches + n) ? tree[l] : 0) +
			((r < branches + n) ? tree[r] : 0);
	};

	auto updb = [&tree, branches, n, upd](size_t i) {
		for (;;) {
			upd(i);
			if (i)
				i = (i - 1) / 2;
			else
				break;
		}
	};

	for (size_t i = branches; i > 0; --i)
		upd(i - 1);

	vector<size_t> res(k, 0);

	for (auto& rei : res) {
		float x = FRAND * tree[0];
		size_t i = 0;
		for (;;) {
			const size_t l = 2 * i + 1;
			const size_t r = 2 * i + 2;

			//cout << "at i " << i << " x: " << x << " in tree: " << tree[i] << endl;

			if (i >= branches) break;
			if (r < branches + n && x >= tree[l]) {
				x -= tree[l];
				i = r;
			}
			else
				i = l;
		}

		tree[i] = 0;
		updb(i);
		rei = i - branches;
	}

	return res;
}

int main()
{
  //const std::string path{ R"ddd(C:\Users\User\source\repos\VideoBrowser\CEmbedSomDLL\data\images-ordered-pca.bin)ddd" };

  //CEmbedSom ces{ path };
  ces = new CEmbedSom(path, 20000);
  std::vector<size_t> input;

  for (size_t i{ 0ULL }; i < 20000; ++i)
  {
    input.push_back(i);
  }
  std::vector<float> probabs;
  probabs.push_back(1.0 / 20000.0);
  for (size_t i{ 1ULL }; i < 20000; ++i)
  {
	  probabs.push_back(probabs[i - 1] + 1.0 / 20000.0);
  }

  std::cout << "Before GetImageEmbeddings" << endl;
  auto result{ ces->GetCollectionRepresentants(input, probabs, 8, 8, 90) };
  std::cout << "After GetImageEmbeddings" << endl;

  const int resSize = 256;
  PointWithId* res = new PointWithId[resSize];

  int ix = 0;
  for (auto&& r : result)
  {
	  res[ix].first = r.first.coords.first;
	  res[ix].second = r.first.coords.second;
	  res[ix].clustId = r.first.clust_id;
	  res[ix].imageId = r.second;
	  ++ix;
  }

  int i = 0;
  for (auto&& r : result)
  {
    std::cout << res[i].first << ", " << res[i].second << '\t' << res[i].clustId << '\t' << res[i].imageId << std::endl;
	++i;
  }

  int tmp;
  std::cin >> tmp;

  return 0;
}

extern "C" __declspec(dllexport) int testSOM() {
	return main();
}

extern "C" __declspec(dllexport) CEmbedSom * initSOM(char* path, size_t datasetSize) {
	ces = new CEmbedSom(path, datasetSize);
	return ces;
}

extern "C" __declspec(dllexport) Point* getSOM (size_t* list, int size, float* probabilities, CEmbedSom* cesLoc) {
	std::vector<size_t> input;
	std::vector<float> probabs;

	for (size_t i{ 0ULL }; i < size; ++i)
	{
		input.push_back(list[i]);
	}

	//if (probabilities == NULL) {
		float uniformProbab = 1.0 / (float)size;
		probabs.push_back(uniformProbab);
		for (size_t i{ 1ULL }; i < size; ++i)
		{
			probabs.push_back(probabs[i - 1] + uniformProbab);
		}
	/*}
	else {
		probabs.push_back(probabilities[0]);
		for (size_t i{ 1ULL }; i < size; ++i)
		{
			probabs.push_back(probabs[i - 1] + probabilities[i]);
		}
	}*/

	auto result{ cesLoc->GetImageEmbeddings(input, probabs) };

	const int resSize = size;
	Point* res = new Point[resSize];

	int i = 0;
	for (auto&& r : result)
	{
		res[i].first = r.coords.first;
		res[i].second = r.coords.second;
		res[i].clustId = r.clust_id;
		//res[i].distCent = r.first.dist_cent;
		//res[i].imageId = list[i];
		++i;
	}
	return res;
}

extern "C" __declspec(dllexport) size_t* randomWeightedSample(float* list, size_t size, size_t count) {
	std::vector<float> weigths(list, list + size);
	std::cerr << "Test count: " << count << "; size: " << size << std::endl;
	auto res = sample_w(weigths, count);
	size_t* sample = new size_t[count];
	size_t i = 0;
	for (auto&& r : res)
	{
		sample[i++] = r;
	}
	return sample;
}

extern "C" __declspec(dllexport) PointWithId* getSOMRepresentants(size_t* list, int* size, float* probabilities, CEmbedSom* cesLoc, size_t xdim, size_t ydim, size_t rlen) {
	std::vector<size_t> input;
	std::vector<float> probabs;

	for (size_t i{ 0ULL }; i < *size; ++i)
	{
		input.push_back(list[i]);
	}
	for (size_t i{ 0ULL }; i < *size; ++i)
	{
		probabs.push_back(probabilities[i]);
	}

	auto result{ cesLoc->GetCollectionRepresentants(input, probabs, xdim, ydim, rlen) };

	const int resSize = xdim * ydim;
	*size = resSize;
	PointWithId* res = new PointWithId[resSize];

	int i = 0;
	for (auto&& r : result)
	{
		res[i].first = r.first.coords.first;
		res[i].second = r.first.coords.second;
		res[i].clustId = r.first.clust_id;
		//res[i].distCent = r.first.dist_cent;
		res[i].imageId = r.second;
		++i;
	}
	return res;
}