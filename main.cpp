
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
	double first;
	double second;
	size_t clustId;
};

CEmbedSom* ces;


// ugly AF
#define FRAND (rand() / (double)RAND_MAX)

vector<size_t>
sample_w(const vector<double>& ws, size_t k)
{
	size_t n = ws.size();

	assert(n >= 2);
	assert(k < n);

	size_t branches = n - 1;
	vector<double> tree(branches + n, 0);
	double sum = 0;
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
		double x = FRAND * tree[0];
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

extern "C" __declspec(dllexport) CEmbedSom * initSOM(double* data, size_t datasetSize, size_t dims) {
	ces = new CEmbedSom(data, datasetSize, dims);
	return ces;
}

extern "C" __declspec(dllexport) Point* getSOM (size_t* list, int size, double* probabilities, CEmbedSom* cesLoc) {
	std::vector<size_t> input(list, list + size);
	std::vector<double> probabs;

	double uniformProbab = 1.0 / (double)size;
	probabs.push_back(uniformProbab);
	for (size_t i{ 1ULL }; i < size; ++i)
	{
		probabs.push_back(probabs[i - 1] + uniformProbab);
	}

	log("Not implemented!!!");

#if 0
	auto result{ cesLoc->GetImageEmbeddings(input, probabs) };
	const int resSize = size;
	Point* res = new Point[resSize];

	int i = 0;
	for (auto&& r : result)
	{
		res[i].first = r.coords.first;
		res[i].second = r.coords.second;
		res[i].clustId = r.clust_id;
		++i;
	}
	return res;

#endif
	return NULL;
}

extern "C" __declspec(dllexport) size_t* randomWeightedSample(double* list, size_t size, size_t count) {
	std::vector<double> weigths(list, list + size);
	auto res = sample_w(weigths, count);
	size_t* sample = new size_t[count];
	size_t i = 0;
	for (auto&& r : res)
	{
		sample[i++] = r;
	}
	return sample;
}

extern "C" __declspec(dllexport) PointWithId* getSOMRepresentants(
	size_t* list,
	int* size,
	double* probabilities,
	CEmbedSom* cesLoc,
	bool useCos,
	bool forceRepre,
	bool mostProbab,
	size_t xdim,
	size_t ydim,
	size_t rlen) 
{
	std::vector<size_t> input(list, list + *size);
	std::vector<double> probabs(probabilities, probabilities + *size);

	vector<PointWithId> result;
	if (useCos)
	{
		result = cesLoc->GetCollectionRepresentantsCos(input, probabs, forceRepre, mostProbab, xdim, ydim, rlen);
	}
	else
	{
		result = cesLoc->GetCollectionRepresentantsManh(input, probabs, forceRepre, mostProbab, xdim, ydim, rlen);
	}

	const size_t resSize = xdim * ydim;
	*size = resSize;
	PointWithId* res = new PointWithId[resSize];
	size_t i = 0;
	for (auto&& r : result)
	{
		res[i++] = r;
	}
	return res;
}


extern "C" __declspec(dllexport) void deletePointWithIdArray(PointWithId* toDel) {
	delete[] toDel;
}

extern "C" __declspec(dllexport) void deletePointArray(Point* toDel) {
	delete[] toDel;
}


extern "C" __declspec(dllexport) double* testFunc(double* list, size_t size)
{
	for (size_t i = 0; i < size; ++i) 
	{
		list[i] = list[i] * 2;
	}
	return list;
}


extern "C" __declspec(dllexport) double* testFunc2(double* list, size_t size)
{
	std::cout << "Hello from testFunc2" << std::endl;
	double* res = new double[size];
	for (size_t i = 0; i < size; ++i)
	{
		res[i] = list[i] * 2;
	}
	return res;
}


extern "C" __declspec(dllexport) void deleteTest(double* toDel) {
	delete[] toDel;
}