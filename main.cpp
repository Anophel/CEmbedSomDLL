
#include <iostream>
#include <string>

#include "CEmbedSom.h"

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