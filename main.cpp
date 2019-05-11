
#include <iostream>
#include <filesystem>
#include <string>

#include "CEmbedSom.h"

int main()
{
  const std::filesystem::path path{ R"ddd(c:\Users\devwe\source\repos\CEmbedSomCP\data\images-ordered-pca.bin)ddd" };

  CEmbedSom ces{ path.string() };

  std::vector<size_t> input;

  for (size_t i{ 0ULL }; i < 4000; ++i)
  {
    input.push_back(i);
  }

  auto result{ ces.GetImageEmbeddings(input) };

  for (auto&& r : result)
  {
    std::cout << r.first.first << ", " << r.first.second << '\t' << r.second << std::endl;
  }

  return 0;
}