
#include <chrono>




std::chrono::high_resolution_clock::time_point TICTOC_START;

double st;


void TIC(){
    TICTOC_START = std::chrono::high_resolution_clock::now();

} 

double TOC() {
  auto duration = std::chrono::high_resolution_clock::now() - TICTOC_START;
  return std::chrono::duration_cast<std::chrono::microseconds>(duration)
      .count() /
      1e6;

  std::cout << "cost time is:"
            << std::chrono::duration_cast<std::chrono::microseconds>(duration)
                       .count() /
                   1e6
            << " second" << std::endl;
}