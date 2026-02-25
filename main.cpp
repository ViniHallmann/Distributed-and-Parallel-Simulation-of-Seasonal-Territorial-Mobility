#include <iostream> 
#include <mpi.h>
#include "sim.hpp"

int main (int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int W = 100, H = 100, T = 10, S = 2, N_agents = 1000;

  try {
    Simulation sim(W, H, T, S, N_agents);

    //sim.test_partitioning(); MPI_Barrier(MPI_COMM_WORLD);
    //sim.verify_consistency(); MPI_Barrier(MPI_COMM_WORLD);
    //sim.test_initialization(); MPI_Barrier(MPI_COMM_WORLD);

  } catch (const std::exception& e) {
    std::cerr << "Erro durante a simulação: " << e.what() << std::endl;
  }

  MPI_Finalize();
  return 0;
}
