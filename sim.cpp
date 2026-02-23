#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

enum class CellType { ALDEIA, PESCA, COLETA, ROCADO, INTERDITADA };
enum class Season {SECA, CHEIA};

class Cell{
  public:
    CellType type;
    float resource;
    bool accessible;
    float accumulated_consumption;

    Cell() : type(CellType::ALDEIA), resource(0.0f), accessbile(true), accumulated_consumption(0.0f) {}
};

class Agent{
  public:
    int id;
    int x, y;
    float energy;

    Agent(int id, int x, int y, int float energy)
      : id(id), x(x), y(y), energy(energy) {}
};

class Simulation {
  private: 
    int W, H, T, S, total_agents;

    int rank, num_procs;
    int local_W, local_H;
    int offsetX, offsetY;

    std::vector<Cell> local_grid;
    std::vector<Agent> local_agents;
    Season current_season;

  public:
    Simulation(int w, int h, int t, int s, int n_agents)
      : W(w), H(h), T(t), S(s), total_agents(n_agents), current_season(Estacao::SECA) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        partition_domain();
        initialize_grid();
        initialize_agents();
    }

    void partition_domain(){

      local_W = W;
      local_H = H / num_procs;
      offsetX = 0;
      offsetY = rank * local_H;

      local_grid.resize(local_W * local_H);
    }

    void run() {
      for(int t = 0; t < T; ++t) {
        update_season(t);
        exchange_halos();
        process_agents();
        migrate_agents();
        update_grid();
        collect_metrics();
      }
    }
  private:
    void update_season(int t);
    void exchange_halos();
    void process_agents();
    void migrate_agents();
    void update_grid();
    void collect_metrics();
    void initialize_grid();
    void initialize_agents();
}
