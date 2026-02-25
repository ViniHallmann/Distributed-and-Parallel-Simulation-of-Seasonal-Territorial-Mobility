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

    Cell() : type(CellType::ALDEIA), resource(0.0f), accessible(true), accumulated_consumption(0.0f) {}
};

class Agent{
  public:
    int id;
    int x, y;
    float energy;

    Agent(int id, int x, int y, float energy)
      : id(id), x(x), y(y), energy(energy) {}
};

class Simulation {
  private: 
    int W, H, T, S, total_agents;

    int rank, num_procs;
    int local_W, local_H;
    int offsetX, offsetY;

    std::vector<float> recv_up;
    std::vector<float> recv_down;

    std::vector<Cell> local_grid;
    std::vector<Agent> local_agents;
    Season current_season;

  public:
    Simulation(int w, int h, int t, int s, int n_agents)
      : W(w), H(h), T(t), S(s), total_agents(n_agents), current_season(Season::SECA) {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        recv_up.assign(local_W, 0.0f);
        recv_down.assign(local_W, 0.0f);

        partition_domain();
        initialize_grid();
        initialize_agents();
        run();
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
        test_exchange_halos();//exchange_halos();
      //  process_agents();
      //  migrate_agents();
      //  update_grid();
      //  collect_metrics();
      }
    }

    void test_partitioning() {
      for (int i = 0; i < num_procs; ++i){
        if (rank == i) {
          std::cout << "[Rank " << rank << "/" << num_procs-1 << "] "
                    << "Subgrid Local: " << local_W << "x" << local_H
                    << " | Offset Global: (" << offsetX << ", " << offsetY << ")"
                    << " | Cobre de Y=" << offsetY << " ate Y=" << (offsetY + local_H - 1)
                    << std::endl << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    void verify_consistency() {
      long local_cells = local_W * local_H;
      long total_cells = 0;

      MPI_Reduce(&local_cells, &total_cells, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      
      if (rank == 0) {
        std::cout << "--- verify_consistency ---" << std::endl;
        std::cout << "Celulas totais esperadas: " << (long)W*H << std::endl;
        std::cout << "Celulas totais calculadas: " << total_cells << std::endl;

        if (total_cells == (long)W*H) {
          std::cout << "SUCESSO" << std::endl;
        } else {
          std::cerr << "ERRO" << std::endl;
        }
      }
    }

    void test_initialization() {
      if (!local_grid.empty()) {
        const Cell& c = local_grid[0];
        std::cout << "[Rank " << rank << "] Primeira Célula - Tipo: " << (int)c.type << "| Recurso: " << c.resource << std::endl;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      int out_of_bound = 0;
      for (const auto& a : local_agents) {
        if (a.x < offsetX || a.x >= offsetX + local_W ||
            a.y < offsetY || a.y >= offsetY + local_H){
          out_of_bound++;
        }
      }

      if (rank == 0) std::cout << "agent validation" << std::endl;

      std::cout << "[Rank " << rank << "] Agents locais: " << local_agents.size() << " | Fora dos limites: " << out_of_bound << std::endl;

      MPI_Barrier(MPI_COMM_WORLD);
    }

    void test_exchange_halos() {
    // 1. Forçar um valor identificável na primeira e última linha de cada rank
    // Usaremos o valor do rank + 100 para diferenciar de recursos iniciais
    for (int i = 0; i < local_W; ++i) {
        local_grid[i].resource = (float)(rank + 100); // Primeira linha
        local_grid[(local_H - 1) * local_W + i].resource = (float)(rank + 100); // Última linha
    }

    // 2. Executar a troca (precisamos de buffers para receber os dados)
    // No seu caso, garanta que os buffers recv_up e recv_down existam na classe
    exchange_halos(); 

    // 3. Verificar se os valores recebidos batem com os ranks dos vizinhos
    bool success = true;
    if (rank > 0) { // Tem vizinho acima
        if (recv_up[0] != (float)(rank - 1 + 100)) success = false;
    }
    if (rank < num_procs - 1) { // Tem vizinho abaixo
        if (recv_down[0] != (float)(rank + 1 + 100)) success = false;
    }

    // 4. Reportar resultado
    for (int i = 0; i < num_procs; ++i) {
        if (rank == i) {
            if (success) {
                std::cout << "[Rank " << rank << "] Teste de Halo: SUCESSO" << std::endl;
            } else {
                std::cout << "[Rank " << rank << "] Teste de Halo: FALHA (Valores incorretos)" << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

  private:
    void update_season(int t) {
      if (t % S == 0){
        if (rank == 0) {
          current_season = (current_season == Season::SECA) ? Season::CHEIA : Season::SECA;
        }

        int season_val = static_cast<int>(current_season);
        MPI_Bcast(&season_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

        current_season = static_cast<Season>(season_val);

        if (rank == 0) {
          std::cout << "[Ciclo " << t << "] Mudança de estação: "
                    << (current_season == Season::SECA ? "SECA" : "CHEIA") << std::endl;
        }
      }
    }
    
    void exchange_halos() {
      int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
      int down_neighbor = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

      std::vector<float> send_up(local_W), send_down(local_W);

      for (int i = 0; i < local_W; ++i) {
        send_up[i] = local_grid[i].resource;
        send_down[i] = local_grid[(local_H - 1) * local_W + i].resource;
      }

      MPI_Sendrecv(send_up.data(), local_W, MPI_FLOAT, up_neighbor, 0,
                   recv_up.data(), local_W, MPI_FLOAT, up_neighbor, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Sendrecv(send_down.data(), local_W, MPI_FLOAT, down_neighbor, 0,
                   recv_down.data(), local_W, MPI_FLOAT, down_neighbor, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    
    void process_agents();
    void migrate_agents();
    void update_grid();
    void collect_metrics();

    void initialize_grid() {
      srand(42 + rank);

      for (int j = 0; j < local_H; ++j){
        for (int i = 0; i < local_W; ++i) {
          int gx = offsetX + i;
          int gy = offsetY + j;

          int local_index = j * local_W + i;
          Cell& cell = local_grid[local_index];

          if ((gx + gy) % 10 == 0) cell.type = CellType::PESCA;
          else if ((gx + gy) % 7 == 0) cell.type = CellType::COLETA;
          else cell.type = CellType::ALDEIA;

          if (cell.type == CellType::PESCA) cell.resource = 100.0f;
          else if(cell.type == CellType::COLETA) cell.resource = 50.0f;
          else cell.resource = 20.0f;

          cell.accessible = true;
        }
      }
    }

    void initialize_agents() {
      int agents_per_proc = total_agents / num_procs;

      for (int i = 0; i < agents_per_proc; ++i) {
        int lx = rand() % local_W;
        int ly = rand() % local_H;

        local_agents.emplace_back(
              rank* 10000 + i,
              offsetX + lx,
              offsetY + ly,
              100.0f
            );
      }
    }
};
