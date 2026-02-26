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

    std::vector<Agent> to_up, to_down;

  public:
    Simulation(int w, int h, int t, int s, int n_agents)
      : W(w), H(h), T(t), S(s), total_agents(n_agents), current_season(Season::SECA) {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        partition_domain();

        recv_up.assign(local_W, 0.0f);
        recv_down.assign(local_W, 0.0f);

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

      double start_time = MPI_Wtime();

      for(int t = 0; t < T; ++t) {
        update_season(t);
        exchange_halos();
        process_agents();
        migrate_agents();
        update_grid();
        visualize_resources(t);
        if (t % S == 0 || t == T - 1) {

          collect_metrics(t);

        }
      }

      double end_time = MPI_Wtime();
      if (rank == 0) {
        std::cout << "[Desempenho] Tempo total de simulação " << (end_time - start_time) << " segundos." << std::endl;
      }
    }

    void visualize_resources(int ciclo) {
      std::vector<float> local_data(local_W * local_H);
      for (int i = 0; i < local_grid.size(); ++i) {
         local_data[i] = local_grid[i].resource;
      }

      std::vector<float> global_grid_data;
      if (rank == 0) {
        global_grid_data.resize(W * H);
      }

      MPI_Gather(local_data.data(), local_W * local_H, MPI_FLOAT,
                 global_grid_data.data(), local_W * local_H, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

      if (rank == 0) {
         std::cout << "\n--- Mapa de Recursos (Ciclo " << ciclo << ") ---\n";
         for (int y = 0; y < H; ++y) {
             for (int x = 0; x < W; ++x) {
                 printf("%5.1f ", global_grid_data[y * W + x]);
              }
             std::cout << "\n";
          }
         std::cout << "------------------------------------------\n";
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
                 recv_down.data(), local_W, MPI_FLOAT, down_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Sendrecv(send_down.data(), local_W, MPI_FLOAT, down_neighbor, 1,
                 recv_up.data(), local_W, MPI_FLOAT, up_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    
    void process_agents() {
      std::vector<Agent> next_cycle_agents;

      #pragma omp parallel
      {
        std::vector<Agent> thread_local_agents;
        std::vector<Agent> thread_to_up, thread_to_down;

        #pragma omp for
        for (int i = 0; i < local_agents.size(); i++) {
          Agent& a = local_agents[i];

          int lx = a.x - offsetX;
          int ly = a.y - offsetY;
          float r = local_grid[ly * local_W + lx].resource;

          //execute_synthetic_load(r);

          auto [next_x, next_y] = decide_destination(a);

          if (next_y >= offsetY && next_y < offsetY + local_H) {
            consume_resource(a.x, a.y);
            a.x = next_x;
            a.y = next_y;
            thread_local_agents.push_back(a);
          } else if (next_y < offsetY) {
            a.y = next_y; a.x = next_x;
            thread_to_up.push_back(a);
          } else {
            a.y = next_y; a.x = next_x;
            thread_to_down.push_back(a);
          }
        }

        #pragma omp critical
        {
          next_cycle_agents.insert(next_cycle_agents.end(), thread_local_agents.begin(), thread_local_agents.end());
          to_up.insert(to_up.end(), thread_to_up.begin(), thread_to_up.end());
          to_down.insert(to_down.end(), thread_to_down.begin(), thread_to_down.end());
        }
      }
      local_agents = std::move(next_cycle_agents);
    }
    void migrate_agents() {
      int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
      int down_neighbor = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

      int count_to_up = to_up.size();
      int count_from_down = 0;
      MPI_Sendrecv(&count_to_up, 1, MPI_INT, up_neighbor, 10,
                   &count_from_down, 1, MPI_INT, down_neighbor, 10,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int count_to_down = to_down.size();
      int count_from_up = 0;
      MPI_Sendrecv(&count_to_down, 1, MPI_INT, down_neighbor, 11,
                   &count_from_up, 1, MPI_INT, up_neighbor, 11,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<Agent> from_down(count_from_down, Agent(0,0,0,0));
      std::vector<Agent> from_up(count_from_up, Agent(0,0,0,0));

      Agent dummy(0,0,0,0);
      void* ptr_to_up = to_up.empty() ? &dummy : to_up.data();
      void* ptr_from_down = from_down.empty() ? &dummy : from_down.data();
      void* ptr_to_down = to_down.empty() ? &dummy : to_down.data();
      void* ptr_from_up = from_up.empty() ? &dummy : from_up.data();

      MPI_Sendrecv(ptr_to_up, count_to_up * sizeof(Agent), MPI_BYTE, up_neighbor, 20,
                   ptr_from_down, count_from_down * sizeof(Agent), MPI_BYTE, down_neighbor, 20,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Sendrecv(ptr_to_down, count_to_down * sizeof(Agent), MPI_BYTE, down_neighbor, 21,
                   ptr_from_up, count_from_up * sizeof(Agent), MPI_BYTE, up_neighbor, 21,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      local_agents.insert(local_agents.end(), from_up.begin(), from_up.end());
      local_agents.insert(local_agents.end(), from_down.begin(), from_down.end());

      to_up.clear();
      to_down.clear();
    }

    void update_grid(){
      
      #pragma omp parallel for collapse(2)
      for (int j = 0; j < local_H; ++j) {
        for (int i = 0; i < local_W; ++i) {
          int index = j * local_W + i;
          Cell& cell = local_grid[index];

          float taxa_regeneracao = (current_season == Season::CHEIA) ? 5.0f : 1.0f;

          cell.resource += taxa_regeneracao;

          float limite_maximo = 20.0f; // Padrão para ALDEIA ou ROCADO
          if (cell.type == CellType::PESCA) {
              limite_maximo = 100.0f;
          } else if (cell.type == CellType::COLETA) {
              limite_maximo = 50.0f;
          }

          if (cell.resource > limite_maximo) {
            cell.resource = limite_maximo;
          }
        }
      }
    }
    void collect_metrics(int t) {
      long local_agents_count = local_agents.size();
      double local_total_resource = 0.0;
      double local_total_consumption = 0.0;

      #pragma omp parallel for reduction(+:local_total_resource, local_total_consumption)
      for (int i = 0; i < local_W * local_H; ++i) {
          local_total_resource += local_grid[i].resource;
          local_total_consumption += local_grid[i].accumulated_consumption;
      }

      long global_agents_count = 0;
      double global_total_resource = 0.0;
      double global_total_consumption = 0.0;

      MPI_Reduce(&local_agents_count, &global_agents_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_total_resource, &global_total_resource, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&local_total_consumption, &global_total_consumption, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (rank == 0) {
          std::cout << "=== MÉTRICAS GLOBAIS (Ciclo " << t << ") ===" << std::endl;
          std::cout << " -> População (Agentes Vivos) : " << global_agents_count << std::endl;
          std::cout << " -> Recurso Total no Ambiente : " << global_total_resource << std::endl;
          std::cout << " -> Consumo Acumulado         : " << global_total_consumption << std::endl;
          std::cout << "======================================\n" << std::endl;
      }
    }

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
    void execute_synthetic_load(float r) {
      const int MAX_LOAD = 10000;
      int iterations = std::min(static_cast<int>(r * 100), MAX_LOAD);
    
      volatile float dummy = 0.0f;
      for(int c = 0; c < iterations; c++) {
          dummy += r * 0.01f;
      }
    }

    std::pair<int, int> decide_destination(const Agent& a) {
      int best_x = a.x;
      int best_y = a.y;

      float max_resource = get_resource_at(a.x, a.y);

      int dx[] = {0,0,1,-1};
      int dy[] = {-1,1,0,0};

      for (int i = 0; i < 4; ++i) {
        int nx = a.x + dx[i];
        int ny = a.y + dy[i];

        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

        float neighbor_resource = get_resource_at(nx, ny);

        if (neighbor_resource > max_resource) {
          max_resource = neighbor_resource;
          best_x = nx;
          best_y = ny;
        }
      }

      return {best_x, best_y};
    }

  float get_resource_at(int gx, int gy) {
    if (gy >= offsetY && gy < offsetY + local_H) {
      int lx = gx - offsetX;
      int ly = gy - offsetY;

      return local_grid[ly * local_W + lx].resource;
    }

    if (gy == offsetY - 1 && rank > 0) {
      return recv_up[gx - offsetX];
    }

    if (gy == offsetY + local_H && rank < num_procs - 1) {
      return recv_down[gx - offsetX];
    }

    return -1.0f;
  }

  void consume_resource(int gx, int gy) {
    int lx = gx - offsetX;
    int ly = gy - offsetY;

    int index = ly * local_W + lx;

    float taxa_consumo = 5.0f;

    #pragma omp critical
    {
      if (local_grid[index].resource >= taxa_consumo) {
          local_grid[index].accumulated_consumption += taxa_consumo; 
          local_grid[index].resource -= taxa_consumo;
      } else {
          local_grid[index].accumulated_consumption += local_grid[index].resource;
          local_grid[index].resource = 0.0f;
      }
    }
  }
};
