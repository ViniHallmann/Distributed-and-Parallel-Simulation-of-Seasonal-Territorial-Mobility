#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

typedef enum { ALDEIA, PESCA, COLETA, ROCADO, INTERDITADA } CellType;
typedef enum { SECA, CHEIA } Estacao;

typedef struct { 
    CellType type; 
} CellStatic;

typedef struct {
    float resource;
    float cost;
    int accessible;
} CellDynamic;

typedef struct {
    int id;
    int grid_x, grid_y;
    float energy;
} Agent;

typedef struct {
    int W, H;
    int T;
    int S;
    int N_agents;
} SimParams;

typedef struct {
    SimParams params;
    
    int rank, num_procs;
    
    int local_W, local_H;
    int offsetX, offsetY;
    
    CellStatic  *grid_static;
    CellDynamic *grid_dynamic;
    
    Agent *agents;
    int n_agents_local;
    
    Estacao estacao_atual;
} SimProcess;