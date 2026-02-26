# Simulação Híbrida de Ecossistema (MPI + OpenMP)

Este projeto implementa uma simulação distribuída e paralelizada de um ecossistema utilizando uma arquitetura híbrida com **MPI (Message Passing Interface)** e **OpenMP**. 

O objetivo da simulação é modelar o comportamento de agentes autônomos (representando indivíduos como pescadores ou coletores) que se movem por um grid espacial, consumindo recursos naturais que se regeneram sazonalmente. O projeto demonstra conceitos avançados de Computação de Alto Desempenho (HPC), incluindo particionamento de domínio, troca de halos, migração de dados entre nós de processamento e concorrência segura entre threads.

## 🚀 Tecnologias Utilizadas
* **C++** (Padrão C++17 recomendado)
* **MPI** (OpenMPI / MPICH) para paralelismo inter-nós (memória distribuída).
* **OpenMP** para paralelismo intra-nós (memória compartilhada).

## ⚙️ Arquitetura de Paralelização

A simulação foi dividida em duas camadas de paralelismo para maximizar a eficiência computacional:

### 1. Nível Distribuído (MPI)
* **Particionamento de Domínio:** O grid global (2D) é fatiado horizontalmente (1D Decomposition). Cada processo MPI é responsável por um subgrid contíguo.
* **Troca de Halos (Ghost Cells):** A cada ciclo, as fronteiras dos subgrids são sincronizadas utilizando `MPI_Sendrecv`. O padrão de comunicação implementado é o de *Shift Unidirecional* (Shift Up / Shift Down) para garantir consistência e evitar *deadlocks*.
* **Migração de Agentes:** Agentes que decidem se mover para fora dos limites do seu subgrid local são empacotados e enviados para o processo vizinho correspondente através da rede.

### 2. Nível Local (OpenMP)
* **Processamento Massivo:** O processamento das decisões e da carga computacional sintética (`execute_synthetic_load`) dos agentes é feito em paralelo usando `#pragma omp parallel for`.
* **Consumo Thread-Safe:** O acesso e a dedução dos recursos do grid pelas threads são protegidos por regiões críticas (`#pragma omp critical`) para evitar *Race Conditions* (condições de corrida) quando múltiplos agentes caem na mesma célula.
* **Regeneração de Matriz:** A regeneração sazonal do grid utiliza `#pragma omp parallel for collapse(2)` para otimizar o balanceamento de carga no acesso à memória bidimensional.
* **Redução para Métricas:** A coleta de métricas globais utiliza a cláusula `reduction` do OpenMP em conjunto com o `MPI_Reduce` para garantir contagens exatas sem gargalos de sincronização.

## 🌍 Regras do Ecossistema

1. **Sazonalidade:** O sistema alterna entre duas estações (`CHEIA` e `SECA`) a cada $S$ ciclos. Na estação cheia, os recursos regeneram rapidamente; na seca, o ambiente sofre colapso.
2. **Carga Sintética:** Cada agente executa um volume de operações matemáticas atreladas à quantidade de recurso da célula, simulando o custo computacional real de processamento em HPC.
3. **Migração Física:** O esgotamento dos recursos em um nó força os agentes a migrarem fisicamente na memória para nós adjacentes para evitar a inanição.

## 🛠️ Compilação e Execução

Para compilar o projeto, você precisará de um compilador C++ com suporte a MPI e OpenMP instalado no seu cluster ou ambiente de desenvolvimento (ex: `gcc`, `openmpi`).

### Compilação
```bash
mpic++ -O3 -fopenmp main.cpp -o sim
```

(A flag -O3 é recomendada para otimização de performance da carga sintética).
### Execução

Defina o número de threads OpenMP por processo e execute com o mpirun ou mpiexec. Exemplo para rodar com 4 processos MPI, sendo 4 threads OpenMP por processo:
```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./sim
```
(Nota: Os parâmetros do tamanho do grid (W,H), ciclos (T), tamanho da estação (S) e número de agentes (N) podem ser ajustados na função main()).

```
