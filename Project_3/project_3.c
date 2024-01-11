#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <stddef.h>

//      Definição da struct de partículas.

typedef struct
{
    double x;
    double y;
    double z;
} PARTICLE;

//      Cabeçário de funções.

double compute_distance(PARTICLE* p1, PARTICLE* p2);
void compute_partition(int N, int p, int *count, int *offset);
void int_swap(int* a, int* b);
void double_swap(double* a, double* b);
double quick_select(double* arr, int low, int high, int k); 
void bubble_sort(double* arr, int* indexes, int n);
MPI_Datatype create_particles_mpi_type();

//      Programa principal.

int main(int argc, char* argv[])
{
    int rank;
    int nprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //      Início da leitura de dados.

    if (argc != 4)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Uso incorreto. Você deve fornecer 3 argumentos.\n");
            fprintf(stderr, "Exemplo de uso: %s arquivo1.pos 5 arquivo2.viz\n", argv[0]);
            
            MPI_Finalize();
            return 1;
        }
    }

    PARTICLE* particles = NULL;
    int n = 0;
    int k = atoi(argv[2]);

    if (rank == 0)
    {
        // O processo 0 lê os dados do arquivo.

        char* input_name = argv[1];
        FILE* input_file = fopen(input_name, "r");

        if (input_file == NULL)
        {
            fprintf(stderr, "Erro ao abrir o arquivo %s de entrada.\n", input_name);
            MPI_Finalize();
            return 1;
        }

        double x, y, z;

        while (fscanf(input_file, "%lf %lf %lf", &x, &y, &z) == 3)
        {
            particles = (PARTICLE*) realloc(particles, (n + 1) * sizeof(PARTICLE));
            particles[n].x = x;
            particles[n].y = y;
            particles[n].z = z;
            n++;
        }

        fclose(input_file);
    }

    //      Fim da leitura de arquivos.

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // Broadcast do valor n lido pelo processo 0.
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); 

    if (rank != 0)
    {
        // Aloca memória para as partículas nos outros processos.
        particles = (PARTICLE *)malloc(n * sizeof(PARTICLE));
    }


    MPI_Datatype mpi_particles_type = create_particles_mpi_type();
    MPI_Bcast(particles, n, mpi_particles_type, 0, MPI_COMM_WORLD);

    int *count = (int *) malloc(nprocs * sizeof(int));
    int *offset = (int *) malloc(nprocs * sizeof(int));

    compute_partition(n, nprocs, count, offset);

    double* dist_data = NULL;

    if (rank == 0)
    {
        dist_data = malloc(n * n * sizeof(double*));        
    }
    
    int i = offset[rank];
    int j = 0;

    double* local_distances = malloc(count[rank] * n * sizeof(double));

    while(j < count[rank])    
    {
        for (int m = 0; m < n; m++)
        {
            local_distances[j * n + m] = compute_distance(&particles[i], &particles[m]);
        }

        i++;
        j++;
    }

    int *total_count = malloc(sizeof(int) * nprocs);
    int *total_offset = malloc(sizeof(int) * nprocs);

    for (int i = 0; i < nprocs; i++)
    {
        total_count[i] = count[i] * n;
        total_offset[i] = offset[i] * n;
    }

    // Distancias locais para a distância global

    MPI_Gatherv(&local_distances[0], count[rank] * n, MPI_DOUBLE, 
                &dist_data[offset[rank]], total_count, &total_offset[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    free(count);
    free(offset);
    free(total_count);
    free(total_offset);

    if(rank == 0)
    {
        int** nearest_neighbors = (int**)malloc(n * sizeof(int*));

        for (int i = 0; i < n; i++)
        {
            nearest_neighbors[i] = (int*)malloc(k * sizeof(int));
        }

        for (int i = 0; i < n; i++)
        {
            double temp_arr[n];

            for (int j = 0; j < n; j++)
            {
                temp_arr[j] = dist_data[i * n + j];
            }
            
            double k_smallest = quick_select(temp_arr, 0, n - 1, k - 1);

            int count_values = 0;
            double* nearest_dist = (double*) malloc(sizeof(double) * k);

            for (int j = 0; j < n; j++)
            {
                if (dist_data[i * n + j] <= k_smallest)
                {
                    nearest_neighbors[i][count_values] = j;
                    nearest_dist[count_values] = dist_data[i * n + j];

                    if (count_values == k)
                    {
                        break;
                    }
                    count_values += 1;
                }   
            }
            
            bubble_sort(nearest_dist, nearest_neighbors[i], k);
            free(nearest_dist);
        }

        gettimeofday(&t2, NULL);

        printf("%f\n", (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6);

        char* output_name = argv[3];
        FILE* output_file = fopen(output_name, "w");

        if (output_file == NULL)
        {
            fprintf(stderr, "Erro ao abrir o arquivo %s de saída.\n", output_name);
            MPI_Finalize();
            return 1;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k - 1; j++)
            {
                fprintf(output_file, "%d ", nearest_neighbors[i][j]);
            }
            fprintf(output_file, "%d\n", nearest_neighbors[i][k - 1]);
        }

        fclose(output_file);

        for (int i = 0; i < n; i++)
        {
            free(nearest_neighbors[i]);
        }

        free(nearest_neighbors);
    }

    free(particles);

    MPI_Type_free(&mpi_particles_type);
    MPI_Finalize();

    return 0;
}


//      Função que cria o tipo de dado MPI_PARTICLES 
//      para o broadcast.

MPI_Datatype create_particles_mpi_type() 
{
    // Número de campos na struct
    int blocklengths[3] = {1, 1, 1};

    // Tipo de dado de cada campo
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}; 
    
    // Offset de cada campo
    MPI_Aint offsets[3];  

    offsets[0] = offsetof(PARTICLE, x);
    offsets[1] = offsetof(PARTICLE, y);
    offsets[2] = offsetof(PARTICLE, z);

    MPI_Datatype mpi_particle;

    MPI_Type_create_struct(3, blocklengths, offsets, types, &mpi_particle);
    MPI_Type_commit(&mpi_particle);

    return mpi_particle;
}

//      Calcula a distância de duas partóculas. 
//      Caso seja a mesma partícula, retorna INF.

double compute_distance(PARTICLE* p1, PARTICLE* p2)
{
    double dist = sqrt((p1->x - p2->x) * (p1->x - p2->x) +
                       (p1->y - p2->y) * (p1->y - p2->y) +
                       (p1->z - p2->z) * (p1->z - p2->z));

    if (dist == 0)
    {
        return INFINITY;
    }
    else
    {
        return dist;
    }
}

//      Funções de troca.

void int_swap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void double_swap(double* a, double* b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

//      Função de particionamento para o quick select.

int part_array(double* arr, int low, int high) 
{
    double pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) 
    {
        if (arr[j] <= pivot) 
        {
            i++;
            double_swap(&arr[i], &arr[j]);
        }
    }

    double_swap(&arr[i + 1], &arr[high]);

    return i + 1;
}

double quick_select(double* arr, int low, int high, int k) 
{
    if (low < high)
    {
        int pivot_index = part_array(arr, low, high);

        if (pivot_index == k) 
        {
            return arr[pivot_index];
        }
        else if (k < pivot_index) 
        {
            return quick_select(arr, low, pivot_index - 1, k);
        }
        else 
        {
            return quick_select(arr, pivot_index + 1, high, k);
        }
    }
    
    return arr[low];
}

void compute_partition(int N, int p, int *count, int *offset) 
{
  int q = N / p;
  int r = N % p;
  int curr_off = 0;

  for (int i = 0; i < p; ++i) 
  {
    count[i] = i < r ? q + 1 : q;
    offset[i] = curr_off;
    curr_off += count[i];
  }
}

// Implementação do bubble sort melhorado.

void bubble_sort(double* arr, int* indexes, int n)
{

    int swapped;
    for (int i = 0; i < n - 1; i++) 
    {
        swapped = 0;
        for (int j = 0; j < n - i - 1; j++) 
        {
            if (arr[j] > arr[j + 1]) 
            {
                double_swap(&arr[j], &arr[j + 1]);
                int_swap(&indexes[j], &indexes[j + 1]);

                swapped = 1;
            }
        }
 
        if (swapped == 0)
        {
            break;
        }
    }
}
