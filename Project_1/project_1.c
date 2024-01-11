#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//
//      Struct que guarda a posição das partículas.
//

typedef struct 
{
    double x;
    double y;
    double z;
}PARTICLE;

//
//      Função que calcula a distância euclidiana entre duas
//      partículas passadas como parâmetro.
//

double compute_distance(PARTICLE particle_1, PARTICLE particle_2)
{
    return sqrt(pow(particle_1.x - particle_2.x, 2)
            +   pow(particle_1.y - particle_2.y, 2)
            +   pow(particle_1.z - particle_2.z, 2));
}

int main(int argc, char *argv[]) 
{
    //      Verifica se o número de argumentos por linha de comando.

    if (argc != 4) 
    {
        fprintf(stderr, "Uso incorreto. Você deve fornecer 3 argumentos.\n");
        fprintf(stderr, "Exemplo de uso: %s arquivo1.pos 5 arquivo2.viz\n", argv[0]);
        return 1; 
    }

    //      Abre os arquivos de entrada e saída.
 
    char *input_name = argv[1];
    FILE *input_file = fopen(input_name, "r");

    if (input_file == NULL) 
    {
        fprintf(stderr, "Erro ao abrir o arquivo %s de entrada.\n", input_name);
        return 1;
    }

    char *output_name = argv[3];
    FILE *output_file = fopen(output_name, "w");

    if (output_file == NULL) 
    {
        fprintf(stderr, "Erro ao abrir o arquivo %s de saída.\n", output_name);
        return 1;
    }

    //      Declaração de variáveis.

    PARTICLE* particles = NULL;

    size_t n = 0;
    size_t k = atoi(argv[2]);

    //      Recebe os dados do arquivo de entrada.

    double x = 0, y = 0, z = 0;

    while(fscanf(input_file, "%lf %lf %lf", &x, &y, &z) == 3)
    {
        particles = (PARTICLE*) realloc(particles, (n + 1) * sizeof(PARTICLE));

        particles[n].x = x;
        particles[n].y = y;
        particles[n].z = z;

        n++;
    }

    // GT: Você cria uma matriz com todos os pares de distâncias, o que significa
    // GT: que você precisa uma matriz de N x N elementos double, o que vai ser
    // GT: muito espaço de memória se N for grande (calcule a quantidade de memória
    // GT: necessária se N = 100.000)

    //      Cria matriz que armazenará as distâncias das
    //      partículas.

    double **distances = (double **) malloc(n * sizeof(double *));

    for (size_t i = 0; i < n; i++) 
    {
        distances[i] = (double *)malloc(n * sizeof(double));
    }

    //      Cria matriz que armazenará os vizinhos mais próximos,
    //      pois não é o objetivo considerar no tempo calculado
    //      o tempo para escrever no arquivo.

    size_t **nearest_neighbors = (size_t **) malloc(n * sizeof(size_t *));

    for (size_t i = 0; i < n; i++) 
    {
        nearest_neighbors[i] = (size_t *)malloc(k * sizeof(size_t));
    }

    //      Resolução do problema:

    struct timeval t1, t2;

    gettimeofday(&t1, NULL);
    
    for(size_t i = 0; i < n; i++)
    {
        distances[i][i] = INFINITY;

        for(size_t j = (i + 1); j < n; j ++)
        {
            distances[i][j] = compute_distance(particles[i], particles[j]);
            distances[j][i] = distances[i][j];
        }

        // Acha os K vizinhos mais próximos de i

        // GT: O algoritmo aqui abaixo tem complexidade O(N * k).
        // GT: Mas como ele é executado para cada um dos N valores de i,
        // GT: o seu algoritmo geral vai ter complexidade O(N^2 * k)
        
        for(size_t j = 0; j < k; j++)
        {
            size_t nearest = 0;

            for(size_t m = 0; m < n; m++)
            {
                if (distances[i][m] < distances[i][nearest])
                {
                    nearest = m;
                }
            }

            distances[i][nearest] = INFINITY;
            nearest_neighbors[i][j] = nearest;
        }
    } 

    gettimeofday(&t2, NULL);

    printf("%f\n", (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6);

    //      Escreve no arquivo de saída.

    for(size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < k - 1; j++)
        {
            fprintf(output_file, "%ld ", nearest_neighbors[i][j]);
        }
        fprintf(output_file, "%ld\n", nearest_neighbors[i][k - 1]);
    }

    //      Fecha os arquivos e libera a memória alocada.

    for(size_t i = 0; i < n; i++)
    {
        free(distances[i]);
        free(nearest_neighbors[i]);
    }

    free(distances);
    free(particles);
    free(nearest_neighbors);

    fclose(input_file);
    fclose(output_file);

    return 0; 
}
