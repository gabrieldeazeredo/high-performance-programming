#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

typedef struct
{
    double x;
    double y;
    double z;
} PARTICLE;

double compute_distance(const PARTICLE* p1, const PARTICLE* p2)
{
    return sqrt((p1->x - p2->x) * (p1->x - p2->x) +
                (p1->y - p2->y) * (p1->y - p2->y) +
                (p1->z - p2->z) * (p1->z - p2->z));
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Uso incorreto. Você deve fornecer 3 argumentos.\n");
        fprintf(stderr, "Exemplo de uso: %s arquivo1.pos 5 arquivo2.viz\n", argv[0]);
        return 1;
    }

    char* input_name = argv[1];
    FILE* input_file = fopen(input_name, "r");

    if (input_file == NULL)
    {
        fprintf(stderr, "Erro ao abrir o arquivo %s de entrada.\n", input_name);
        return 1;
    }

    char* output_name = argv[3];
    FILE* output_file = fopen(output_name, "w");

    if (output_file == NULL)
    {
        fprintf(stderr, "Erro ao abrir o arquivo %s de saída.\n", output_name);
        return 1;
    }

    PARTICLE* particles = NULL;

    size_t n = 0;
    size_t k = atoi(argv[2]);

    double x, y, z;

    while (fscanf(input_file, "%lf %lf %lf", &x, &y, &z) == 3)
    {
        particles = (PARTICLE*)realloc(particles, (n + 1) * sizeof(PARTICLE));
        particles[n].x = x;
        particles[n].y = y;
        particles[n].z = z;
        n++;
    }

    size_t** nearest_neighbors = (size_t**)malloc(n * sizeof(size_t*));

    for (size_t i = 0; i < n; i++)
    {
        nearest_neighbors[i] = (size_t*)malloc(k * sizeof(size_t));
    }

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

#pragma omp parallel for shared(particles, nearest_neighbors, n, k) default(none)
    for (size_t i = 0; i < n; i++)
    {
        double* distances = (double*)malloc(n * sizeof(double));

#pragma omp parallel for shared(particles, distances, n, i) default(none)
        for (size_t j = 0; j < n; j++)
        {
            distances[j] = compute_distance(&particles[i], &particles[j]);
        }

        for(size_t j = 0; j < k; j++)
        {
            size_t nearest = 0;

            for(size_t m = 0; m < n; m++)
            {
                if (distances[m] < distances[nearest])
                {
                    nearest = m;
                }
            }

            distances[nearest] = INFINITY;
            nearest_neighbors[i][j] = nearest;
        }
        
        free(distances);
    }

    gettimeofday(&t2, NULL);

    printf("%f\n", (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < k - 1; j++)
        {
            fprintf(output_file, "%ld ", nearest_neighbors[i][j]);
        }
        fprintf(output_file, "%ld\n", nearest_neighbors[i][k - 1]);
    }

    for (size_t i = 0; i < n; i++)
    {
        free(nearest_neighbors[i]);
    }

    free(particles);
    free(nearest_neighbors);

    fclose(input_file);
    fclose(output_file);

    return 0;
}
