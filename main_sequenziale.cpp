#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>    // solo per omp_get_wtime()

using namespace std;

int main(int argc, char const *argv[])
{
    // Controllo argomenti (una sola volta)
    if (argc < 3) {
        cerr << "Uso: " << argv[0]
             << " <dimensione_matrice> <numero_step>\n";
        return 1;
    }

    // Parametri da riga di comando
    int base_dim = atoi(argv[1]);   // es. 128
    int steps    = atoi(argv[2]);   // es. 1000, ad esempio

    // Vector e file per i tempi
    vector<double> times;
    times.reserve(100);
    ofstream time_file("time_seq.csv");
    time_file << "Run,TimeSeconds\n"; // intestazione CSV


    // Ripeto il ciclo n volte
    for (int h = 0; h < steps; ++h)
    {
        // Inizio cronometraggio
        double start = omp_get_wtime();

        // dimensioni reali (+1 per dimensione utente + bordo)
        int dimension = base_dim + 1;
        float alpha = 0.5f;
        float dt    = 0.1f;

        // inizializzazione matrice
        float** matrix      = new float*[dimension];
        float** next_matrix = new float*[dimension];
        for (int i = 0; i < dimension; ++i) {
            matrix[i]      = new float[dimension]();
            next_matrix[i] = new float[dimension]();
        }

        // azzera le matrici
        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j < dimension; ++j) {
                matrix[i][j]      = 0.0f;
                next_matrix[i][j] = 0.0f;
            }
        }

        // parametri costanti
        matrix[5][5]   = 5.0f;
        matrix[20][20] = 3.0f;


        for (int k = 0; k < steps; ++k) {
            for (int i = 1; i < dimension - 1; ++i) {
                for (int j = 1; j < dimension - 1; ++j) {
                    next_matrix[i][j] = matrix[i][j]
                        + alpha * dt * (
                            matrix[i+1][j] +
                            matrix[i-1][j] +
                            matrix[i][j+1] +
                            matrix[i][j-1]
                            - 4.0f * matrix[i][j]
                          );
                }
            }

            // Reimposta le sorgenti
            next_matrix[5][5]   = 5.0f;
            next_matrix[20][20] = 3.0f;

            // Swap delle matrici
            swap(matrix, next_matrix);
        }

        // Output finale (solo a scopo di verifica, puoi commentarlo se non ti serve)
        {
            ofstream fileo_out("output_seq.txt", ios::out | ios::trunc);
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; j < dimension; ++j) {
                    fileo_out << matrix[i][j] << " ";
                }
                fileo_out << "\n";
            }
        }

        // Deallocazione
        for (int i = 0; i < dimension; ++i) {
            delete[] matrix[i];
            delete[] next_matrix[i];
        }
        delete[] matrix;
        delete[] next_matrix;

        // Fine cronometraggio
        double finish = omp_get_wtime();
        double elapsed = finish - start;

        // Salvo il tempo


        times.push_back(elapsed);
        time_file << h + 1 << "," << elapsed << "\n";
    }

    time_file.close();
    return 0;
}
