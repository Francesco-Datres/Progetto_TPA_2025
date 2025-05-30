#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>    // solo per omp_get_wtime()
#include <iomanip>  // per stampare in virgola mobile
//52.28 s
using namespace std;

int main(int argc, char const *argv[])
{
    // Controllo argomenti
    if (argc < 3) {
        cerr << "Uso: " << argv[0]
             << " <dimensione_matrice> <epsilon_convergenza>\n";
        return 1;
    }

    // Parametri da riga di comando
    const int base_dim = atoi(argv[1]);
    const float epsilon = atof(argv[2]);
    const int max_steps = 5000;

    float sum_new = 0.0;
    float sum_old = 0.0;

    // Parametri e dimensioni
    const int dimension = base_dim + 2;
    const float alpha = 0.5f;
    const float dt = 0.1f;

    // Parametri per misurare il tempo
    double finish = 0;
    double start  = 0;
    double elapsed = 0;

    // Allocazione matrici
    float** matrix = new float*[dimension];
    float** next_matrix = new float*[dimension];
    for (int i = 0; i < dimension; ++i) {
        matrix[i] = new float[dimension]();
        next_matrix[i] = new float[dimension]();
    }

    // Inizializzazione
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j)
            matrix[i][j] = next_matrix[i][j] = 0.0f;
    
    matrix[5][5]      = 5.0f;
    next_matrix[5][5] = 5.0f;
    matrix[20][20]      = 3.0f;
    next_matrix[20][20]  = 3.0f;

    // File tempi
    ofstream time_file("time_seq.csv");
    time_file << "Run,TimeSeconds\n";

    // inizio cronometraggio
    start = omp_get_wtime();

    // Iterazione con controllo convergenza
    int k;
    for (k = 0; k < max_steps; ++k) {

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
                sum_new += next_matrix[i][j];
                
            }
        }

        // Reimposta le sorgenti
        next_matrix[5][5] = 5.0f;
        next_matrix[20][20] = 3.0f;
        next_matrix[500][500] = 5.0f;
        next_matrix[200][200] = 3.0f;
        next_matrix[800][800] = 5.0f;
        next_matrix[700][700] = 3.0f;

        // Verifica convergenza
        if (abs(sum_new-sum_old) < epsilon && k>100)
            break;
        
            sum_old = sum_new;
            sum_new = 0;

        // Scambia le matrici
        swap(matrix, next_matrix);
        // Fine cronometraggio
        
        finish = omp_get_wtime();
        elapsed = finish - start;
        start = finish;
        // Scrivo tempo su file
        time_file << fixed << setprecision(8) << k  << ", "<< elapsed *1e3 << "\n";
    }


    time_file.close();

    // Output finale (opzionale)
    ofstream out("output_seq.txt");
    for (int i = 1; i < dimension-1; ++i) {
        for (int j = 1; j < dimension-1; ++j)
            out << fixed << setprecision(3) << matrix[i][j] << " ";
            out << "\n";
    }

    // Deallocazione
    for (int i = 0; i < dimension; ++i) {
        delete[] matrix[i];
        delete[] next_matrix[i];
    }
    delete[] matrix;
    delete[] next_matrix;

    // Stampa passi eseguiti
    cout << "Convergenza raggiunta in " << k << " passi.\n";

    return 0;
}
