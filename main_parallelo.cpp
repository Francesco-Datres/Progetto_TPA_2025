#include <cstdlib>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <numeric>  // per accumulate

using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Uso: " << argv[0] << " <dimensione_matrice> <numero_step>\n";
        return 1;
    }

    // Parametri da riga di comando
    int BASE_N = atoi(argv[1]);   // es. 128
    int STEPS  = atoi(argv[2]);   // es. 5000

    // Aggiungo 2 per il bordo di supporto
    int N = BASE_N + 2;

    const float alpha = 0.5f;
    const float dt    = 0.1f;

    // Alloco le matrici 
    float** matrix = new float*[N];
    float** next_matrix = new float*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new float[N];
        next_matrix[i] = new float[N];
    }

    // Inizializzo le matrici
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j++){
            matrix[i][j] = 0;
            next_matrix[i][j] = 0;
        }
    }

    // Imposto le sorgenti fisse
    matrix[5][5]   = 5.0f;
    matrix[20][20] = 3.0f;

    // Imposto variabili per convergenza
    double sum_old = 0.0;
    double sum_new = 0.0;
    double epsilon = 0.01;
    bool flag = 0;

    // Configuro OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(6);

    // Inizio cronometraggio
    vector<double> times;
    double last_time = omp_get_wtime();

    // Regione parallela
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        // Calcolo quante righe interne toccano a ciascun thread
        int internal = N - 2;                      // righe 1..N-2
        int base    = internal / nth;              // righe di base
        int extra   = internal % nth;              // quelle in più da distribuire
        int start_i = 1 + tid * base + min(tid, extra);
        int count   = base + (tid < extra ? 1 : 0);
        int end_i   = start_i + count;             // non incluso

        for (int step = 0; step < STEPS; ++step) {
            double local_sum_new = 0;  // Variabile locale per la somma del thread corrente
            // 1) calcolo next_matrix sulla mia fetta
            for (int i = start_i; i < end_i; ++i) {
                for (int j = 1; j < N-1; ++j) {
                    next_matrix[i][j] = matrix[i][j]
                        + alpha * dt * (
                            matrix[i+1][j] +
                            matrix[i-1][j] +
                            matrix[i][j+1] +
                            matrix[i][j-1] -
                            4.0f * matrix[i][j]
                          );
                local_sum_new += next_matrix[i][j];
                }
            }

            // Accumulo la somma di tutti i thread
            #pragma omp atomic
            sum_new += local_sum_new;

            // 2) sincronizzo tutti i thread
            #pragma omp barrier

            // 3) un solo thread resetta le sorgenti e fa lo swap
            #pragma omp single
            {
                if (abs(sum_new-sum_old) < epsilon)
                {
                    cout<< "Raggiunta convergenza: " << epsilon << endl;
                    flag = true;
                }

                sum_old = sum_new;
                
                next_matrix[5][5]   = 5.0f;
                next_matrix[20][20] = 3.0f;
                // swap dei puntatori per il ping‑pong

                std::swap(matrix, next_matrix);
                double now = omp_get_wtime();
                times.push_back(now - last_time);
                last_time = now;

                // Resetto la variabile per il prossimo step
                sum_new = 0;
            } 
            if (flag)
            {
               break;
            }
            

        }
    } // fine parallel

    // Fine cronometraggio
    /*double t1 = omp_get_wtime();
    double total = t1 - t0;
    double avg    = total / STEPS;*/

    // Scrivo i tempi di ogni step su file time.txt
    ofstream tf("time_parallel.txt");
    for (double t : times) tf << t << "\n";
    tf.close();

    // Stampo i tempi
    //cout << "Tempo totale: " << total << " s\n";
    //cout << "Tempo medio per step: " << avg   << " s\n";

    // Salvo il risultato finale su file
    ofstream fout("output.txt");
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            fout << matrix[i][j] << (j+1 < N-1 ? ' ' : '\n');
        }
    }
    fout.close();

    // Dealloco
    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
        delete[] next_matrix[i];
    }
    delete[] matrix;
    delete[] next_matrix;

    return 0;
}
