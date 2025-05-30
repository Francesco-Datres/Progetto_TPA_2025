#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <iomanip>  // per stampare in virgola mobile

using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Uso: " << argv[0] << " <dimensione_matrice> <epsilon>\n";
        return 1;
    }
    if (atoi(argv[1])<20)
    {
        cerr << "Dimensione matrince: " << argv[1] << " < 20, matrice non supportata!";
        return 1;
    }
    

    // Parametri da riga di comando
    const int BASE_N = atoi(argv[1]);
    const float EPSILON  = atof(argv[2]);

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
            matrix[i][j] = 0.0f;
            next_matrix[i][j] = 0.0f;
        }
    }

    // Imposto le sorgenti fisse
    matrix[5][5]      = 5.0f;
    next_matrix[5][5] = 5.0f;
    matrix[20][20]      = 3.0f;
    next_matrix[20][20]  = 3.0f;

    // Imposto variabili per convergenza
    float sum_old  = 0.0;
    float sum_new  = 0.0;
    const int STEPS = 5000; // Limita l'esecuzione nel caso epsilon è eccessivamente piccolo
    bool flag       = 0;

    // Configuro OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(6);

    // Inizio cronometraggio
    vector<double> times;
    double last_time = omp_get_wtime();

    int nth = omp_get_max_threads();
    double* sum_buffers = new double[nth];


    // Regione parallela
   #pragma omp parallel
    {
    int tid = omp_get_thread_num();
    
        /*
        Devo suddividere tutte le colonne in parti uguali per il thread, ovvero:
        -  N^2: mi da il numero totale di celle su cui lavorare
        -  nth: numero thread
        - (N^2/nth): 
            - parte_intera: è il numero base di colonne da assegnare
            - resto: devo suddividerlo equamente tra i vari threads
        - quindi devo ricavarmi il resto e distribuirlo:
            - N/parte_intera mi da quindi un resto che chiamiamo resto_2 e parte_intera_2
            - N - (parte_intera*parte_intera_2) = numero di colonne escluse da assegnare
            - devo quindi creare un loop dove assegno ad ogni thread le righe da cui iniziare e finire e distribuire fino a che non è
                zero il numero di colonne escluse da assegnare
        */

    // Suddivisione 2D in blocchi rettangolari per ogni thread

    // Numero totale di celle da elaborare (escludendo i bordi)
    int internal_N = N - 2; // lavoro effettivo: righe e colonne da 1 a N-2
    int total_cells = internal_N * internal_N;

    // Celle di base per ogni thread
    int base_cells = total_cells / nth;
    int extra_cells = total_cells % nth;

    // Celle assegnate a questo thread
    // Le extra_cells sono sicuramente minori di nth quindi posso assegnarle in ordine
    int my_cells = base_cells + (tid < extra_cells ? 1 : 0); 

    // Calcolo l'indice di partenza (offset) globale per questo thread
    int offset = tid * base_cells + min(tid, extra_cells);

    // Converto l'indice lineare di partenza in coordinate 2D (i,j)
    int start_i = offset / internal_N + 1;       // +1 per bordi
    int start_j = offset % internal_N + 1;

    // Inizio computazione
    for (int step = 0; step < STEPS; ++step) {
        double local_sum_new = 0; // privata

        int i = start_i;
        int j = start_j;

        for (int cell = 0; cell < my_cells; ++cell) {
            next_matrix[i][j] = matrix[i][j]
                + alpha * dt * (
                    matrix[i+1][j] +
                    matrix[i-1][j] +
                    matrix[i][j+1] +
                    matrix[i][j-1] -
                    4.0f * matrix[i][j]
                );
            local_sum_new += next_matrix[i][j];

            // Passa alla prossima cella
            j++;
            if (j == N - 1) { // fuori dai bordi
                j = 1;
                i++;
            }
        }

        sum_buffers[tid] = local_sum_new;

        // sincronizzo tutti i thread
        #pragma omp barrier

        // un solo thread resetta le sorgenti e fa lo swap
        #pragma omp single
        {
            sum_new = 0.0;
            for (int t = 0; t < nth; ++t) {
                sum_new += sum_buffers[t];
            }

            if ((abs(sum_new-sum_old)< EPSILON) && step>100 )
            {
                cout<< "Raggiunta convergenza: " << EPSILON << " in " << step << "steps";
                flag = true;
            }

            sum_old = sum_new;
            // Resetto la variabile per il prossimo step
            sum_new = 0.0;

            next_matrix[5][5]   = 5.0f;
            next_matrix[20][20] = 3.0f;
            // swap dei puntatori

            swap(matrix, next_matrix);
            double now = omp_get_wtime();
            times.push_back(now - last_time);
            last_time = now;
        }
        
        if (flag)
        {break;}
    
        }
    } // fine parallelo

    // Scrivo i tempi di ogni step su file time.txt
    ofstream tf("time_parallel.csv");
    for (double t : times) tf << fixed << setprecision(8) << t << "\n";
    tf.close();

    // Salvo il risultato finale su file
    ofstream fout("output.txt");
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            fout << fixed << setprecision(3) << matrix[i][j] << (j+1 < N-1 ? ' ' : '\n');
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
