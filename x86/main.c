#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/** Return values:
* 0 - OK
* 1 - Can't open matrix file;
* 2 - Can't allocate CPUMEM for matrix
* 3 - Can't allocate GPUMEM for matrix
*/

//Kernel definition
void doSOR(int Nx, int Ny, int Nz, float *matrix, float *Tmatrix, float *Qmatrix, int *charged_black_points, int *charged_white_points, float *tmp_phi)
{
    float *T, *phi;
    float *Q;
    float residual = 0, residual_norm2 = 0; // Doesn't need to be initialized
    float rms_criterion = 1e-6F, max_criterion = 1e-6F;
    float max_residual = max_criterion + 1;
    float rms_change = rms_criterion + 1;
    int max_iterations = 750, iteration = 0, black, white;
    int check_after_iterations = 10;
    int x, y, z, Nxy, N; 
    int i, l = 2080638;
    float spectral_radius=0.99969899654388427734375;

    int number_of_charged_black_points = 5330, number_of_charged_white_points = 5340;

    Nxy = Nx * Ny;
    N = Nxy * Nz;

    phi = matrix; //FIXME
    T = Tmatrix;  //FIXME
    Q = Qmatrix;  //FIXME

    printf("RMS_Crit = %.20f - Max_crit = %.20f\n", rms_criterion, max_criterion);

    //BALL_START
    float omega = 1, lambda = 1 - omega;

    while ((iteration < max_iterations)  && ((max_residual > max_criterion) || (rms_change > rms_criterion)))
    {

      // first half of Gauss-Seidel iteration (black fields only)
      for (z = 1; z < (Nx - 1); z++)
      {
        for (y = 1; y < (Nx - 1); y++)
        {
          black = ((y % 2) + (z % 2)) % 2;
          i = y * Nx + z * Nxy + 1 + black;
          for (x = 1 + black; x < (Nx - 1); x += 2)
          {
            phi[i] = omega * (T[6 * i    ] * phi[i + 1 ]
                              + T[6 * i + 1] * phi[i - 1 ]
                              + T[6 * i + 2] * phi[i + Nx ]
                              + T[6 * i + 3] * phi[i - Nx ]
                              + T[6 * i + 4] * phi[i + Nxy]
                              + T[6 * i + 5] * phi[i - Nxy])
                     + lambda * phi[i];
            i += 2;
            //if(i==20676)
           // {
            //      printf("Phi(i)= %.20f - ", phi[i]);
            //      printf("Iteration= %d\n", (int)iteration); 
           // }

          }
        }
      }

      int* charge_pointer;
      charge_pointer = charged_black_points;
      for (charge_pointer = charged_black_points;
           charge_pointer < &charged_black_points[number_of_charged_black_points];
           charge_pointer++)
      {
        phi[*charge_pointer] += omega * Q[*charge_pointer];
      }

      // Chebyshev acceleration: omega approaches its
      // optimal value asymptotically. This usually gives
      // better convergence for the first few iterations
      if (spectral_radius != 0.0)
      {
        if (l == 0)
        {
          omega = 1 / (1 - spectral_radius / 2);
        }
        else
        {
          omega = 1 / (1 - spectral_radius * omega / 4);
        }
        lambda = 1 - omega;
      }

      // second half of Gauss-Seidel iteration (white fields only)
      for (z = 1; z < (Nx - 1); z++)
      {
        for (y = 1; y < (Nx - 1); y++)
        {
          white = 1 - ((y % 2) + (z % 2)) % 2;
          i = y * Nx + z * Nxy + 1 + white;
          for (x = 1 + white; x < (Nx - 1); x += 2)
          {
            phi[i] = omega * (T[6 * i    ] * phi[i + 1 ]
                              + T[6 * i + 1] * phi[i - 1 ]
                              + T[6 * i + 2] * phi[i + Nx ]
                              + T[6 * i + 3] * phi[i - Nx ]
                              + T[6 * i + 4] * phi[i + Nxy]
                              + T[6 * i + 5] * phi[i - Nxy])
                     + lambda * phi[i];
            i += 2;
//            if(i==20676)
  //          {
    //              printf("Phi(i)= %.20f - ", phi[i]);
      //            printf("Iteration= %d\n", (int)iteration);
        //    }
          }
        }
      }

      charge_pointer = charged_white_points;
      for (charge_pointer = charged_white_points;
           charge_pointer < &charged_white_points[number_of_charged_white_points];
           charge_pointer++)
      {
        phi[*charge_pointer] += omega * Q[*charge_pointer];
      }

      // Chebyshev acceleration for the second Gauss-Seidel step
      if (spectral_radius != 0.0)
      {
        omega = 1 / (1 - spectral_radius * omega / 4);
        lambda = 1 - omega;
      }

      // calculate the gradient every check_after_iterations
      if ((iteration % check_after_iterations) == 0)
      {
        if (iteration > 0)
        {
          max_residual = 0;
          residual_norm2 = 0;

          // sum up all squared changes in the phi array since
          // the last iteration
          for (i = 1; i < (N - 1); i++)
          {
            residual = fabs(tmp_phi[i] - phi[i]);
            if (max_residual < residual)
                max_residual=residual;
            residual_norm2 += residual * residual;
          }
          printf("Res_norm2: %.20f\n", residual_norm2);
          rms_change = sqrt(residual_norm2 / (float)N);
          printf("Max Residual = %.20f\n", max_residual);
          printf("RMS_Change: %.20f\n", rms_change);
        }
      }
      
      if (((iteration + 1) % check_after_iterations) == 0)
      {
        // save the actual settings phi
        for (i = 1; i < N-1; i++)
        {
            tmp_phi[i] = phi[i];
        }
        //memcpy( tmp_phi, phi, (N * sizeof(phi[0])) );
      }

      if ( (iteration % 10) == 0)
      {
          printf("Iteration number: %d\n", iteration);
          printf("phi(teste): %.20f\n", phi[100550]);
      }
       
      iteration++;
    }
    //BALL_END
 
    //  printf("%f\n", *(matrix+10));
     if ((rms_change <= rms_criterion) && (max_residual <= max_criterion))
     {
       printf("Converged - iteration: %d\n", iteration);
     }
     else
     {
        printf("Not converged - iteration: %d\n", iteration);
     } 
}

int loadMemFromFile(char *file, void *output, int size)
{
    int fRead;

    fRead = open(file, O_RDONLY);
    
    read(fRead, output, size);

    close(fRead);
    
    return 0;
}

int main(void)
{
    int size[] = {128, 128, 128};
    int Msize  = size[0]*size[1]*size[2];
    
    // Calculate the number of elements of the matrix,
    // which matches to the number of lines yet to read from file.
   // float *h_matrix;
   // h_matrix = malloc( sizeof(float)*Msize ); 
    
    float *h_matrix;
    h_matrix = malloc( sizeof(float)*Msize ); 
    loadMemFromFile( "Phi.dump", h_matrix, sizeof(float)*Msize );
    //printf("%f\n", *(h_matrix+1000));
    //cudaMemcpy(d_matrix,h_matrix, sizeof(float)*Msize, cudaMemcpyHostToDevice);

    //printf("phi(before): %.20f\n", h_matrix[100550]);

    float *Tmatrix;
    //float *Tmatrix_dev;
    //cudaMallocHost((void **) &Tmatrix, (sizeof(float)*6*Msize) );    
    Tmatrix = malloc( sizeof(float)*6*Msize );
    loadMemFromFile( "T.dump", Tmatrix, (sizeof(float)*6*Msize) );
    //cudaMalloc( (void **) &Tmatrix_dev, (sizeof(float)*6*Msize) ); 
    //cudaMemcpy( Tmatrix_dev, Tmatrix, (sizeof(float)*6*Msize), cudaMemcpyHostToDevice );
    //cudaFreeHost(Tmatrix);

    float *Qmatrix;
    //int *Qmatrix_dev;
    //cudaMallocHost((void **) &Qmatrix, (sizeof(int)*Msize) );
    Qmatrix = malloc( sizeof(float)*Msize );
    loadMemFromFile( "Q.dump", Qmatrix, (sizeof(float)*Msize) );
    //cudaMalloc( (void **) &Qmatrix_dev, (sizeof(int)*Msize) );
    //cudaMemcpy( Qmatrix_dev, Qmatrix, (sizeof(int)*Msize), cudaMemcpyHostToDevice );
    //cudaFreeHost(Qmatrix);

    int *CBPoints;
    //int *CBPoints_dev;
    //cudaMallocHost((void **) &CBPoints, (sizeof(int)*5330) );
    CBPoints = malloc( sizeof(int)*5330 );
    loadMemFromFile( "Black_points.dump", CBPoints, (sizeof(int)*5330) );
    //cudaMalloc( (void **) &CBPoints_dev, (sizeof(int)*5330) );
    //cudaMemcpy( CBPoints_dev, CBPoints, (sizeof(int)*5330), cudaMemcpyHostToDevice );
    //cudaFreeHost(CBPoints);

    int *CWPoints;
    //int *CWPoints_dev;
    //cudaMallocHost((void **) &CWPoints, ( sizeof(int)*5340) );
    CWPoints = malloc( sizeof(int)*5340 );
    loadMemFromFile( "White_points.dump", CWPoints, (sizeof(int)*5340) );
    //cudaMalloc( (void **) &CWPoints_dev, (sizeof(int)*5340) );
    //cudaMemcpy( CWPoints_dev, CWPoints, (sizeof(int)*5340), cudaMemcpyHostToDevice );
    //cudaFreeHost(CWPoints);

    float *tmp_phi;
    tmp_phi = malloc( sizeof(float)*Msize);
    //cudaMalloc( (void **) &tmp_phi_dev, (sizeof(float)*Msize) );

    int i=0;
    for(i=0; i<Msize; i++)
        tmp_phi[i]=0;

	//Call kernel
//    cudaThreadSynchronize();
//    doSOR <<<1, 1>>> (size[0], size[1], size[2], d_matrix, Tmatrix_dev, Qmatrix_dev, CBPoints_dev, CWPoints_dev, tmp_phi_dev);
//    cudaMemcpy( h_matrix, d_matrix, (sizeof(float)*Msize), cudaMemcpyDeviceToHost );
    doSOR(size[0], size[1], size[2], h_matrix, Tmatrix, Qmatrix, CBPoints, CWPoints, tmp_phi);

    free(Tmatrix);
    free(Qmatrix);
    free(CBPoints);
    free(CWPoints);
    free(h_matrix);
    free(tmp_phi);
    //cudaFree(tmp_phi_dev);
    //cudaFree(Tmatrix_dev);
    //cudaFree(Qmatrix_dev);
    //cudaFree(CBPoints_dev);
    //cudaFree(CWPoints_dev);
    //cudaFree(d_matrix);
    //cudaFreeHost(h_matrix);
	
    return 0;
}
