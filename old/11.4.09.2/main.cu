#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "/home/cudauser/cuda/include/cuda.h"


/** Return values:
* 0 - OK
* 1 - Can't open matrix file;
* 2 - Can't allocate CPUMEM for matrix
* 3 - Can't allocate GPUMEM for matrix
*/

//Kernel definition
__global__ void doSOR(int Nx, int Ny, int Nz, float *matrix, float *Tmatrix, int *Qmatrix, int *charged_black_points, int *charged_white_points, float *tmp_phi, float *rms_change, float *max_residual, int *iteration)
{
    float *T, *phi;
    int *Q;
    float residual, residual_norm2; // Doesn't need to be initialized
    float rms_criterion = 1e-5F, max_criterion = 1e-4F;
    int max_iterations = 1500, black, white;
    int check_after_iterations = 10;
    int x, y, z, Nxy, N; 
    int i, l = 0;
    float spectral_radius=0.99969899654388427734375;
    int number_of_charged_black_points = 5330, number_of_charged_white_points = 5340;
    
    *max_residual = max_criterion + 1;
    *rms_change = rms_criterion + 1;
    *iteration = 0;

    Nxy = Nx * Ny;
    N = Nxy * Nz;

    phi = matrix; //FIXME
    T = Tmatrix;  //FIXME
    Q = Qmatrix;  //FIXME

    //BALL_START
    float omega = 1, lambda;
    lambda = 1 - omega;

    while ((*iteration < max_iterations)  && ((*max_residual > max_criterion) || (*rms_change > rms_criterion)))
    {

      // first half of Gauss-Seidel iteration (black fields only)
      for (z = 1; z < (int)(Nx - 1); z++)
      {
        for (y = 1; y < (int)(Nx - 1); y++)
        {
          black = ((y % 2) + (z % 2)) % 2;
          i = y * Nx + z * Nxy + 1 + black;
          for (x = 1 + black; x < (int)(Nx - 1); x += 2)
          {
            phi[i] = omega * (T[6 * i    ] * phi[i + 1 ]
                              + T[6 * i + 1] * phi[i - 1 ]
                              + T[6 * i + 2] * phi[i + Nx ]
                              + T[6 * i + 3] * phi[i - Nx ]
                              + T[6 * i + 4] * phi[i + Nxy]
                              + T[6 * i + 5] * phi[i - Nxy])
                     + lambda * phi[i];
            i += 2;
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
      for (z = 1; z < (int)(Nx - 1); z++)
      {
        for (y = 1; y < (int)(Nx - 1); y++)
        {
          white = 1 - ((y % 2) + (z % 2)) % 2;
          i = y * Nx + z * Nxy + 1 + white;
          for (x = 1 + white; x < (int)(Nx - 1); x += 2)
          {
            phi[i] = omega * (T[6 * i    ] * phi[i + 1 ]
                              + T[6 * i + 1] * phi[i - 1 ]
                              + T[6 * i + 2] * phi[i + Nx ]
                              + T[6 * i + 3] * phi[i - Nx ]
                              + T[6 * i + 4] * phi[i + Nxy]
                              + T[6 * i + 5] * phi[i - Nxy])
                     + lambda * phi[i];
            i += 2;
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
      if ((*iteration % check_after_iterations) == 0)
      {
        if (*iteration > 0)
        {
          *max_residual = 0;
          residual_norm2 = 0;

          // sum up all squared changes in the phi array since
          // the last iteration
          for (i = 1; i < N - 1; i++)
          {
            residual = fabs(tmp_phi[i] - phi[i]);
            if (*max_residual < residual)
                *max_residual=residual;
            //max_residual = std::max(residual, max_residual); //FIXME, fixed with above?
            residual_norm2 += residual * residual;
          }

          *rms_change = sqrt(residual_norm2 / (float)N);

          //if (verbosity > 0)
          //{
          //  Log.info(1) << "Iteration " << iteration << " RMS: "
          //    << rms_change << "   MAX: " << max_residual << endl;
          //}
        }
      }
      
      if (((*iteration + 1) % check_after_iterations) == 0)
      {
        // save the actual settings phi
        //memcpy( tmp_phi, phi, N * sizeof(phi[0]) );
        for (i = 0; i < N; i++)
        {
            tmp_phi[i] = phi[i];        
        }

      }

      // increase iteration count
      *iteration++;
//      if ( (iteration % 10) == 0)
//          printf("Iteration number: %d\n", iteration);
    }
    //BALL_END

    // Return convergence values
}

/*
void calcChargedPoints()
{
    
    // Now, find out which grid points are charged and store them (or,
    // more precisely, their indices) into two arrays
    int         number_of_charged_black_points;
    int         number_of_charged_white_points;

    // pointer to array to hold the indices
    int*        charged_black_points;
    int*        charged_white_points;


    // get the number of charged grid_points
    number_of_charged_black_points = 0;
    number_of_charged_white_points = 0;

    for (k = 1; k < Nz - 1; k++)
    {
      for (j = 1; j < Ny - 1; j++)
      {
        for (i = 1; i < Nx - 1; i++)
        {
          l = i + j * Nx + k * Nxy;
          if (Q[l] != 0.0)
          {
            if ((i + j + k) % 2 == 1)
            {
              number_of_charged_black_points++;
            }
            else
            {
              number_of_charged_white_points++;
            }
          }
        }
      }
    }

        charged_black_points = new int[number_of_charged_black_points];
    if (charged_black_points == 0)
    {
      throw Exception::OutOfMemory(__FILE__, __LINE__, number_of_charged_black_points * (Size)sizeof(int));
    }

    charged_white_points = new int[number_of_charged_white_points];
    if (charged_white_points == 0)
    {
      throw Exception::OutOfMemory(__FILE__, __LINE__, number_of_charged_white_points * (Size)sizeof(int));
    }


    number_of_charged_black_points = 0;
    number_of_charged_white_points = 0;

    for (k = 1; k < Nz - 1; k++)
    {
      for (j = 1; j < Ny - 1; j++)
      {
        for (i = 1; i < Nx - 1; i++)
        {
          l = i + j * Nx + k * Nxy;
          if (Q[l] != 0.0)
          {
            if ((i + j + k) % 2 == 1)
            {
              charged_black_points[number_of_charged_black_points++] = (int)l;
            }
            else
            {
              charged_white_points[number_of_charged_white_points++] = (int)l;
            }
          }
        }
      }
    }

    
}
*/

int loadMemFromFile(char *file, void *output, int size)
{
    int fRead;

    fRead = open(file, O_RDONLY);
    
    read(fRead, output, size);

    close(fRead);
    
    return 0;
}

int checkConvergence(float rms_change, float max_residual, int iteration)
{
    float rms_criterion = 1e-5F, max_criterion = 1e-4F;

    if ((rms_change <= rms_criterion) && (max_residual <= max_criterion))
    {
        printf("Converged after %d iterations", iteration);
    }
    else
    {
        printf("Not converged! (after %d iterations)", iteration);
    }
    return 0;
}

int main(void)
{
    int size[3] = {127, 127, 127};  // Real size is 128x128x128, this is for indices
                                    // and it's what the PB program prints
    int Msize = (size[0]+1)*(size[1]+1)*(size[2]+1);

    // Calculate the number of elements of the matrix,
    // which corresponds to the number of lines yet to read from file.
    float *h_matrix;
    //if ( (matrix = (float *) malloc(sizeof(float) * nlines)) == NULL )
    if ( cudaMallocHost((void **) &h_matrix, sizeof(float)*Msize) 
                                == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for matrix.\n");
        return 2;    
    }

    float *d_matrix;
    //if ( (resmatrix = (float *) malloc(sizeof(float) * nlines)) == NULL )
    if ( cudaMalloc((void **) &d_matrix, sizeof(float)*Msize) 
                                == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM matrix.\n");
        return 3;
    }
    loadMemFromFile( "Phi.dump", h_matrix, sizeof(float)*Msize );
    //printf("%f\n", *(h_matrix+1000));
    cudaMemcpy(d_matrix,h_matrix, sizeof(float)*Msize, cudaMemcpyHostToDevice);

    float *Tmatrix;
    float *Tmatrix_dev;
    cudaMallocHost((void **) &Tmatrix, (sizeof(float)*6*Msize) );    
    loadMemFromFile( "T.dump", Tmatrix, (sizeof(float)*6*Msize) );
    cudaMalloc( (void **) &Tmatrix_dev, (sizeof(float)*6*Msize) ); 
    cudaMemcpy( Tmatrix_dev, Tmatrix, (sizeof(float)*6*Msize), cudaMemcpyHostToDevice );
    cudaFreeHost(Tmatrix);

    int *Qmatrix;
    int *Qmatrix_dev;
    cudaMallocHost((void **) &Qmatrix, (sizeof(int)*Msize) );
    loadMemFromFile( "Q.dump", Qmatrix, (sizeof(int)*Msize) );
    cudaMalloc( (void **) &Qmatrix_dev, (sizeof(int)*Msize) );
    cudaMemcpy( Qmatrix_dev, Qmatrix, (sizeof(int)*Msize), cudaMemcpyHostToDevice );
    cudaFreeHost(Qmatrix);

    int *CBPoints;
    int *CBPoints_dev;
    cudaMallocHost((void **) &CBPoints, (sizeof(int)*5330) );
    loadMemFromFile( "Black_points.dump", CBPoints, (sizeof(int)*5330) );
    cudaMalloc( (void **) &CBPoints_dev, (sizeof(int)*5330) );
    cudaMemcpy( CBPoints_dev, CBPoints, (sizeof(int)*5330), cudaMemcpyHostToDevice );
    cudaFreeHost(CBPoints);

    int *CWPoints;
    int *CWPoints_dev;
    cudaMallocHost((void **) &CWPoints, ( sizeof(int)*5340) );
    loadMemFromFile( "White_points.dump", CWPoints, (sizeof(int)*5340) );
    cudaMalloc( (void **) &CWPoints_dev, (sizeof(int)*5340) );
    cudaMemcpy( CWPoints_dev, CWPoints, (sizeof(int)*5340), cudaMemcpyHostToDevice );
    cudaFreeHost(CWPoints);

    float *tmp_phi_dev;
    cudaMalloc( (void **) &tmp_phi_dev, (sizeof(float)*Msize) );
    
    float *rms_change_dev, *max_residual_dev, rms_change, max_residual; 
    int *iteration_dev, iteration;
    cudaMalloc( (void **) &rms_change_dev, (sizeof(float)) );
    cudaMalloc( (void **) &max_residual_dev, (sizeof(float)) );
    cudaMalloc( (void **) &iteration_dev, (sizeof(int)) );
	//Call kernel
	doSOR <<<1, 1>>> (size[0], size[1], size[2], d_matrix, Tmatrix_dev, Qmatrix_dev, CBPoints_dev, CWPoints_dev, tmp_phi_dev, rms_change_dev, max_residual_dev, iteration_dev);
    cudaMemcpy( h_matrix, d_matrix, (sizeof(float)*Msize), cudaMemcpyDeviceToHost );
    cudaMemcpy( &rms_change, rms_change_dev, (sizeof(float)), cudaMemcpyDeviceToHost );
    cudaMemcpy( &max_residual, max_residual_dev, (sizeof(float)), cudaMemcpyDeviceToHost );
    cudaMemcpy( &iteration, iteration_dev, (sizeof(int)), cudaMemcpyDeviceToHost );
    printf("%f - %f, %d\n", rms_change, max_residual, iteration);
    checkConvergence(rms_change, max_residual, iteration);

    cudaFree(rms_change_dev);
    cudaFree(max_residual_dev);
    cudaFree(iteration_dev);
    cudaFree(tmp_phi_dev);
    cudaFree(Tmatrix_dev);
    cudaFree(Qmatrix_dev);
    cudaFree(CBPoints_dev);
    cudaFree(CWPoints_dev);
    cudaFree(d_matrix);
    cudaFreeHost(h_matrix);

//	printf("Result:\t%d\n", res);

	return 0;
}
