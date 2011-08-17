#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "/home/tmarques/cuda/include/cuda.h"
//#include "/home/tmarques/NVIDIA_CUDA_SDK/common/inc/cutil.h"

//VERSION NOTES:
// First version that runs well with 
// split T & phi matrices.
// CalcCP and residual calculation unoptimized

/** Return values:
 * 0 - OK
 * 1 - Can't open one of the data files
 * 2 - Can't allocate CPUMEM for matrix
 * 3 - Can't allocate GPUMEM for matrix
 * 4 - Matrix coordinates can't exceed 512
 */

int dumpMemToFile(char *file, void *input, int size);

int splitT( float *T, float *T_white, float *T_black, int Nx, int Ny, int Nz )
{
    int i, black = 0, white = 0;
    int x, y, z, incr;
    //FIXME: Move to GPU, use OpenMP for these cycles?
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                T_white[white] = T[i*6];
                T_white[white+1] = T[i*6+1];
                T_white[white+2] = T[i*6+2];
                T_white[white+3] = T[i*6+3];
                T_white[white+4] = T[i*6+4];
                T_white[white+5] = T[i*6+5];
                white+=6;
            }
        }
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = 1 - ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                T_black[black] = T[i*6];
                T_black[black+1] = T[i*6+1];
                T_black[black+2] = T[i*6+2];
                T_black[black+3] = T[i*6+3];
                T_black[black+4] = T[i*6+4];
                T_black[black+5] = T[i*6+5];
                black+=6;
            }
        }
    return 0;
}

int splitPhi( float *phi, float *phi_white, float *phi_black, int Nx, int Ny, int Nz )
{
    int i, black = 0, white = 0;
    int x, y, z, incr;
    //FIXME: Move to GPU, use OpenMP for these cycles?
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                phi_white[white]=phi[ i ];
                white++;
            }
        }
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = 1 - ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                phi_black[black]=phi[ i ];
                black++;
            }
        }
    return 0;
}

int joinPhi( float *phi, float *phi_white, float *phi_black, int Nx, int Ny, int Nz )
{
    int i, black = 0, white = 0;
    int x, y, z, incr;
    //FIXME: Move to GPU, use OpenMP for these cycles?
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                phi[i]=phi_white[white];
                white++;
            }
        }
    for ( z = 0; z < Nz; z++ )
        for ( y = 0; y < Ny; y++ )
        {
            incr = 1 - ( ( y % 2 ) + ( z % 2 ) ) % 2;
            for ( x = incr; x < Nx; x+=2 )
            {
                i = y * Nx + z * Nx * Ny + x;
                phi[i]=phi_black[black];
                black++;
            }
        }
    return 0;
}

__global__ void calcCP(float *phi_half, float *Q, int *charged_points, float omega)
{    
    int cp = *(charged_points+blockIdx.x);
    phi_half[ cp / 2 + cp % 1 ] += omega * Q[cp];
}

__global__ void blackSOR( float *phi_black, float *phi_white, float *T, float omega, float lambda )
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'

    // We don't calculate the border values.
    if ( ! ( ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) ) 
                || ( blockIdx.x == (gridDim.x - 1) ) || ( blockIdx.y == (gridDim.y - 1) ) ) )
    {
        unsigned int incr = 1 - ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
        // Black?
        if ( threadIdx.x != ( incr * ( blockDim.x - 1 ) ) )
        {
            unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
//            printf("%d, %d, %d: %lf %lf %lf %lf %lf %lf\n", 
//                        threadIdx.x, blockIdx.x, blockIdx.y, T[ 6 * i ], T[ 6 * i+1 ], T[ 6 * i+2 ], T[ 6 * i+3 ], T[ 6 * i+4 ], T[ 6 * i+5 ]);
            phi_black[i] = omega * (
                    T[ 6 * i ] * phi_white[ i + incr ] 
                    + T[ 6 * i + 1 ] * phi_white[ i - 1 + incr ] 
                    + T[ 6 * i + 2 ] * phi_white[ i + blockDim.x ]
                    + T[ 6 * i + 3 ] * phi_white[ i - blockDim.x ]
                    + T[ 6 * i + 4 ] * phi_white[ i + blockDim.x * gridDim.x ]
                    + T[ 6 * i + 5 ] * phi_white[ i - blockDim.x * gridDim.x ])
                + lambda * phi_black[i];

/*        if ( i == 1956154 )
                printf("%d, %d, %d: %.40f\n, 
                        T: %.40f\n,%.40f\n,%.40f\n,%.40f\n,%.40f\n,%.40f\n
                        phi: ,%.40f\n,%.40f\n,%.40f\n,%.40f\n,%.40f\n,%.40f\n",
                      threadIdx.x, blockIdx.x, blockIdx.y, phi_black[i],
                      T[ 6 * i ],  T[ 6 * i + 1 ], T[ 6 * i + 2 ], T[ 6 * i + 3 ], T[ 6 * i + 4 ], T[ 6 * i + 5 ], 
                      phi_white[ i - 1 + incr ], phi_white[ i + blockDim.x ], phi_white[ i - blockDim.x ], );*/
        }
    }
}

__global__ void whiteSOR( float *phi_white, float *phi_black, float *T, float omega, float lambda )
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'

    // We don't calculate the border values
    if ( ! ( ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) ) 
                || ( blockIdx.x == (gridDim.x - 1) ) || ( blockIdx.y == (gridDim.y - 1) ) ) )
    {
        unsigned int incr = ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
        // White?
        if (  threadIdx.x != ( incr * ( blockDim.x - 1 ) ) )
        {
            unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
            phi_white[i] = omega * (
                    T[ 6 * i  ] * phi_black[ i + incr ] 
                    + T[ 6 * i + 1 ] * phi_black[ i - 1 + incr ] 
                    + T[ 6 * i + 2 ] * phi_black[ i + blockDim.x ]
                    + T[ 6 * i + 3 ] * phi_black[ i - blockDim.x ]
                    + T[ 6 * i + 4 ] * phi_black[ i + blockDim.x * gridDim.x ]
                    + T[ 6 * i + 5 ] * phi_black[ i - blockDim.x * gridDim.x ])
                + lambda * phi_white[i];
        //    printf("WTF? %d: %.20f\n", i, phi_white[i]);
        }
    }
}

int doSOR (int *size, float *phi_host, float *T_host, float *Q_host, int *CBPoints_host, int *CWPoints_host)
{

    float residual = 0, residual_norm2 = 0; 
    float rms_criterion = 1e-6F, max_criterion = 1e-6F;
    float max_residual = max_criterion + 1;
    float rms_change = rms_criterion + 1;
    int max_iterations = 411, iteration = 0;
    int check_after_iterations = 10;
    int i, l = 2080638;
    float spectral_radius=0.99969899654388427734375;

    int number_of_charged_black_points = 5330, number_of_charged_white_points = 5340;

    int Nx = size[0], Ny = size[1], Nz = size[2];
    int Nxy = Nx * Ny;
    int N = Nxy * Nz;

    int Msize = size[0]*size[1]*size[2];

    //Setup CUDA kernel parameters
    if ( ( Nz > 512 ) || ( Ny > 512 ) || ( Nx > 1024 ) )
    {
        printf("Matrix size is too large, must be less than 512\n");
        return 4;
    }
    //Only 2D grids supported at this time
    dim3 gridSize( Ny, Nz, 1 );
    dim3 blockSize( (Nx / 2 ), 1, 1 );
    dim3 gridSizeCP( Ny, Nz, 1 );
    dim3 blockSizeCP( (Nx ), 1, 1 );
    //End setup

    //Check for number_of_charged_***_points
    //to be less than ~2^16:
    if ( ( number_of_charged_black_points > 65530 ) || ( number_of_charged_white_points > 65530 ) )
    {
        printf("One of charged points is too big, must be less than 65500.\n");
        printf("This is a shortcoming of naively implemented code, please address if necessary.\n");
        return 5;
    }

    // Split grid into black and white grids:
    float *phi_black;
    if ( cudaMalloc((void **) &phi_black, sizeof(float)*( Msize/2 )) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for phi_black matrix.\n");
        return 3;
    }
    float *phi_white;
    if ( cudaMalloc((void **) &phi_white, sizeof(float)* ( Msize/2 ) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for phi_white matrix.\n");
        return 3;
    }
    float *phi_black_host;
    if ( cudaMallocHost((void **) &phi_black_host, sizeof(float)*( Msize/2 )) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for phi_black matrix.\n");
        return 3;
    }
    float *phi_white_host;
    if ( cudaMallocHost((void **) &phi_white_host, sizeof(float)*( Msize/2 )) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for phi_white matrix.\n");
        return 3;
    }

    splitPhi ( phi_host, phi_white_host, phi_black_host, Nx, Ny, Nz );
//    joinPhi ( phi_host, phi_black_host, phi_white_host, Msize );
//    dumpMemToFile( "Phi-rejoined.dump", phi_host, Msize*sizeof(float) );
//    cudaFree(phi_host);

    cudaMemcpy(phi_black, phi_black_host, sizeof(float)*( Msize/2 ), cudaMemcpyHostToDevice);
    cudaMemcpy(phi_white, phi_white_host, sizeof(float)*( Msize/2 ), cudaMemcpyHostToDevice);

    float *tmp_phi_black_host;
    if ( cudaMallocHost((void **) &tmp_phi_black_host, sizeof(float)*( Msize/2 )) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for tmp_phi_black matrix.\n");
        return 3;
    }
    float *tmp_phi_white_host;
    if ( cudaMallocHost((void **) &tmp_phi_white_host, sizeof(float)*( Msize/2 )) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for tmp_phi_white matrix.\n");
        return 3;
    }

    float *T_white;
    if ( cudaMalloc( (void **) &T_white, (sizeof(float)*3*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for T_white matrix.\n");
        return 3;
    }
    float *T_black;
    if ( cudaMalloc( (void **) &T_black, (sizeof(float)*3*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for T_black matrix.\n");
        return 3;
    }
    float *T_white_host;
    if ( cudaMallocHost( (void **) &T_white_host, (sizeof(float)*3*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for T_white_host matrix.\n");
        return 3;
    }
    float *T_black_host;
    if ( cudaMallocHost( (void **) &T_black_host, (sizeof(float)*3*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for T_black_host matrix.\n");
        return 3;
    }
    splitT( T_host, T_white_host, T_black_host, size[0], size[1], size[2] );
    cudaMemcpy( T_white, T_white_host, (sizeof(float)*3*Msize), cudaMemcpyHostToDevice );
    cudaMemcpy( T_black, T_black_host, (sizeof(float)*3*Msize), cudaMemcpyHostToDevice );
    cudaFreeHost( T_black_host );
    cudaFreeHost( T_white_host );

    float *tmp_phi_host;
    if ( cudaMallocHost( (void **) &tmp_phi_host, (sizeof(float)*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for tmp_phi_host matrix.\n");
        return 2;
    }

    float *Q;
    if ( cudaMalloc( (void **) &Q, (sizeof(float)*Msize) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for Q matrix.\n");
        return 3;
    }
    cudaMemcpy( Q, Q_host, (sizeof(float)*Msize), cudaMemcpyHostToDevice );

    int *charged_black_points;
    if ( cudaMalloc( (void **) &charged_black_points, (sizeof(int)*number_of_charged_black_points) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for charged_black_points.\n");
        return 3;
    }
    cudaMemcpy( charged_black_points, CBPoints_host, (sizeof(int)*number_of_charged_black_points), cudaMemcpyHostToDevice );

    int *charged_white_points;
    if ( cudaMalloc( (void **) &charged_white_points, (sizeof(int)*number_of_charged_white_points) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for charged_black_points.\n");
        return 3;
    }
    cudaMemcpy( charged_white_points, CWPoints_host, (sizeof(int)*number_of_charged_white_points), cudaMemcpyHostToDevice );

    // FIXME: Not needed?
    cudaThreadSynchronize();

    //BALL_START
    float omega = 1, lambda = 1 - omega;

    //dumpMemToFile("output.dump", phi_host, sizeof(float)*Msize);

    while ( (iteration < max_iterations)  && ((max_residual > max_criterion) || (rms_change > rms_criterion)) )
    {

        // first half of Gauss-Seidel iteration (black fields only)
        blackSOR<<< gridSize, blockSize >>> ( phi_black, phi_white, T_black, omega, lambda );
        cudaThreadSynchronize();
/*    printf("iteration: %d\n", iteration);
    if (iteration == 4 )
    {
        cudaMemcpy( phi_black_host, phi_black, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
        cudaMemcpy( phi_white_host, phi_white, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
        int teste;
        joinPhi( phi_host, phi_white_host, phi_black_host,  Nx, Ny, Nz);
        for (teste = 0; teste < Msize; teste++)
            printf("%d: %.40f\n", teste, phi_host[teste]);

        //dumpMemToFile( "Phi-run@65.dump", phi_host, Msize*sizeof(float) );
        return 0;
    }*/
        calcCP<<< number_of_charged_black_points, 1 >>> ( phi_black, Q, charged_black_points, omega );
        cudaThreadSynchronize();

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
        whiteSOR<<< gridSize, blockSize >>> ( phi_white, phi_black, T_white, omega, lambda );
        cudaThreadSynchronize();
        calcCP<<< number_of_charged_white_points, 1 >>> ( phi_white, Q, charged_white_points, omega );
        cudaThreadSynchronize();

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
                //FIXME: Both these copies can be done right after kernel execution, and "tmp" also can.
                cudaMemcpy( phi_black_host, phi_black, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
                cudaMemcpy( phi_white_host, phi_white, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
                joinPhi( phi_host, phi_white_host, phi_black_host,  Nx, Ny, Nz);

                max_residual = 0;
                residual_norm2 = 0;

                // sum up all squared changes in the phi array since
                // the last iteration
/*                for (i = 1; i < ( (N/2) ); i++)
                {
                    residual = fabs(tmp_phi_white_host[i] - phi_white_host[i]);
                    if (max_residual < residual)
                        max_residual=residual;
                    residual_norm2 += residual * residual;
                }
                for (i = 0; i < ( (N/2) - 1 ); i++)
                {
                    residual = fabs(tmp_phi_black_host[i] - phi_black_host[i]);
                    if (max_residual < residual)
                        max_residual=residual;
                    residual_norm2 += residual * residual;
                }*/
                for (i = 1; i < (N - 1); i++)
                {
                    residual = fabs(tmp_phi_host[i] - phi_host[i]);
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
            cudaMemcpy( tmp_phi_black_host, phi_black, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
            cudaMemcpy( tmp_phi_white_host, phi_white, sizeof(float)*( Msize/2 ), cudaMemcpyDeviceToHost );
            joinPhi(tmp_phi_host, tmp_phi_white_host, tmp_phi_black_host, size[0], size[1], size[2] );
        }

        if ( (iteration % 10) == 0)
        {
            printf("Iteration number: %d\n", iteration);
        }
        iteration++;
    }
    //BALL_END

    if ((rms_change <= rms_criterion) && (max_residual <= max_criterion))
    {
        printf("Converged - iteration: %d\n", iteration);
    }
    else
    {
        printf("Not converged - iteration: %d\n", iteration);
    }

    //FIXME: must free after rejoin
    cudaFree(phi_black);
    cudaFree(phi_white);
    cudaFreeHost(phi_black_host);
    cudaFreeHost(phi_white_host);
    cudaFreeHost(tmp_phi_black_host);
    cudaFreeHost(tmp_phi_white_host);
    cudaFree(Q);
    cudaFree(charged_black_points);
    cudaFree(charged_white_points);
    cudaFreeHost(tmp_phi_host);
//    cudaFreeHost(phi_host);

    return 0;
}

int loadMemFromFile(char *file, void *output, int size)
{
    int fRead;

    fRead = open(file, O_RDONLY);
    if ( fRead == -1 )
    {
        printf("Error opening %s file.\n", size);
        return 1;
    }

    read(fRead, output, size);

    close(fRead);

    return 0;
}

int dumpMemToFile(char *file, void *input, int size)
{
    int fWrite;

    fWrite = open(file, O_WRONLY|O_CREAT, 0666);

    write(fWrite, input, size);

    close(fWrite);

    return 0;    
}

/*int checkConvergence(float rms_change, float max_residual, int iteration)
  {
  float rms_criterion = 1e-6F, max_criterion = 1e-6F;

  if ((rms_change <= rms_criterion) && (max_residual <= max_criterion))
  {
  printf("Converged after %d iterations.\n", iteration);
  }
  else
  {
  printf("Not converged! (after %d iterations.\n)", iteration);
  }
  return 0;
  }*/

int main(void)
{
    int size[3] = {128, 128, 128};
    int Msize = size[0]*size[1]*size[2];

    // Calculate the number of elements of the matrix,
    // which corresponds to the number of lines yet to read from file.
    float *phi;
    if ( cudaMallocHost((void **) &phi, sizeof(float)*Msize) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for matrix.\n");
        return 2;    
    }
    loadMemFromFile( "Phi.dump", phi, (sizeof(float)*Msize) );

    //FIXME: Check for allocation Errors
    float *Tmatrix;
    cudaMallocHost((void **) &Tmatrix, (sizeof(float)*6*Msize) );    
    loadMemFromFile( "T.dump", Tmatrix, (sizeof(float)*6*Msize) );
    float *Qmatrix;
    cudaMallocHost((void **) &Qmatrix, (sizeof(float)*Msize) );
    loadMemFromFile( "Q.dump", Qmatrix, (sizeof(float)*Msize) );
    int *CBPoints;
    cudaMallocHost((void **) &CBPoints, (sizeof(int)*5330) );
    loadMemFromFile( "Black_points.dump", CBPoints, (sizeof(int)*5330) );
    int *CWPoints;
    cudaMallocHost((void **) &CWPoints, ( sizeof(int)*5340) );
    loadMemFromFile( "White_points.dump", CWPoints, (sizeof(int)*5340) );

    cudaThreadSynchronize();

    //float rms_change = 0, max_residual = 0; int num_iteration = 0;
    //float *rms_change, *max_residual; int *num_iteration;
    //float *rms_change_dev, *max_residual_dev; int *num_iteration_dev;
    //cudaMalloc( (void **) &rms_change_dev, (sizeof(float)) );
    //cudaMalloc( (void **) &max_residual_dev, (sizeof(float)) );
    //cudaMalloc( (void **) &num_iteration_dev, (sizeof(int)) );
    //cudaMallocHost( (void **) &rms_change, (sizeof(float)) );
    //cudaMallocHost( (void **) &max_residual, (sizeof(float)) );
    //cudaMallocHost( (void **) &num_iteration, (sizeof(int)) );
    //*rms_change = 2;
    //*max_residual = 4;
    //*num_iteration = 6;
    //printf("%.20f - %.20f, %d\n", *rms_change, *max_residual, *num_iteration);
    //checkConvergence(*rms_change, *max_residual, *num_iteration);

    doSOR(size, phi, Tmatrix, Qmatrix, CBPoints, CWPoints);
    printf("Finished doSOR.\n");

    cudaFreeHost(phi);
    cudaFreeHost(Tmatrix);
    cudaFreeHost(Qmatrix);
    cudaFreeHost(CBPoints);
    cudaFreeHost(CWPoints);

    return 0;
}
