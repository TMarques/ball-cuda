#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "/home/tmarques/cuda/include/cuda.h"
//#include "/home/tmarques/NVIDIA_CUDA_SDK/common/inc/cutil.h"

//VERSIONS NOTES:
// This version simply takes advantage of some memory coalescing
// by remapping 'x', 'y' and 'z' and how they relate to 
// blockId and threadId, since the threads inside a block should
// map to 'x' in order to share nearby values in a "warp".
// This was not possible when threadId.x was mapped to 'z',
// as each thread in the same block would be loading only
// different 'x,y' lines in memory. We want to load the same 'x,y,z'
// line and perform operations just changing 'x', as this is sequential 
// layout of the data in memory.
// It is now about the same speed as the CPU in an 128x128x128 matrix

/** Return values:
 * 0 - OK
 * 1 - Can't open one of the data files
 * 2 - Can't allocate CPUMEM for matrix
 * 3 - Can't allocate GPUMEM for matrix
 * 4 - Matrix coordinates can't exceed 512
 */

int dumpMemToFile(char *file, void *input, int size);

__global__ void blackSOR(float *phi, float *T, float omega, float lambda)
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'
    
    // We don't calculate the border values.
    if ( ! ( ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) || ( threadIdx.x == 0 ) ) 
          || ( blockIdx.x == 127 ) || ( blockIdx.y == 127 ) || ( threadIdx.x == 127 ) ) )
    {
        unsigned int incr = ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
        // Black?
        if ( ( threadIdx.x % 2 )  != incr )
        {
            unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
            phi[i] = omega * (
                  T[6 * i    ] * phi[i + 1 ] 
                + T[6 * i + 1] * phi[i - 1 ] 
                + T[6 * i + 2] * phi[i + gridDim.x ]
                + T[6 * i + 3] * phi[i - gridDim.x ]
                + T[6 * i + 4] * phi[i + gridDim.x * gridDim.y ]
                + T[6 * i + 5] * phi[i - gridDim.x * gridDim.y ])
                + lambda * phi[i];
        }
    }
}

__global__ void redSOR(float *phi, float *T, float omega, float lambda)
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'

    // We don't calculate the border values
    if  ( ! ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) || ( threadIdx.x == 0 )
       || ( blockIdx.x == 127 ) || ( blockIdx.y == 127 ) || ( threadIdx.x == 127 ) ) )
    {
        unsigned int incr = 1 - ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
        // Red?
        if ( (  threadIdx.x % 2 ) != incr )
        {
            unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
            phi[i] = omega * (
                  T[6 * i    ] * phi[i + 1 ]
                + T[6 * i + 1] * phi[i - 1 ]
                + T[6 * i + 2] * phi[i + gridDim.x ]
                + T[6 * i + 3] * phi[i - gridDim.x ]
                + T[6 * i + 4] * phi[i + gridDim.x * gridDim.y ]
                + T[6 * i + 5] * phi[i - gridDim.x * gridDim.y ])
                + lambda * phi[i];
        }
    }
}

int doSOR (int *size, float *phi_host, float *T_host, float *Q, int *CBPoints_host, int *CWPoints_host)
{

    float residual = 0, residual_norm2 = 0; 
    float rms_criterion = 4e-6F, max_criterion = 4e-6F;
    float max_residual = max_criterion + 1;
    float rms_change = rms_criterion + 1;
    int max_iterations = 750, iteration = 0;
    int check_after_iterations = 10;
    int i, l = 2080638;
    float spectral_radius=0.99969899654388427734375;
    int *charged_black_points = CBPoints_host;
    int *charged_white_points = CWPoints_host;

    int number_of_charged_black_points = 5330, number_of_charged_white_points = 5340;
    
    int Nx = size[0], Ny = size[1], Nz = size[2];
    int Nxy = Nx * Ny;
    int N = Nxy * Nz;

    int Msize = size[0]*size[1]*size[2];

    float *phi;
    if ( cudaMalloc((void **) &phi, sizeof(float)*Msize) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM phi matrix.\n");
        return 3;
    }
    cudaMemcpy(phi, phi_host, sizeof(float)*Msize, cudaMemcpyHostToDevice);
    
    float *T;
    if ( cudaMalloc( (void **) &T, (sizeof(float)*6*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM T matrix.\n");
        return 3;
    }
    cudaMemcpy( T, T_host, (sizeof(float)*6*Msize), cudaMemcpyHostToDevice ); 

    float *tmp_phi_host;
    if ( cudaMallocHost( (void **) &tmp_phi_host, (sizeof(float)*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
       fprintf(stderr, "Can't allocate CPUMEM tmp_phi_host matrix.\n");
        return 2;
    }

    //Setup CUDA kernel parameters
    if ( Nz > 512 )
    {
        printf("Matrix size is too large, must be less than 512\n");
        return 4; 
    }
    //Only 2D grids supported at this time
    dim3 gridSize( Nx, Ny, 1 );
    dim3 blockSize( Nz, 1, 1 );
    //End setup
    cudaThreadSynchronize();

    //BALL_START
    float omega = 1, lambda = 1 - omega;
    
    //dumpMemToFile("output.dump", phi_host, sizeof(float)*Msize);

    while ((iteration < max_iterations)  && ((max_residual > max_criterion) || (rms_change > rms_criterion)))
    {

        // first half of Gauss-Seidel iteration (black fields only)
        blackSOR<<< gridSize, blockSize >>> ( phi, T, omega, lambda );
        cudaThreadSynchronize();

        cudaMemcpy( phi_host, phi, sizeof(float)*Msize, cudaMemcpyDeviceToHost );
        int* charge_pointer;
        charge_pointer = charged_black_points;
        for (charge_pointer = charged_black_points;
                charge_pointer < &charged_black_points[number_of_charged_black_points];
                charge_pointer++)
        {
            phi_host[*charge_pointer] += omega * Q[*charge_pointer];
        }
        //Copy back after changes
        cudaMemcpy( phi, phi_host, sizeof(float)*Msize, cudaMemcpyHostToDevice );
        
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
 
        // second half of Gauss-Seidel iteration (red fields only)
        redSOR<<< gridSize, blockSize >>> ( phi, T, omega, lambda );
        cudaThreadSynchronize();
        
        cudaMemcpy( phi_host, phi, sizeof(float)*Msize, cudaMemcpyDeviceToHost );
        charge_pointer = charged_white_points;
        for (charge_pointer = charged_white_points;
                charge_pointer < &charged_white_points[number_of_charged_white_points];
                charge_pointer++)
        {
            phi_host[*charge_pointer] += omega * Q[*charge_pointer];
        }
        //Copy back after changes
        cudaMemcpy( phi, phi_host, sizeof(float)*Msize, cudaMemcpyHostToDevice );

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
            memcpy( tmp_phi_host, phi_host, (Msize * sizeof(float)) );
        }

        if ( (iteration % 10) == 0)
        {
            printf("Iteration number: %d\n", iteration);
//            printf("phi(teste): %.20f\n", phi_host[100550]);
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
    
    cudaFree(phi);
    cudaFree(T);
    cudaFreeHost(tmp_phi_host);

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
    loadMemFromFile( "Phi.dump", phi, (sizeof(float)*6*Msize) );
    
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
