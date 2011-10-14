#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "/home/tmarques/cuda/include/cuda.h"
#include <pthread.h>
//#include "/home/tmarques/NVIDIA_CUDA_SDK/common/inc/cutil.h"

//VERSION NOTES:
// Move cudaMemcpy to inside one of the conditional
// checks for the number of iterations,
// which reduces the number of copies to
// around the number of number_iterations/check_after_iterations

/** Return values:
 * 0 - OK
 * 1 - Can't open one of the data files
 * 2 - Can't allocate CPUMEM for matrix
 * 3 - Can't allocate GPUMEM for matrix
 * 4 - Matrix coordinates can't exceed 512
 */

typedef struct 
{
    int *size;
    float *phi;
    float *T;
    float *Q;
    int *CBPoints;
    int *CWPoints;
    int threadNum;
    int deviceCount;
} DoSORArg;

//First include the texture reference,declared globally:
texture<float, 1, cudaReadModeElementType> phi_texture;

//Phi memory pointers for GPUs
float *phiGPU[3];

int dumpMemToFile(char *file, void *input, int size);

__global__ void calcCP( float *phi, float *Q, int *charged_points, int number_of_charged_points, float omega, int start, int end )
{    
    int *charge_pointer = charged_points + blockIdx.x;

    if ( ! 
            ( ( ( phi + *charge_pointer ) < ( phi + start ) ) 
              ||   ( ( phi + *charge_pointer ) > ( phi + end ) ) ) 
       )
    {
        phi[ *charge_pointer ] += omega * Q[ *charge_pointer ];
    }
}

__global__ void blackSOR(float *phi, float *T, float omega, float lambda, int start, int end)
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'

    // We don't calculate the border values.
    if 
        (
         ( ! ( ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) || ( threadIdx.x == 0 ) )
               || ( blockIdx.x == blockDim.x - 1 ) || ( blockIdx.y == gridDim.y - 1 ) || ( threadIdx.x == blockDim.x - 1 ) ) ) &&
         ( ! ( ( blockIdx.x < start ) || ( blockIdx.x > end ) ) ) // And we don't want to write out of our block.
        )
        {
            unsigned int incr = ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
            // Black?
            if ( ( threadIdx.x % 2 ) != incr )
            {
                unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
                float phi1 = tex1Dfetch( phi_texture, i + 1 ),
                      phi2 = tex1Dfetch( phi_texture, ( i - 1 ) ),
                      phi3 = tex1Dfetch( phi_texture, ( i + blockDim.x ) ),
                      phi4 = tex1Dfetch( phi_texture, ( i - blockDim.x ) ),
                      phi5 = tex1Dfetch( phi_texture, ( i + blockDim.x * gridDim.x ) ),
                      phi6 = tex1Dfetch( phi_texture, ( i - blockDim.x * gridDim.x ) ),
                      phi0 = tex1Dfetch( phi_texture, i );

                phi[i] = omega * (
                        T[6 * i    ] * phi1
                        + T[6 * i + 1] * phi2
                        + T[6 * i + 2] * phi3
                        + T[6 * i + 3] * phi4
                        + T[6 * i + 4] * phi5
                        + T[6 * i + 5] * phi6 )
                    + lambda * phi0;
            }
        }
}

__global__ void whiteSOR(float *phi, float *T, float omega, float lambda, int start, int end)
{
    //FIXME: Check if using "i" is faster than recalculating it's value
    // Use block.Idx.x as 'y', blockIdx.y as 'z' and threadIdx.x as 'x'

    // We don't calculate the border values
    if 
        (
         ( ! ( ( ( blockIdx.x == 0 ) || ( blockIdx.y == 0 ) || ( threadIdx.x == 0 ) )
               || ( blockIdx.x == blockDim.x - 1 ) || ( blockIdx.y == gridDim.y - 1 ) || ( threadIdx.x == blockDim.x - 1 ) ) ) &&
         ( ! ( ( blockIdx.x < start ) || ( blockIdx.x > end ) ) ) // And we don't want to write out of our block.
        )
        {
            unsigned int incr = 1 - ( ( ( blockIdx.x % 2 ) + ( blockIdx.y % 2 ) ) % 2 );
            // Red?
            if ( (  threadIdx.x % 2 ) != incr )
            {
                unsigned int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
                float phi1 = tex1Dfetch( phi_texture, i + 1 ),
                      phi2 = tex1Dfetch( phi_texture, ( i - 1 ) ),
                      phi3 = tex1Dfetch( phi_texture, ( i + blockDim.x ) ),
                      phi4 = tex1Dfetch( phi_texture, ( i - blockDim.x ) ),
                      phi5 = tex1Dfetch( phi_texture, ( i + blockDim.x * gridDim.x ) ),
                      phi6 = tex1Dfetch( phi_texture, ( i - blockDim.x * gridDim.x ) ),
                      phi0 = tex1Dfetch( phi_texture, i );

                phi[i] = omega * (
                        T[6 * i    ] * phi1
                        + T[6 * i + 1] * phi2
                        + T[6 * i + 2] * phi3
                        + T[6 * i + 3] * phi4
                        + T[6 * i + 4] * phi5
                        + T[6 * i + 5] * phi6 )
                    + lambda * phi0;
            }
        }
}

void *doSOR ( void *do_SOR_arg )
{
    DoSORArg *host = ( DoSORArg * ) do_SOR_arg;

    //    printf( "%i\n", host->threadNum );
    //    pthread_cancel( pthread_self() );
    //    pthread_exit( NULL );

    float residual = 0, residual_norm2 = 0; 
    float rms_criterion = 1e-6F, max_criterion = 1e-6F;
    float max_residual = max_criterion + 1;
    float rms_change = rms_criterion + 1;
    int max_iterations = 411, iteration = 0;
    int check_after_iterations = 50;
    int i, l = 2080638;
    float spectral_radius=0.99969899654388427734375;

    int number_of_charged_black_points = 5330, number_of_charged_white_points = 5340;

    int Nx = host->size[0], Ny = host->size[1], Nz = host->size[2];
    int Nxy = Nx * Ny;
    int N = Nxy * Nz;
    float *offset, *target;

    //GPU boundaries, last GPU takes the bigger slice
    int start, end;
    //Initialize memory pointers for GPUs, needed for communication
    float *phi = phiGPU[ host->threadNum - 1 ];
    float *phiPrev = NULL;
    float *phiNext = NULL;

    if ( host->deviceCount > 1 )
    {
        start = ( host->threadNum - 1 ) * ( Ny / host->deviceCount );
        if ( host->deviceCount == host->threadNum )
            end = host->threadNum * ( Ny / host->deviceCount ) + Ny % host->deviceCount;
        else
        {
            end = host->threadNum * ( Ny / host->deviceCount ) - 1;
            phiNext = phiGPU[ host->threadNum ];
        }

        if ( host->threadNum > 1 )
            phiPrev = phiGPU[ host->threadNum - 2 ];
        cudaSetDevice( host->threadNum - 1 );
    }
    else
    {
        start = 0;
        end = Ny - 1;
    }

    //Calculate the new values for the matrices to be copied
    //    Nx = end - start + 1;
    //    Nxy = Nx * Ny;
    //    N = Nxy * Nz;
    //Memory ranges, with boundaries

    //FIXME: Clean this up
    int Msize = N;

    //    float *phi;
    if ( cudaMalloc((void **) &phi, sizeof(float)*Msize) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for phi matrix.\n");
        //return 3;
    }
    cudaMemcpy(phi, host->phi, sizeof(float)*Msize, cudaMemcpyHostToDevice);

    //FIXME: 
    //       Watch out for size limits for textures(2^27) / MAXIMUM_TEXTURE??
    //       Don't copy the full phi, T and Q matrices to all GPUs
    int textureError = cudaBindTexture( 0, phi_texture, phi, (sizeof(float)*Msize ) );
    if ( textureError != cudaSuccess )
    {
        if ( textureError == cudaErrorInvalidTexture )
        {
            printf("Error, invalid phi_white texture.\n");
            //            return 5;
        } else if ( textureError == cudaErrorInvalidValue )
        {
            printf("Error, invalid value!\n");
            //          return 5;
        } else
        {
            printf("Undefined error in phi_white_texture binding.\n");
            //        return 5;
        }
    }

    float *T;
    if ( cudaMalloc( (void **) &T, (sizeof(float)*6*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for T matrix.\n");
        //return 3;
    }
    cudaMemcpy( T, host->T, (sizeof(float)*6*Msize), cudaMemcpyHostToDevice ); 

    float *tmp_phi_host;
    if ( cudaMallocHost( (void **) &tmp_phi_host, (sizeof(float)*Msize) ) 
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate CPUMEM for tmp_phi_host matrix.\n");
        //return 2;
    }

    float *Q;
    if ( cudaMalloc( (void **) &Q, (sizeof(float)*Msize) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for Q matrix.\n");
        //return 3;
    }
    cudaMemcpy( Q, host->Q, (sizeof(float)*Msize), cudaMemcpyHostToDevice );

    int *charged_black_points;
    if ( cudaMalloc( (void **) &charged_black_points, (sizeof(int)*number_of_charged_black_points) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for charged_black_points.\n");
        //return 3;
    }
    cudaMemcpy( charged_black_points, host->CBPoints, (sizeof(int)*number_of_charged_black_points), cudaMemcpyHostToDevice );

    int *charged_white_points;
    if ( cudaMalloc( (void **) &charged_white_points, (sizeof(int)*number_of_charged_white_points) )
            == cudaErrorMemoryAllocation )
    {
        fprintf(stderr, "Can't allocate GPUMEM for charged_black_points.\n");
        //return 3;
    }
    cudaMemcpy( charged_white_points, host->CWPoints, (sizeof(int)*number_of_charged_white_points), cudaMemcpyHostToDevice );


    //Setup CUDA kernel parameters
    if ( Nz > 512 )
    {
        printf("Matrix size is too large, must be less than 512\n");
        //return 4; 
    }
    //Only 2D grids supported at this time
    dim3 gridSize( Nx, Ny, 1 );
    dim3 blockSize( Nz, 1, 1 );
    //End setup

    //Check for number_of_charged_***_points
    //to be less than ~2^16:
    if ( ( number_of_charged_black_points > 65530 ) || ( number_of_charged_white_points > 65530 ) )
    {
        printf("One of charged points is too big, must be less than 65500.\n");
        printf("This is a shortcoming of naively implemented code, please address if necessary.\n");
        //return 5;
    }
    cudaThreadSynchronize();

    //BALL_START
    float omega = 1, lambda = 1 - omega;

    //dumpMemToFile("output.dump", phi_host, sizeof(float)*Msize);

    while ((iteration < max_iterations)  && ((max_residual > max_criterion) || (rms_change > rms_criterion)))
    {

        // first half of Gauss-Seidel iteration (black fields only)
        blackSOR<<< gridSize, blockSize >>> ( phi, T, omega, lambda, start, end );
        cudaThreadSynchronize();
       // calcCP<<< number_of_charged_black_points, 1 >>> ( phi, Q, charged_black_points, number_of_charged_black_points, omega, start, end );
        cudaThreadSynchronize();
        //Synchronize Phi's barriers over GPUs
        if ( host->deviceCount > 1 )
        {
            if ( host->threadNum == host->deviceCount )
            {
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
                    target = phiPrev + ( start * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
            }
            else if ( host->threadNum == 1 )
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( end * Nx  + i * Nxy  ); 
                    target = phiNext + ( end * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
            else
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); 
                    target = phiPrev + ( start * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                    offset = phi + ( end * Nx  + i * Nxy ); 
                    target = phiNext + ( end * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
        }
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

        // second half of Gauss-Seidel iteration (red fields only)
        whiteSOR<<< gridSize, blockSize >>> ( phi, T, omega, lambda, start, end );
        cudaThreadSynchronize();
     //   calcCP<<< number_of_charged_white_points, 1 >>> ( phi, Q, charged_white_points, number_of_charged_white_points, omega, start, end );
        cudaThreadSynchronize();
        //Synchronize Phi's barriers over GPUs
        if ( host->deviceCount > 1 )
        {
            if ( host->threadNum == host->deviceCount )
            {
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
                    target = phiPrev + ( start * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
            }
            else if ( host->threadNum == 1 )
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( end * Nx  + i * Nxy  ); 
                    target = phiNext + ( end * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
            else
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); 
                    target = phiPrev + ( start * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                    offset = phi + ( end * Nx  + i * Nxy ); 
                    target = phiNext + ( end * Nx  + i * Nxy );
                    //                    cudaMemcpyAsync( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice, NULL );
                    cudaMemcpy( target, offset, Nx * sizeof( float ), cudaMemcpyDeviceToDevice );
                }
        }
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
            //Copy back the blocks
            if ( host->deviceCount > 1 )
            {
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
                    target = host->phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
                //    cudaMemcpyAsync(target , offset, ( end - start + 1 ) * Nx * sizeof( float ), cudaMemcpyDeviceToHost, NULL );
                    cudaMemcpy(target, offset, ( end - start + 1 ) * Nx * sizeof( float ), cudaMemcpyDeviceToHost );
                }
                cudaThreadSynchronize();
            }
            else
                cudaMemcpy( host->phi, phi, sizeof(float)*Msize, cudaMemcpyDeviceToHost );

            if (iteration > 0)
            {
                max_residual = 0;
                residual_norm2 = 0;

                // sum up all squared changes in the phi array since
                // the last iteration
                for (i = 1; i < (N - 1); i++)
                {
                    residual = fabs(tmp_phi_host[i] - host->phi[i]);
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
            //Copy back the blocks
            if ( host->deviceCount > 1 )
            {
                for ( i = 0; i < Nz; i++ )
                {
                    offset = phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
                    target = host->phi + ( start * Nx  + i * Nxy ); // Line to copy, counted in floats
//                    cudaMemcpyAsync(target , offset, ( end - start + 1 ) * Nx * sizeof( float ), cudaMemcpyDeviceToHost, NULL );
                    cudaMemcpy(target, offset, ( end - start + 1 ) * Nx * sizeof( float ), cudaMemcpyDeviceToHost );
                }
                cudaThreadSynchronize();
            }
            else
                cudaMemcpy( host->phi, phi, sizeof(float)*Msize, cudaMemcpyDeviceToHost );

            memcpy( tmp_phi_host, host->phi, (Msize * sizeof(float)) );
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
    cudaFree(Q);
    cudaFree(charged_black_points);
    cudaFree(charged_white_points);
    cudaFreeHost(tmp_phi_host);

    return 0;
}

int loadMemFromFile(char *file, void *output, int size)
{
    int fRead;

    fRead = open(file, O_RDONLY);
    if ( fRead == -1 )
    {
        printf( "Error opening %s file.\n", file );
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

int main( void )
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
    float *T;
    cudaMallocHost((void **) &T, (sizeof(float)*6*Msize) );    
    loadMemFromFile( "T.dump", T, (sizeof(float)*6*Msize) );
    float *Q;
    cudaMallocHost((void **) &Q, (sizeof(float)*Msize) );
    loadMemFromFile( "Q.dump", Q, (sizeof(float)*Msize) );
    int *CBPoints;
    cudaMallocHost((void **) &CBPoints, (sizeof(int)*5330) );
    loadMemFromFile( "Black_points.dump", CBPoints, (sizeof(int)*5330) );
    int *CWPoints;
    cudaMallocHost((void **) &CWPoints, ( sizeof(int)*5340) );
    loadMemFromFile( "White_points.dump", CWPoints, (sizeof(int)*5340) );

    cudaThreadSynchronize();

    //Get number of devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        if (device == 0) 
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
    }

    if ( deviceCount > 1 )    
    {
        pthread_t thread1, thread2;

        DoSORArg do_SOR_arg  = { size, phi, T, Q, CBPoints, CWPoints, 1, deviceCount };
        DoSORArg do_SOR_arg2 = { size, phi, T, Q, CBPoints, CWPoints, 2, deviceCount };

        int threadRtn1 = pthread_create( &thread1, NULL, doSOR, (void *) &do_SOR_arg );
        int threadRtn2 = pthread_create( &thread2, NULL, doSOR, (void *) &do_SOR_arg2 );

        pthread_join( thread1, NULL);
        pthread_join( thread2, NULL);

        printf( "GPU 1 done, returns: %d\n", threadRtn1 );
        printf( "GPU 2 done, returns: %d\n", threadRtn2 );
    }
    //    else 
    //        doSOR( size, phi, T, Q, CBPoints, CWPoints, 1 );    


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

    //    doSOR(size, phi, T, Q, CBPoints, CWPoints);
    printf("Finished doSOR.\n");

    cudaFreeHost(phi);
    cudaFreeHost(T);
    cudaFreeHost(Q);
    cudaFreeHost(CBPoints);
    cudaFreeHost(CWPoints);

    return 0;
}
