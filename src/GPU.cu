#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512
#define SUBARRAY_LENGTH 4

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
float *d_histogram;

// convert arcminutes To Radians
void arcminutesToRadians(int NoofGalaxies, float * ra, float * decl) {
  for (int i = 0; i < NoofGalaxies; i++) {
    ra[i] = ra[i] / 60.0 * M_PI / 180.0;
    decl[i] = decl[i] / 60.0 * M_PI / 180.0;
  }
}

// CUDA Kernel 
__global__ void calculateHistograms(float * ra_1, float * decl_1, float * ra_2, float * decl_2,
  unsigned int * histogram, int NoofReal) {

    __shared__ float ra_1_SharedData[threadsperblock*SUBARRAY_LENGTH]; // Specific ra_real data to shared memory
    __shared__ float decl_1_SharedData[threadsperblock*SUBARRAY_LENGTH]; // Specific decl_real to shared memory
    __shared__ float ra_2_SharedData[threadsperblock*SUBARRAY_LENGTH]; // Specific ra_sim to shared memory
    __shared__ float decl_2_SharedData[threadsperblock*SUBARRAY_LENGTH]; // Specific decl_sim to shared memory
    __shared__ unsigned int local_histogram[totaldegrees * binsperdegree * sizeof(unsigned int)];// Specific local histogram to shared memory
   
    // Get Global ThreadID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate the start Index
    int startIndex = (threadId%threadsperblock)*SUBARRAY_LENGTH;

    // calculate the useful threads
    long int usefulThread = (NoofReal/SUBARRAY_LENGTH)*(NoofReal/SUBARRAY_LENGTH);

    // Copy data from Global Memory to Shared Memory
    // In this cause, Used threads 100000*100000/16/16 = 39062500
    // Each thread will copy its target data from global memory to the shared memory
    if (threadId < usefulThread){
      for (int i = 0; i < SUBARRAY_LENGTH; i++) {
          ra_1_SharedData[i+startIndex] = ra_1[(threadId / (NoofReal/SUBARRAY_LENGTH)) * SUBARRAY_LENGTH + i];
          decl_1_SharedData[i+startIndex] = decl_1[(threadId / (NoofReal/SUBARRAY_LENGTH)) * SUBARRAY_LENGTH + i];
          ra_2_SharedData[i+startIndex] = ra_2[(threadId % (NoofReal/SUBARRAY_LENGTH)) * SUBARRAY_LENGTH + i];
          decl_2_SharedData[i+startIndex] = decl_2[(threadId % (NoofReal/SUBARRAY_LENGTH)) * SUBARRAY_LENGTH + i];
        }
    }
    // Synchronize to ensure all threads per block have finished copying for target data
    __syncthreads();

    // if it is the first thread in the thread block, initalize the local_histogram shared memory to all 0 
    if (threadIdx.x == 0) {
      for(int i =0;i<totaldegrees * binsperdegree; i++){
        local_histogram[i] = 0;
      }
    }
    // Synchronize to ensure all threads per block are ready to use initialized local_histogram
    __syncthreads();

    // within the useful threads
    if (threadId < usefulThread){
      // calculate data, update histograms into local_histogram
      for (int i = 0; i < SUBARRAY_LENGTH; i++) {
        for (int j = 0; j < SUBARRAY_LENGTH; j++) {
          // calculate angle
            float angle = ( cosf(decl_1_SharedData[i+startIndex]) * \
                              cosf(decl_2_SharedData[j+startIndex]) * \
                              cosf(ra_1_SharedData[i+startIndex]-ra_2_SharedData[j+startIndex]) + \
                              sinf(decl_1_SharedData[i+startIndex]) * \
                              sinf(decl_2_SharedData[j+startIndex]));
            if (angle > +1.0f) angle = +1.0f;
            if (angle < -1.0f) angle = -1.0f;
            float degree = acosf(angle) * 180.0f /M_PI;
            atomicAdd(&local_histogram[(int)(degree * 4.0f)], 1);
        }
      }
      // Synchronize to ensure all threads per block finish calculattion and update the local_histogram
       __syncthreads();
      // Up date the local_histogram to global histogram
      if (threadIdx.x == 0) {
        for(int i =0;i<totaldegrees * binsperdegree; i++){
          atomicAdd(&histogram[i], local_histogram[i]); 
        }
      }
    }
}

int main(int argc, char *argv[])
{
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
  //  cudaError_t myError;

   FILE *outfile;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);



  // Convert ArcMinutes to Radians
  arcminutesToRadians(NoofReal, ra_real, decl_real);
  arcminutesToRadians(NoofSim, ra_sim, decl_sim);

  // kerneltime = 0.0;
  // gettimeofday(&_ttime, &_tzone);
  // start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

  // Allocate memory on Both GPU and CPU using cudaMallocManaged()
  // Note that ra_real, decl_real, ra_sim and decl_sim have been assigned value already, while histogramDR,DD and RR are all 0
  cudaMallocManaged((void ** ) & histogramDR, totaldegrees * binsperdegree * sizeof(unsigned int));
  cudaMallocManaged((void ** ) & histogramDD, totaldegrees * binsperdegree * sizeof(unsigned int));
  cudaMallocManaged((void ** ) & histogramRR, totaldegrees * binsperdegree * sizeof(unsigned int));

  // Launch the kernel with a 2D grid of blocks using minimum blocks and 512 threads per block

  int numberofblocks = ((NoofReal/SUBARRAY_LENGTH)*(NoofReal/SUBARRAY_LENGTH) + threadsperblock - 1) / threadsperblock;
  
  // start time
  kerneltime = 0.0;
  gettimeofday(&_ttime, &_tzone);
  start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

  // cudaEventRecord(start);
  calculateHistograms << < numberofblocks, threadsperblock >>> (ra_real, decl_real, ra_sim, decl_sim,
    histogramDR, NoofReal); 
  calculateHistograms << < numberofblocks, threadsperblock >>> (ra_real, decl_real, ra_real, decl_real,
    histogramDD, NoofReal); 
  calculateHistograms << < numberofblocks, threadsperblock >>> (ra_sim, decl_sim, ra_sim, decl_sim,
     histogramRR, NoofReal); 
  // Explicityly Call cudaDeviceSynchronize to synchonise For Data Consistency
  cudaDeviceSynchronize();

  // end time
   gettimeofday(&_ttime, &_tzone);
   end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
   kerneltime += end-start;
   
  // calculate Omega
  float omega[360];
  for (int i = 0; i< totaldegrees * binsperdegree/2; i++){
    // I cannot understand...
    //  omega[i] = ((float)(histogramDD[i] - 2 * histogramDR[i] + histogramRR[i]) / (float)histogramRR[i]);
     omega[i] = (((float)histogramDD[i] - 2 * histogramDR[i] + histogramRR[i]) / (float)histogramRR[i]);
  }

    // Open the file for writing
    outfile = fopen(argv[3], "w");
    // Check if the file was successfully opened
    if (outfile == NULL) {
        perror("Error opening the file");
        return 1;
    }
    // print results
    printf("   no of blocks: %d , threads per block: %d, total treads: %d\n", numberofblocks, threadsperblock, numberofblocks * threadsperblock);
    for (int i = 0; i < 6; i++) {
      if (i==0){
          printf( "     Omega     histogramDD   histogramDR   histogramRR\n" );
        }
        printf("   %f    %d    %d    %d\n", omega[i], histogramDD[i],histogramDR[i],histogramRR[i]);
    }
    printf("   time in GPU is %f\n", kerneltime);
    printf("   data writes in %s\n", argv[3]);

    // Write the array elements to the file
    for (int i = 0; i < 361; i++) {
        if (i==0){
          fprintf(outfile, "Omega   histogramDD   histogramDR   histogramRR\n" );
        }
        fprintf(outfile, "%f    %d    %d    %d\n", omega[i], histogramDD[i],histogramDR[i],histogramRR[i]);
    }
    // Close the file
    fclose(outfile);



  // Cleanup: Free GPU memory, if use cudaMallocManaged, still need to free memory on GPU
  cudaFree(ra_real);
  cudaFree(decl_real);
  cudaFree(ra_sim);
  cudaFree(decl_sim);
  cudaFree(histogramDR);
  cudaFree(histogramDD);
  cudaFree(histogramRR);


  return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;
                                         

  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  // ra_real   = (float *)calloc(NoofReal,sizeof(float));
  // decl_real = (float *)calloc(NoofReal,sizeof(float));
  cudaMallocManaged((void ** ) & ra_real, NoofReal * sizeof(float));
  cudaMallocManaged((void ** ) & decl_real, NoofReal * sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  cudaMallocManaged((void ** ) & ra_sim, NoofSim * sizeof(float));
  cudaMallocManaged((void ** ) & decl_sim, NoofSim * sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

