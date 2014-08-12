
extern "C" __global__ void simple(occaKernelInfoArg, int N, float *d_a){
	   
  // Convert thread and thread-block indices into array index 
  const int n  = threadIdx.x + blockDim.x*blockIdx.x;
	   
  // If index is in [0,N-1] add entries
  if(n<N)
    d_a[n] = n;
}
