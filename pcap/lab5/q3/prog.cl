__kernel void vector_add( __global int *A)
{
// Get the index of the current work item
int i = get_global_id(0);
// Do the operation
int t=A[i*2];
A[i*2]=A[i*2+1];
A[i*2+1]=t;

}
