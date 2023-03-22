__kernel void prog( __global int *A,__global int *B)
{
// Get the index of the current work item
int i = get_global_id(0);
// Do the operation
int j=0,pow=1,t=A[i];
while(t>0)
{
    j=j+t%10*pow;
    t/=10;
    pow*=2;
}
B[i]=j;
}
