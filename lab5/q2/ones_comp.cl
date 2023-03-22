__kernel void ones_comp(__global int *A, __global int *C){
    //Get index of the current work item
    int i = get_global_id(0);
	C[i]=~A[i];
}