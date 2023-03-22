__kernel void convert_octal(__global int *A, __global int *C){
    //Get index of the current work item
    int i = get_global_id(0);
	int octalNumber[100],x=1,j;
    int quotient = A[i];
	while(quotient!=0) {
		octalNumber[x++]= quotient % 8;
		quotient = quotient / 8;
	}
    C[i]=0;
    
	for (j = x -1 ;j> 0;j--){
        C[i]*=10;
        C[i]+=octalNumber[j];
    }
}