__kernel void str_chgcase(__global int *A, __global int *B) {
  int idx = get_global_id(0);
  int n = get_global_size(0);
  int data = A[idx], pos = 0;
  for (int i = 0; i < n; i++) {
    if (A[i] < data || (A[i] == data && i < idx))
      pos++;
  }
  B[pos] = data;
}