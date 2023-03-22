__kernel void str_chgcase(__global char *A, __global int *l, __global char *B) {
  int i = get_global_id(0);
  int n = get_global_size(0);
  int len = *l;
  for (int j = i * len; j < (i + 1) * len; j++)
    B[j] = A[j - i * len];
}