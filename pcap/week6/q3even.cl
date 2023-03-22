__kernel void even_kernel(__global int *A) {
  // Get the index of the current work item
  int i = get_global_id(0);
  // Do the operation
  int n = get_global_size(0) * 2;
  if (i * 2 + 2 < n) {
    if (A[i * 2 + 1] > A[i * 2 + 2]) {
      int t = A[i * 2 + 1];
      A[i * 2 + 1] = A[i * 2 + 2];
      A[i * 2 + 2] = t;
    }
  }
}
