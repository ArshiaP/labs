__kernel void rev_word(__global char *A) {
  int spac_idx = get_global_id(0);
  int id = 0, startid = 0, num_spaces = 0;
  while (num_spaces != spac_idx + 1) {
    if (A[id] == ' ' || A[id] == '\0') {
      num_spaces++;
      if (num_spaces == spac_idx)
        startid = id;
    }
    id++;
  }
  id -= 2;
  for (int i = startid; i < (startid + id) / 2; i++) {
    char t = A[i];
    A[i] = A[id - i];
    A[id - i] = t;
  }
}