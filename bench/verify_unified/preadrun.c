// preadrun DYLIB PROGRAM ARGS...: exec PROGRAM with DYLIB inserted (see preadlog.c)
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char** argv) {
  if (argc < 3) return fprintf(stderr, "preadrun DYLIB PROGRAM ARGS...\n"), 2;
  setenv("DYLD_INSERT_LIBRARIES", argv[1], 1);
  execv(argv[2], argv + 2);
  perror("execv");
  return 127;
}
