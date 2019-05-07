#ifndef _PEPECNN_H
#define _PIPECNN_H

int memoryCreat();
int pipeCNN(signed char *weightsCaffe,signed char* imageCaffe,signed char *outputCaffe);
void cleanupSDaccel();
#endif
