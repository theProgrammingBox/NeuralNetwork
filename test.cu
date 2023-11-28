#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>

#define HISTOGRAM_SIZE 10
#define ITERATIONS 1000000

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2 ^ 0x4ba1bb47;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1 ^ 0xb7ebcb79;
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;)  mixSeed(seed1, seed2);
}

int main() {
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);

    unsigned int histogram[HISTOGRAM_SIZE] = {0};

    for (int i = 0; i < ITERATIONS; i++) {
        mixSeed(&seed1, &seed2);
        int index = (int)(((double)seed1 / (double)UINT32_MAX) * HISTOGRAM_SIZE);
        if (index >= 0 && index < HISTOGRAM_SIZE) {
            histogram[index]++;
        }
    }

    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        printf("Range %d: %u\n", i, histogram[i]);
    }

    return 0;
}
