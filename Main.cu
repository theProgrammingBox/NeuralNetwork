#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#define BOARD_SIZE 8
#define VISION_SIZE 8

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 ^= (*seed1 >> 13) * 0x9c7493ad;
    *seed1 ^= (*seed2 >> 17) * 0xbf324c81;
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) mixSeed(seed1, seed2);
}

uint32_t generateRandomUI32(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return *seed1;
}

int main() {
    uint8_t board[BOARD_SIZE * BOARD_SIZE] = {0};
    uint8_t px, py;
    uint8_t gx, gy;
    char move;
    
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    px = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    py = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    do {
        gx = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
        gy = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    } while (gx == px && gy == py);
    board[py * BOARD_SIZE + px] = 1;
    board[gy * BOARD_SIZE + gx] = 2;
    while (true) {
        system("clear");
        
        for (uint8_t ry = 0; ry < BOARD_SIZE; ry++) {
            for (uint8_t rx = 0; rx < BOARD_SIZE; rx++) {
                switch (board[ry * BOARD_SIZE + rx]) {
                    case 0:
                        printf("- ");
                        break;
                    case 1:
                        printf("P ");
                        break;
                    case 2:
                        printf("C ");
                        break;
                }
            }
            printf("\n");
        }
        printf("\n");
        
        for (int8_t ry = py - VISION_SIZE; ry <= py + VISION_SIZE; ry++) {
            for (int8_t rx = px - VISION_SIZE; rx <= px + VISION_SIZE; rx++) {
                switch (board[(ry + BOARD_SIZE) % BOARD_SIZE * BOARD_SIZE + (rx + BOARD_SIZE) % BOARD_SIZE]) {
                    case 0:
                        printf("- ");
                        break;
                    case 1:
                        printf("P ");
                        break;
                    case 2:
                        printf("C ");
                        break;
                }
            }
            printf("\n");
        }
        printf("\n");
        
        board[py * BOARD_SIZE + px] = 0;
        
        printf("Move (wasd): ");
        scanf(" %c", &move);
        px += (move == 'd') - (move == 'a');
        py += (move == 's') - (move == 'w');
        px = (px + BOARD_SIZE) % BOARD_SIZE;
        py = (py + BOARD_SIZE) % BOARD_SIZE;
        
        if (px == gx && py == gy) {
            board[gy * BOARD_SIZE + gx] = 0;
            do {
                gx = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
                gy = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
            } while (gx == px && gy == py);
            board[gy * BOARD_SIZE + gx] = 2;
        }
        board[py * BOARD_SIZE + px] = 1;
    }
    
    return 0;
}