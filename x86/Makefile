CC=gcc
ARCHFLAGS=-m32

all:
	${CC} main.c -o main -lm ${ARCHFLAGS}
openmp:
	${CC} main.c -o main -lm ${ARCHFLAGS} -fopenmp
BALLoptimized:
	${CC} -pipe -fPIC -O3 -Wall -W -pedantic -Wno-long-long main.c -o main -lm ${ARCHFLAGS}
optimized:
	${CC} -Os -ffast-math -pipe -march=athlon-xp -msse3 main.c -o main -lm ${ARCHFLAGS}
clean:
	rm -rf main *.o
