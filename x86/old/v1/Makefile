all:
	gcc main.c -o main -lm
BALLoptimized:
	gcc -pipe -fPIC -O3 -Wall -W -pedantic -Wno-long-long main.c -o main -lm
optimized:
	gcc -Os -ffast-math -pipe -mtune=core2 -funroll-loops -msse3 main.c -o main -lm
optimized64:
	gcc -Os -ffast-math -pipe -march=nocona -msse3 main.c -o main -lm
clean:
	rm -rf main *.o
