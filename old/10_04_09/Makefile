all:
	nvcc main.cu -o main
emulation:
	nvcc -deviceemu main.cu -o main
clean:
	rm -rf main *.o
