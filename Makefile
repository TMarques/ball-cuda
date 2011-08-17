all:
	nvcc main.cu -o main
emulation: 
	nvcc -deviceemu main.cu -o main
debugemu:  
	nvcc -O0 -deviceemu -g -G main.cu -o main 
debug:  
	nvcc -O0 -g -G main.cu -o main --ptxas-options=-v
clean:
	rm -rf main *.o
