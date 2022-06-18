GCC = g++
NVCC = nvcc
CFLAGS = -Wall -std=c++0x
#No linking flags for now
OBJECTS = randv cpu_velocityverlet cpuopt_velocityverlet gpu_velocityverlet gpuopt_velocityverlet

.PHONY: all
all: $(OBJECTS)

randv : randv.c randoms
	chmod +x randoms
	gcc -std=c99 -o randv randv.c
	mkdir init_sets
	./randv "init_sets/testing_set.dat" 8

cpu_velocityverlet : cpu_velocityverlet.cpp
	$(GCC) $(CFLAGS) -o $@ $<

cpuopt_velocityverlet : cpuopt_velocityverlet.cpp
	$(GCC) $(CFLAGS) -o $@ $<

gpu_velocityverlet : gpu_velocityverlet.cu book.h
	$(NVCC) -o $@ $<

gpuopt_velocityverlet : gpuopt_velocityverlet.cu book.h
	$(NVCC) -o $@ $<


.PHONY: clean
clean:
	rm -f $(OBJECTS)
	rm -R init_sets

