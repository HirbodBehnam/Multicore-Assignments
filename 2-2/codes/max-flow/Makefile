SRCS=$(wildcard *.cpp)

OEXECUTABLES=$(SRCS:.cpp=.out)

all: compile
compile: $(OEXECUTABLES)

%.out: %.cpp
	g++ $< -fopenmp -o $@ -O3

clean:
	rm -f $(OEXECUTABLES)
