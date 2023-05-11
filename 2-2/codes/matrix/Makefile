SRCS=$(wildcard *.c)

OEXECUTABLES=$(SRCS:.c=.out)

all: compile
compile: $(OEXECUTABLES)

%.out: %.c
	gcc $< -std=c11 -fopenmp -o $@

clean:
	rm -f $(OEXECUTABLES)
