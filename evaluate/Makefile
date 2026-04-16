CXX      := g++
CXXFLAGS := -O3 -std=c++20 -march=native -Wall -I../include

SRCS     := $(wildcard *.cpp)

TARGETS  := $(SRCS:.cpp=.out)

.PHONY: all clean

all: $(TARGETS)

%.out: %.cpp Makefile
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)