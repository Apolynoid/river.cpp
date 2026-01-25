CXX      := g++
CXXFLAGS := -O3 -std=c++20 -march=native -Wall -I../include
TARGET   := ex_phising
SRC      := ex_phising.cpp
.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)