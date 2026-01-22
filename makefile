main : main.cpp makefile
	g++ -O2 -std=c++20 $< -o main -Iinclude