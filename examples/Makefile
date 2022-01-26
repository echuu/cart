
# Makefile for Writing Make Files Example
 
# *****************************************************
# Variables to control Makefile operation
 
CC = g++
CFLAGS = -Wall -g
 
# ****************************************************
# Targets needed to bring the executable up to date
 
main: main.o Tree.o Interval.o
	$(CC) $(CFLAGS) -o main main.o Tree.o Interval.o
 
# The main.o target can be written more simply
 
main.o: main.cpp Tree.h Interval.h
	$(CC) $(CFLAGS) -c main.cpp
 
Tree.o: Tree.h cart_types.h util.h Node.h


