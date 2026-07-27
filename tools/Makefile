.SUFFIXES: .h .cpp
#-----------------------------------------------
# Makefile for 'tools' directory
# Please modify the variables properly.
# We recommend to use Intel c++ compiler.
#-----------------------------------------------

#Intel compiler
#CXX = icpc

# OSX (gcc)
#CXX = g++-9

# OSX (clang)
CXX = g++

# OSX
#CXX = clang++

CXXFLAGS = -O2 -std=c++11

# Add the directory containing boost header files here
INCLUDE = -I../include -I$(HOME)/include

#----------------------------------------------
# General rules
#----------------------------------------------

.cpp.o:
	${CXX} ${CXXFLAGS} ${INCLUDE} -c $<

PROG = qe2alm dfc2 fc_virtual

CXXSRC = qe2alm.cpp dfc2.cpp fc_virtual.cpp

OBJS= ${CXXSRC:.cpp=.o}

default: all

all: ${PROG}

qe2alm: qe2alm.o
	${CXX} ${CXXFLAGS} -o $@ $<

dfc2: dfc2.o
	${CXX} ${CXXFLAGS} -o $@ $<

fc_virtual: fc_virtual.o
	${CXX} ${CXXFLAGS} -o $@ $<

clean:
	rm -f ${OBJS}

.PHONY: clean

depend:
	gcc -MM *.cpp > .depend

-include .depend
