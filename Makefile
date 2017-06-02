#CC=g++
#CFLAGS=-c -Wall
#LDFLAGS=
#SOURCES=main.cpp hello.cpp factorial.cpp
#OBJECTS=$(SOURCES:.cpp=.o)
#EXECUTABLE=hello
#
#all: $(SOURCES) $(EXECUTABLE)
#
#$(EXECUTABLE): $(OBJECTS) 
#	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
#
#.cpp.o:
#	$(CC) $(CFLAGS) $< -o $@



TARGET = classify
CC = g++
DEFINES       = -DUSE_OPENCV
CFLAGS = -g -std=c++11 -Wall $(DEFINES)
#LDFLAGS = -L/home/gxw/workspace/caffe/caffe/distribute/lib -lcaffe -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --libs opencv` # -lcublas 
LDFLAGS = -L/home/gxw/workspace/pva-faster-rcnn/caffe-fast-rcnn/distribute/lib -lcaffe -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --libs opencv` # -lcublas 
COMMON += `pkg-config --cflags opencv`
IDIR = -I/home/gxw/workspace/pva-faster-rcnn/caffe-fast-rcnn/distribute/include -I/usr/local/cuda/include
#IDIR = -I/home/gxw/workspace/caffe/caffe/distribute/include -I/usr/local/cuda/include
#IDIR = /disk01/workspace/pva-faster-rcnn/20170421/caffe-fast-rcnn/include/

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.cpp $(HEADERS)
	$(CC) -c $< -o $@ $(COMMON) $(CFLAGS) $(IDIR) 

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(COMMON) $(LDFLAGS) -Wall $(LDFLAGS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
