BIN_DIR := ../../bin/

OBJECTS := \
	libfm.o \
	tools/transpose.o \
	tools/convert.o \

all: libFM transpose convert

libFM: libfm.o
	mkdir -p $(BIN_DIR)
	g++ -O3 -Wall libfm.o -o $(BIN_DIR)libFM

%.o: %.cpp
	g++ -O3 -Wall -c $< -o $@

clean:	clean_lib
	mkdir -p $(BIN_DIR)
	rm -f $(BIN_DIR)libFM $(BIN_DIR)convert $(BIN_DIR)transpose

clean_lib:
	rm -f $(OBJECTS)


transpose: tools/transpose.o
	mkdir -p $(BIN_DIR)
	g++ -O3 tools/transpose.o -o $(BIN_DIR)transpose

convert: tools/convert.o
	mkdir -p $(BIN_DIR)
	g++ -O3 tools/convert.o -o $(BIN_DIR)convert

