CC = gcc
CFLAGS = -std=c99 -O3 -Wall -pedantic
CXX = g++
CXXFLAGS = -std=c++98 -O3 -Wall -pedantic

# uncomment to disable array headers
#CFLAGS += -DWITHOUT_HEADER

all: fpzip testfpzip

fpzip: fpzip.cpp ../lib/libfpzip.a
	$(CXX) $(CXXFLAGS) -I../inc -L../lib fpzip.cpp -lfpzip -o $@

testfpzip: testfpzip.o ../lib/libfpzip.a
	$(CXX) $(CXXFLAGS) -L../lib testfpzip.o -lfpzip -o $@

testfpzip.o: testfpzip.c
	$(CC) $(CFLAGS) -I../inc -c $<

clean:
	@rm -f fpzip testfpzip testfpzip.o
