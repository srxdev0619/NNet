CC = clang++
CFLAGS = -Wall -Werror -ggdb -std=c++11 -O3

TARGET = NNet


CPP:
	$(CC) $(CFLAGS) -o main main.cpp  NNet.cpp -larmadillo

CPP1:
	$(CC) $(CFLAGS) -o main1 main1.cpp  NNet.cpp -larmadillo

CPP2:
	$(CC) $(CFLAGS) -o main2 main2.cpp NNet.cpp -larmadillo

clean:
	rm -r main
