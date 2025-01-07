#must include the libraries using -I -L
g++ chason-host.cpp chason.cpp -o chason -O2  -I/usr/include -L/usr/lib -ltapa -lfrt -lglog -lgflags -lOpenCL -std=c++17 -w

