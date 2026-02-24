CXX = clang++
CXXFLAGS = -std=c++20 -I./metal-cpp
LDFLAGS = -framework Metal -framework Foundation -framework QuartzCore

all: metal_sbi

metal_sbi: src/main.o
	$(CXX) $(LDFLAGS) -o $@ $^

src/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o metal_sbi
