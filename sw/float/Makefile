CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

TARGET   := mnist_cnn

SRCS := main.cpp \
        mnist_loader.cpp \
        conv_layer.cpp \
        pool_layer.cpp \
        fc_layer.cpp \
        weights_io.cpp \
        cnn.cpp

OBJS := $(SRCS:.cpp=.o)

COMMON   := ../common
DATA_DIR := $(COMMON)/data
WGTS_DIR := $(COMMON)/weights

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

# Run inference on MNIST test set using trained weights
run: $(TARGET)
	./$(TARGET) $(DATA_DIR)/t10k-images-idx3-ubyte \
	            $(DATA_DIR)/t10k-labels-idx1-ubyte  \
	            $(WGTS_DIR)

# Smoke test — random weights, verifies shapes compile and run correctly
smoke: $(TARGET)
	./$(TARGET) $(DATA_DIR)/t10k-images-idx3-ubyte \
	            $(DATA_DIR)/t10k-labels-idx1-ubyte

.PHONY: all clean run smoke
