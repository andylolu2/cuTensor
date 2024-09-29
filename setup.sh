#!/bin/bash

# Install dependencies
apt-get update
apt-get install -y ninja-build

# Build the project
cmake -B build -G Ninja
cmake --build build