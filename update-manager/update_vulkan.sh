#!/bin/bash

OUTPUT="../../lib"

if [ -d "vulkan-zig" ]; then
  rm -rf "vulkan-zig"
fi

git clone https://github.com/Snektron/vulkan-zig.git

cd vulkan-zig

# Build the program if there is no exe
if [ ! -f "zig-out/bin/vulkan-zig-generator" ]; then
    zig build -Doptimize=ReleaseFast
fi

if [ ! -d "xml" ]; then
  mkdir -p "xml"
  echo "Directory xml created"
fi

# Delete old file
if [ -f "xml/vk.xml" ]; then
    rm "xml/vk.xml"
fi

# Download new vk.xml
echo "Downloading vk.xml"
curl -o xml/vk.xml https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/refs/heads/main/registry/vk.xml

# Run the generator
zig-out/bin/vulkan-zig-generator xml/vk.xml $OUTPUT/vk.zig
echo "Vulkan updated!"
# Now the vk.zig should be in xml/vk.zig

# Cleanup
cd ..
rm -rf "vulkan-zig"
