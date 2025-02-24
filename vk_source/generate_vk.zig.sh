#!/bin/bash
set -e

./update_xml.sh

if [ -d "vulkan-zig" ]; then
  rm -rf "vulkan-zig"
fi

git clone https://github.com/Snektron/vulkan-zig.git

cd vulkan-zig

# Build the program if there is no exe
if [ ! -f "zig-out/bin/vulkan-zig-generator" ]; then
    zig build -Doptimize=ReleaseFast
fi

# Run the generator
zig-out/bin/vulkan-zig-generator ../vk.xml ../vk.zig
echo "Generated vk.zig"
# Now the vk.zig should be in xml/vk.zig

# Cleanup
cd ..
rm -rf "vulkan-zig"
