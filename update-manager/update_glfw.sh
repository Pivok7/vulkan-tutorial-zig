#!/bin/bash

OUTPUT="../lib"

if [ -d "vulkan-zig" ]; then
  rm -rf "vulkan-zig"
fi

git clone https://github.com/zig-gamedev/zglfw.git

mv zglfw/src/zglfw.zig $OUTPUT/

echo "GLFW updated!"

# Cleanup
rm -rf "zglfw"
