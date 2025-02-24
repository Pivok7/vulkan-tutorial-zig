#!/bin/bash
set -e

# Delete old file
if [ -f "xml/vk.xml" ]; then
    rm "xml/vk.xml"
fi

# Download new vk.xml
echo "Downloading vk.xml"
curl -o vk.xml https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/refs/heads/main/registry/vk.xml

echo "Updated vk.xml"
