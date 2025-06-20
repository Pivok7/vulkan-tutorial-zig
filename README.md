# Vulkan tutorial in zig!

Project based on this tutorial: https://vulkan-tutorial.com/

This branch covers only the "Drawing a triangle" chapter. For full tutorial check the full-tutorial branch

## Building

You will need:

* Zig compiler (newest version)

* Vulkan SDK and glslc <br>

(optional) Update vk.xml by running 
```bash
cd vk_source
./update_xml.sh
```

Then run `zig build run`

If you are on Windows and get an error related to glslc then try adding glslc.exe to your system path so that it can be invoked from cmd

Validation layers are enabled in Debug mode

## Third party libraries used in this project

* vulkan-zig: https://github.com/Snektron/vulkan-zig.git <br>
Licensed under the MIT License.


* zglfw: https://github.com/zig-gamedev/zglfw.git <br>
Licensed under the MIT License.


* Parts of the code borrowed from here: <br>
https://github.com/Cy-Tek/vulkan-tutorial-zig/tree/main


