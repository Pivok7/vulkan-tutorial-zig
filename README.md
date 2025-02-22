# Vulkan project in zig!

## Building

You will need:

* zig compiler (newest version)

* Vulkan SDK and glslc <br>
Check out this tutorial: https://vulkan-tutorial.com/Development_environment



(optional) Update libraries by running 
```bash
cd update_manager
./update_all.sh
```

Then run `zig build run`

If you are on Windows and get an error related to glslc then try adding glslc.exe to you system path so that it can be invoked from cmd

## Third party libraries used in this project

* vulkan-zig: https://github.com/Snektron/vulkan-zig.git <br>
Licensed under the MIT License.


* zglfw: https://github.com/zig-gamedev/zglfw.git <br>
Licensed under the MIT License.


* Parts of the code borrowed from here: <br>
https://github.com/Cy-Tek/vulkan-tutorial-zig/tree/main


