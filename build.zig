const std = @import("std");
const builtin = @import("builtin");

fn buildLog(comptime string: []const u8, args: anytype) !void {
    std.debug.print("build: " ++ string, args);
}

pub fn build(b: *std.Build) void {

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "vulkan",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Libraries
    if (builtin.target.os.tag == .windows) {
        exe.addIncludePath(b.path("lib_windows/include"));
        exe.addLibraryPath(b.path("lib_windows/lib"));
        if (optimize == .ReleaseSmall or optimize == .ReleaseFast) {
            exe.subsystem = .Windows;
        }

        // We move dynamic libraries to where the executable is located
        b.installFile("lib_windows/lib/glfw3.dll", "bin/glfw3.dll");
        
    }

    exe.linkSystemLibrary("glfw3");
    exe.linkLibC();

    const vulkan = b.addModule("vulkan", .{
        .root_source_file = b.path("lib/vk.zig")
    });
    exe.root_module.addImport("vulkan", vulkan);

    const zglfw = b.addModule("zglfw", .{
        .root_source_file = b.path("lib/zglfw.zig")
    });
    exe.root_module.addImport("zglfw", zglfw);
    
    b.installArtifact(exe);

    // Shader compilation
    const compile_vert_shader = b.addSystemCommand(&.{
        "glslc",
        "shaders/shader.vert",
        "-o",
        "zig-out/bin/shaders/vert.spv",
    });

    const compile_frag_shader = b.addSystemCommand(&.{
        "glslc",
        "shaders/shader.frag",
        "-o",
        "zig-out/bin/shaders/frag.spv"
    });

    exe.step.dependOn(&compile_vert_shader.step);
    if (optimize == .Debug) try buildLog("Compiled vertex shader\n", .{});

    exe.step.dependOn(&compile_frag_shader.step);
    if (optimize == .Debug) try buildLog("Compiled fragment shader\n", .{});

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);
}
