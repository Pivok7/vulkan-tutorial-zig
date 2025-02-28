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
    
    exe.linkLibC();

    const os_tag = target.query.os_tag orelse builtin.target.os.tag;
    switch (os_tag) {
        .windows => {
            // Disable console window in Release mode
            if (optimize != .Debug) exe.subsystem = .Windows;

            // This library is needed for static linking with glfw
            exe.linkSystemLibrary("gdi32");
            exe.addObjectFile(b.path("lib_windows/libglfw3.a"));
        },
        .linux => {
            exe.linkSystemLibrary("glfw3");
        },
        else => {
            std.log.warn("{} may be unsuported", .{os_tag});
            exe.linkSystemLibrary("glfw3");
        },
    }

    const vulkan = b.dependency("vulkan_zig", .{
        .target = target,
        .optimize = optimize,
        .registry = b.path("vk_source/vk.xml"),
    }).module("vulkan-zig");
    exe.root_module.addImport("vulkan", vulkan);

    const zglfw = b.dependency("zglfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zglfw", zglfw.module("root"));

    const zalgebra = b.dependency("zalgebra", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zalgebra", zalgebra.module("zalgebra"));

    
    b.installArtifact(exe);

    // Shader compilation
    const compile_vert_shader = b.addSystemCommand(&.{
        "glslc",
        "src/shaders/shader.vert",
        "-o",
        "src/shaders/vert.spv",
    });

    const compile_frag_shader = b.addSystemCommand(&.{
        "glslc",
        "src/shaders/shader.frag",
        "-o",
        "src/shaders/frag.spv"
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
