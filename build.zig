const std = @import("std");

pub fn build(b: *std.Build) void {

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "vulkan",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    exe.linkSystemLibrary("glfw");
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

    const run_exe = b.addRunArtifact(exe);

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);
}
