const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const c = @cImport({
    @cInclude("string.h");
});

const Allocator = std.mem.Allocator;
const c_allocator = std.heap.c_allocator;

/// To construct base, instance and device wrappers for vulkan-zig, you need to pass a list of 'apis' to it.
const apis: []const vk.ApiInfo = &.{
    vk.features.version_1_0,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
};

/// Next, pass the `apis` to the wrappers to create dispatch tables.
const BaseDispatch = vk.BaseWrapper(apis);
const InstanceDispatch = vk.InstanceWrapper(apis);
const DeviceDispatch = vk.DeviceWrapper(apis);

// Also create some proxying wrappers, which also have the respective handles
const Instance = vk.InstanceProxy(apis);
const Device = vk.DeviceProxy(apis);


const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
    //"VK_LAYER_LUNARG_api_dump",
};

const device_extensions = [_][*:0]const u8{
    vk.swapchain_extension_name,
};

const HelloTriangleApplication = struct {
    app_name: [:0]const u8 = "Vulkan App",
    allocator: Allocator = undefined,
    enable_validation_layers: bool = switch (builtin.mode) {
        .Debug, .ReleaseSafe => true,
        else => false,
    },

    vkb: BaseDispatch = undefined,

    window: ?*glfw.Window = null,
    instance: Instance = undefined,

    pub fn init(allocator: Allocator) @This() {

        return @This(){
            .allocator = allocator,
        };
    }

    pub fn run(self: *@This()) !void {
        try self.initWindow();
        try self.initVulkan();
        try self.mainLoop();
    }

    pub fn deinit(self: *@This()) void {
        self.instance.destroyInstance(null);
        std.log.info("Deinitialized Vulkan instance", .{});

        glfw.destroyWindow(self.window.?);
        std.log.info("Closed GLFW window", .{});
        glfw.terminate();
        std.log.info("Terminated GLFW", .{});
    }

    fn mainLoop(self: *@This()) !void {
        while (!glfw.windowShouldClose(self.window.?)) {
            glfw.pollEvents();
        }
    }

    fn initWindow(self: *@This()) !void {
        try glfw.init();
        std.log.info("Initialized GLFW", .{});

        glfw.windowHint(glfw.WindowHint.client_api, glfw.ClientApi.no_api);
        glfw.windowHint(glfw.WindowHint.resizable, false);
        self.window = try glfw.createWindow(WIDTH, HEIGHT, self.app_name, null);

        std.log.info("Created GLFW window", .{});
    }

    fn initVulkan(self: *@This()) !void {
        try createVkInstance(self);
    }
    
    fn createVkInstance(self: *@This()) !void {
        // The glfwGetInstanceProcAddress is defined in lib/vk_context.zig
        // and is not a part of vulkan-zig
        self.vkb = try BaseDispatch.load(vk.glfwGetInstanceProcAddress);
        const vki = try self.allocator.create(InstanceDispatch);
        errdefer self.allocator.destroy(vki);

        if (self.enable_validation_layers) {
            _ = try checkValidationLayerSupport(self);
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = self.app_name,
            .application_version = vk.makeApiVersion(1, 0, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0),
            .api_version = vk.API_VERSION_1_2,
        };

        const glfw_extensions = try glfw.getRequiredInstanceExtensions();

        const create_info = try self.vkb.createInstance(&.{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = if (self.enable_validation_layers) @intCast(validation_layers.len) else 0,
            .pp_enabled_layer_names = if (self.enable_validation_layers) @ptrCast(&validation_layers) else null,
            .enabled_extension_count = @intCast(glfw_extensions.len),
            .pp_enabled_extension_names = glfw_extensions.ptr,
        }, null);

        // Some magic I don't really understand. It creates a vulkan instance
        vki.* = try InstanceDispatch.load(create_info, self.vkb.dispatch.vkGetInstanceProcAddr);
        self.instance = Instance.init(create_info, vki);
        std.log.info("Initialized Vulkan instance", .{});
    }

    fn checkValidationLayerSupport(self: *@This()) !bool {
        var layer_count: u32 = 0;
        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

        const available_layers = try c_allocator.alloc(vk.LayerProperties, layer_count);
        defer c_allocator.free(available_layers);

        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, @ptrCast(available_layers));

        // Print available layers if debug mode is on
        if (builtin.mode == .Debug) {
            std.debug.print("Available validation layers ({}): \n", .{layer_count});
            for (available_layers) |layer| {
                std.debug.print("\t{s}\n", .{layer.layer_name}); 
            }
        }

        // We check if validation layers that we specified at the start are present.
        for (validation_layers) |val_layer| {
            var found_layer: bool = false;

            for (available_layers) |ava_layer| {
                if (c.strcmp(&ava_layer.layer_name, val_layer) == 0) {
                    found_layer = true;
                    break;
                }
            }
            if (!found_layer) {
                std.log.err("Validation layer \"{s}\" not found", .{val_layer});
                return error.ValidationLayerNotAvailable;
            }
        }
      
        return true;
    }
};

pub fn main() !void {
    var app = HelloTriangleApplication.init(c_allocator);
    defer app.deinit();
    try app.run();
}
