const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const vk_ctx = @import("vk_context.zig");
const c = @cImport({
    @cInclude("string.h");
});

const Allocator = std.mem.Allocator;
const c_allocator = std.heap.c_allocator;

const VkAssert = vk_ctx.VkAssert;
const BaseDispatch = vk_ctx.BaseDispatch;
const InstanceDispatch = vk_ctx.InstanceDispatch;
const DeviceDispatch = vk_ctx.DeviceDispatch;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
    //"VK_LAYER_LUNARG_api_dump",
};

const device_extensions = [_][*:0]const u8{
    vk.extensions.khr_swapchain.name,
};


const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    fn isComplete(self : @This()) bool {
        return (self.graphics_family != null and self.present_family != null);
    }
};

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR = undefined,
    formats: std.ArrayList(vk.SurfaceFormatKHR) = undefined,
    present_modes: std.ArrayList(vk.PresentModeKHR) = undefined,

    pub fn init(allocator: Allocator) !@This() {
        return SwapChainSupportDetails{
            .formats = std.ArrayList(vk.SurfaceFormatKHR).init(allocator),
            .present_modes = std.ArrayList(vk.PresentModeKHR).init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.formats.deinit();
        self.present_modes.deinit();
    }
};

const HelloTriangleApplication = struct {
    app_name: [:0]const u8 = "Vulkan App",
    allocator: Allocator = undefined,
    enable_validation_layers: bool = switch (builtin.mode) {
        .Debug, .ReleaseSafe => true,
        else => false,
    },

    window_width: u32 = 800,
    window_height: u32 = 600, 

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    instance_extensions: std.ArrayList([*:0]const u8) = undefined,

    window: ?*glfw.Window = null,
    instance: vk.Instance = undefined,

    physical_device: ?vk.PhysicalDevice = null,
    surface: vk.SurfaceKHR = undefined,

    //-------------------------------------------
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

    fn initVulkan(self: *@This()) !void {
        try self.getRequiredExtensions();
        try self.createInstance();
        try self.createSurface();
        try self.pickPhysicalDevice();
    }
    
    pub fn deinit(self: *@This()) void {
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
        std.log.info("Deinitialized Vulkan instance", .{});

        glfw.destroyWindow(self.window.?);
        std.log.info("Closed GLFW window", .{});
        glfw.terminate();
        std.log.info("Terminated GLFW", .{});

        self.instance_extensions.deinit();
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
        self.window = try glfw.createWindow(@intCast(self.window_width), @intCast(self.window_height), self.app_name, null);

        std.log.info("Created GLFW window", .{});
    }

    fn createInstance(self: *@This()) !void {
        self.vkb = try BaseDispatch.load(vk_ctx.glfwGetInstanceProcAddress);

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

        const create_info = vk.InstanceCreateInfo{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = if (self.enable_validation_layers) @intCast(validation_layers.len) else 0,
            .pp_enabled_layer_names = if (self.enable_validation_layers) @ptrCast(&validation_layers) else null,
            .enabled_extension_count = @intCast(self.instance_extensions.items.len),
            .pp_enabled_extension_names = self.instance_extensions.items.ptr,
        };

        self.instance = try self.vkb.createInstance(&create_info, null);
        std.log.info("Initialized Vulkan instance", .{});

        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn createSurface(self: *@This()) !void {
        if (vk_ctx.glfwCreateWindowSurface(self.instance, self.window.?, null, &self.surface) != vk.Result.success) {
            std.log.err("Failed to create window surface!", .{});
            return error.SurfaceCreationFail;
        }
    }

    fn checkValidationLayerSupport(self: *@This()) !bool {
        var layer_count: u32 = 0;
        var result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

        const available_layers = try c_allocator.alloc(vk.LayerProperties, layer_count);
        defer c_allocator.free(available_layers);

        result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, @ptrCast(available_layers));
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

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

    fn getRequiredExtensions(self: *@This()) !void {
        var glfw_extensions: [][*:0]const u8 = undefined;
        glfw_extensions = try glfw.getRequiredInstanceExtensions();

        var extensions = std.ArrayList([*:0]const u8).init(self.allocator);
        try extensions.appendSlice(glfw_extensions);

        self.instance_extensions = extensions;
    }

    fn pickPhysicalDevice(self: *@This()) !void {
        var device_count: u32 = 0;
        var result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");

        const available_devices = try self.allocator.alloc(vk.PhysicalDevice, device_count);
        defer self.allocator.free(available_devices);

        result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, available_devices.ptr);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");
    
        for (available_devices) |device| {
            if (try self.isDeviceSuitable(device)) {
                self.physical_device = device;
                
                std.log.info("Device: {s}", .{self.vki.getPhysicalDeviceProperties(device).device_name});
                break;
            }
        }

        if (self.physical_device == null) {
            std.log.err("Failed to find a suitable GPU!", .{});
            return error.SuitableGPUNotFound;
        }
    }

    fn isDeviceSuitable(self: *@This(), device: vk.PhysicalDevice) !bool {
        const indices: QueueFamilyIndices = try self.findQueueFamilies(device);
        
        const extensions_supported: bool = try self.checkDeviceExtensionSupport(device);

        var swap_chain_adequate: bool = false;
        if (extensions_supported) {
            var swap_chain_support: SwapChainSupportDetails = try self.querySwapChainSupport(device);
            defer swap_chain_support.deinit();
            swap_chain_adequate = swap_chain_support.formats.items.len != 0 and swap_chain_support.present_modes.items.len != 0;
        }

        return (indices.isComplete() and extensions_supported and swap_chain_adequate);

    }

    fn findQueueFamilies(self: *@This(), device: vk.PhysicalDevice) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{};

        var queue_family_count: u32 = 0;
        self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

        const available_queue_families = try self.allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer self.allocator.free(available_queue_families);
    
        self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, available_queue_families.ptr);

        for (available_queue_families, 0..) |queue_family, i| {
            if (queue_family.queue_flags.graphics_bit) {
                indices.graphics_family = @intCast(i);
            }

            if (try self.vki.getPhysicalDeviceSurfaceSupportKHR(device, @intCast(i), self.surface) == vk.TRUE) {
                indices.present_family = @intCast(i);
            }

            if (indices.isComplete()) break;
        }

        return indices;
    }

    fn checkDeviceExtensionSupport(self: *@This(), device: vk.PhysicalDevice) !bool {
        var extension_count: u32 = 0;
        var result = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, null);
        try VkAssert.basic(result);

        const available_extensions = try self.allocator.alloc(vk.ExtensionProperties, extension_count);
        defer self.allocator.free(available_extensions);
    
        result = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);
        try VkAssert.basic(result);

        for (device_extensions) |dev_extension| {
            var found_layer: bool = false;

            for (available_extensions) |ava_extension| {
                if (c.strcmp(&ava_extension.extension_name, dev_extension) == 0) {
                    found_layer = true;
                    break;
                }
            }
            if (!found_layer) {
                std.log.err("Validation layer \"{s}\" not found", .{dev_extension});
                return false;
            }
        }
        return true;
    }

    fn querySwapChainSupport(self: *@This(), device: vk.PhysicalDevice) !SwapChainSupportDetails {
        var details = try SwapChainSupportDetails.init(self.allocator);

        details.capabilities = try self.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface);
        
        var format_count: u32 = 0;
        var result = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, null);
        try VkAssert.withMessage(result, "Failed to get physical device surface formats.");

        if (format_count > 0) {
            try details.formats.resize(format_count);
            result = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, @ptrCast(details.formats.items.ptr));
            try VkAssert.withMessage(result, "Failed to get physical device surface formats.");
        }

        var present_mode_count: u32 = 0;
        result = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, null);
        try VkAssert.withMessage(result, "Failed to get physical device surface present modes.");

        if (present_mode_count > 0) {
            try details.present_modes.resize(present_mode_count);
            result = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, @ptrCast(details.present_modes.items.ptr));
            try VkAssert.withMessage(result, "Failed to get physical device surface present modes.");
        }

        return details;
    }

};

pub fn main() !void {
    var app = HelloTriangleApplication.init(c_allocator);
    defer app.deinit();
    try app.run();
}
