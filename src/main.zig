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
        .Debug => true,
        else => false,
    },

    window_width: u32 = 800,
    window_height: u32 = 600, 

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    instance_extensions: std.ArrayList([*:0]const u8) = undefined,

    window: *glfw.Window = undefined,
    instance: vk.Instance = undefined,

    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = undefined,
    graphics_queue: vk.Queue = undefined,
    present_queue: vk.Queue = undefined,
    surface: vk.SurfaceKHR = undefined,

    swapchain: vk.SwapchainKHR = .null_handle,
    swapchain_images: []vk.Image = undefined,
    swapchain_image_views: []vk.ImageView = undefined,
    swapchain_image_format: vk.Format = undefined,
    swapchain_extent: vk.Extent2D = undefined,

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
        try self.createLogicalDevice();
        try self.createSwapChain();
        try self.createImageViews();
        try self.createGraphicsPipeline();
    }
    
    pub fn deinit(self: *@This()) void {
        
        for (self.swapchain_image_views) |image_view| {
            self.vkd.destroyImageView(self.device, image_view, null);
        }

        self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
        std.log.info("Deinitialized Vulkan instance", .{});

        glfw.destroyWindow(self.window);
        std.log.info("Closed GLFW window", .{});
        glfw.terminate();
        std.log.debug("Terminated GLFW", .{});

        self.allocator.free(self.swapchain_image_views);
        self.allocator.free(self.swapchain_images);
        self.instance_extensions.deinit();
    }

    fn mainLoop(self: *@This()) !void {
        while (!glfw.windowShouldClose(self.window)) {
            glfw.pollEvents();
        }
    }

    fn initWindow(self: *@This()) !void {
        try glfw.init();
        std.log.debug("Initialized GLFW", .{});

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
        const result = vk_ctx.glfwCreateWindowSurface(self.instance, self.window, null, &self.surface);
        try VkAssert.withMessage(result, "Failed to create window surface!");
    }

    fn checkValidationLayerSupport(self: *@This()) !bool {
        var layer_count: u32 = 0;
        var result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

        const available_layers = try c_allocator.alloc(vk.LayerProperties, layer_count);
        defer c_allocator.free(available_layers);

        result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, @ptrCast(available_layers));
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

        // Print validation layers if debug mode is on
        if (builtin.mode == .Debug and validation_layers.len > 0) {
            std.debug.print("Active validation layers ({d}): \n", .{validation_layers.len});
            for (validation_layers) |val_layer| {
                for (available_layers) |ava_layer| {
                    if (c.strcmp(&ava_layer.layer_name, val_layer) == 0) {
                        std.debug.print("\t [X] {s}\n", .{ava_layer.layer_name});
                    } else {
                        std.debug.print("\t [ ] {s}\n", .{ava_layer.layer_name});
                    }
                }
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

        if (self.physical_device == .null_handle) {
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

    fn createLogicalDevice(self: *@This()) !void {
        const indices: QueueFamilyIndices = try self.findQueueFamilies(self.physical_device);

        var unique_queue_families = std.ArrayList(u32).init(self.allocator);
        defer unique_queue_families.deinit();

        const all_queue_families = &[_]u32{
            indices.graphics_family.?,
            indices.present_family.?
        };

        for (all_queue_families) |queue_family| {
            for (unique_queue_families.items) |item| {
                if (item == queue_family) {
                    continue;
                }
            }
            try unique_queue_families.append(queue_family);
        }

        var queue_create_infos = try self.allocator.alloc(vk.DeviceQueueCreateInfo, unique_queue_families.items.len);

        const queue_priority: f32 = 1.0;
        for (unique_queue_families.items, 0..) |queue_family, i| {
            queue_create_infos[i] = vk.DeviceQueueCreateInfo{
                .s_type = .device_queue_create_info,
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = @ptrCast(&queue_priority),
            };
        }
        
        var device_features: vk.PhysicalDeviceFeatures = .{};        

        var create_info = vk.DeviceCreateInfo{
            .s_type = .device_create_info,
            .p_queue_create_infos = queue_create_infos.ptr,
            .queue_create_info_count = 1,
            .p_enabled_features = &device_features,
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = @ptrCast(&device_extensions),
            .enabled_layer_count = if (self.enable_validation_layers) @intCast(validation_layers.len) else 0,
            .pp_enabled_layer_names = if (self.enable_validation_layers) @ptrCast(&validation_layers) else null,
        };

        self.device = try self.vki.createDevice(self.physical_device, &create_info, null);
        self.vkd = try DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr);

        self.graphics_queue = self.vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
        self.present_queue = self.vkd.getDeviceQueue(self.device, indices.present_family.?, 0);

        std.log.debug("Created logical device", .{});
    }

    fn createSwapChain(self: *@This()) !void {
        var swap_chain_support: SwapChainSupportDetails = try self.querySwapChainSupport(self.physical_device);
        defer swap_chain_support.deinit();

        const surface_format: vk.SurfaceFormatKHR = chooseSwapSurfaceFormat(swap_chain_support.formats.items);
        const present_mode: vk.PresentModeKHR = chooseSwapPresentMode(swap_chain_support.present_modes.items);
        const extent: vk.Extent2D = self.chooseSwapExtent(&swap_chain_support.capabilities);

        var image_count: u32 = swap_chain_support.capabilities.min_image_count + 1;
        if (swap_chain_support.capabilities.max_image_count > 0 and image_count > swap_chain_support.capabilities.max_image_count) {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        const indices: QueueFamilyIndices = try self.findQueueFamilies(self.physical_device);
        const queue_family_indices = [_]u32{indices.graphics_family.?, indices.present_family.?};

        var create_info = vk.SwapchainCreateInfoKHR{
            .s_type = .swapchain_create_info_khr,
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = undefined,
            .pre_transform = swap_chain_support.capabilities.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = .null_handle,
        };

        if (indices.graphics_family != indices.present_family) {
            create_info.image_sharing_mode = .concurrent;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = @ptrCast(&queue_family_indices);
        } else {
            create_info.image_sharing_mode = .exclusive;
        }

        self.swapchain = try self.vkd.createSwapchainKHR(self.device, &create_info, null);

        var result = try self.vkd.getSwapchainImagesKHR(self.device, self.swapchain, &image_count, null);
        try VkAssert.withMessage(result, "Failed to get swapchain images.");

        self.swapchain_images = try self.allocator.alloc(vk.Image, image_count);

        result = try self.vkd.getSwapchainImagesKHR(self.device, self.swapchain, &image_count, self.swapchain_images.ptr);
        try VkAssert.withMessage(result, "Failed to get swapchain images.");

        self.swapchain_image_format = surface_format.format;
        self.swapchain_extent = extent;

        std.log.debug("Created swapchain", .{});
    }

    fn chooseSwapSurfaceFormat(available_formats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
        for (available_formats) |available_format| {
            if (available_format.format == .b8g8r8a8_srgb and 
                available_format.color_space == .srgb_nonlinear_khr) 
            {
                return available_format;
            }
        }
        return available_formats[0];
    }

    fn chooseSwapPresentMode(available_present_modes: []vk.PresentModeKHR) vk.PresentModeKHR {
        for (available_present_modes) |available_present_mode| {
            if (available_present_mode == .mailbox_khr) {
                return available_present_mode;
            }
        }
        return .fifo_khr;
    }

    fn chooseSwapExtent(self: @This(), capabilities: *vk.SurfaceCapabilitiesKHR) vk.Extent2D {
        if (capabilities.current_extent.width != std.math.maxInt(u32)) {
            return capabilities.current_extent;
        } else {
            var width: u32 = 0;
            var height: u32 = 0;
            glfw.getFramebufferSize(self.window, @ptrCast(&width), @ptrCast(&height));

            var actual_extent = vk.Extent2D{
                .width = width,
                .height = height,
            };

            actual_extent.width = std.math.clamp(actual_extent.width, capabilities.min_image_extent.width, capabilities.max_image_extent.width);
            actual_extent.height = std.math.clamp(actual_extent.height, capabilities.min_image_extent.height, capabilities.max_image_extent.height);

            return actual_extent;
        }
    }

    fn createImageViews(self: *@This()) !void {
        self.swapchain_image_views = try self.allocator.alloc(vk.ImageView, self.swapchain_images.len);

        for (self.swapchain_images, 0..) |image, i| {
            const create_info = vk.ImageViewCreateInfo{
                .s_type = .image_view_create_info,
                .image = image,
                .view_type = .@"2d",
                .format = self.swapchain_image_format,
                .components = .{ 
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            };

            self.swapchain_image_views[i] = try self.vkd.createImageView(self.device, &create_info, null);
        }
        std.log.debug("Created image views", .{});
    }

    fn createGraphicsPipeline(self: *@This()) !void {
        const vert_file align(@alignOf(u32)) = @embedFile("shaders/vert.spv").*;
        const frag_file align(@alignOf(u32)) = @embedFile("shaders/frag.spv").*;

        const vert_shader_module = try self.createShaderModule(&vert_file);
        defer self.vkd.destroyShaderModule(self.device, vert_shader_module, null);
        std.log.debug("Created vertex shader module", .{});

        const frag_shader_module = try self.createShaderModule(&frag_file);
        defer self.vkd.destroyShaderModule(self.device, frag_shader_module, null);
        std.log.debug("Created fragment shader module", .{});

        const vert_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .s_type = .pipeline_shader_stage_create_info,
            .stage = .{ .vertex_bit = true },
            .module = vert_shader_module,
            .p_name = "main",
        };

        const frag_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .s_type = .pipeline_shader_stage_create_info,
            .stage = .{ .fragment_bit = true },
            .module = frag_shader_module,
            .p_name = "main",
        };

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
            vert_shader_stage_info,
            frag_shader_stage_info,
        };
        _ = shader_stages;
    }

    fn createShaderModule(self: @This(), code: []const u8) !vk.ShaderModule {
        const create_info = vk.ShaderModuleCreateInfo{
            .s_type = .shader_module_create_info,
            .code_size = code.len,
            .p_code = @ptrCast(@alignCast(code.ptr)),
        };
        return try self.vkd.createShaderModule(self.device, &create_info, null);
    }
};

pub fn main() !void {
    var app = HelloTriangleApplication.init(c_allocator);
    defer app.deinit();
    try app.run();
}
