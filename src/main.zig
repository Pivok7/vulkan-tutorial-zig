const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const vk_ctx = @import("vk_context.zig");
const za = @import("zalgebra");

const Allocator = std.mem.Allocator;
const c_allocator = std.heap.c_allocator;

const VkAssert = vk_ctx.VkAssert;
const BaseDispatch = vk_ctx.BaseDispatch;
const InstanceDispatch = vk_ctx.InstanceDispatch;
const DeviceDispatch = vk_ctx.DeviceDispatch;

pub const debug_mode = switch (builtin.mode) {
    .Debug => true,
    else => false,
};

const Vec2 = za.Vec2;
const Vec3 = za.Vec3;

const Vertex = struct{
    pos: Vec2 = Vec2.zero(),
    color: Vec3 = Vec3.zero(),

    pub fn getBindingDesciption() vk.VertexInputBindingDescription {
        return vk.VertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .input_rate = .vertex,
        };
    }

    pub fn getAttributeDescriptions() []const vk.VertexInputAttributeDescription {
        return &[_]vk.VertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = .r32g32b32_sfloat,
                .offset = @offsetOf(Vertex, "color"), 
            }
        };
    }
};

const vertices = [_]Vertex{
    Vertex{ .pos = Vec2.new(0.0, -0.5), .color = Vec3.new(1.0, 0.0, 0.0) },
    Vertex{ .pos = Vec2.new(0.5, 0.5), .color = Vec3.new(0.0, 1.0, 0.0) },
    Vertex{ .pos = Vec2.new(-0.5, 0.5), .color = Vec3.new(0.0, 0.0, 1.0) },
};

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
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
    app_name: [:0]const u8 = "Hello Triangle",
    allocator: Allocator = undefined,

    instance_extensions: std.ArrayList([*:0]const u8) = undefined,
    enable_validation_layers: bool = debug_mode,

    window_width: u32 = 800,
    window_height: u32 = 600, 
    max_frames_in_flight: u32 = 2,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    window: *glfw.Window = undefined,
    instance: vk.Instance = undefined,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null_handle,

    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = undefined,
    graphics_queue: vk.Queue = undefined,
    present_queue: vk.Queue = undefined,
    surface: vk.SurfaceKHR = undefined,

    swapchain: vk.SwapchainKHR = undefined,
    swapchain_images: []vk.Image = undefined,
    swapchain_image_views: []vk.ImageView = undefined,
    swapchain_image_format: vk.Format = undefined,
    swapchain_extent: vk.Extent2D = undefined,

    render_pass: vk.RenderPass = undefined,
    pipeline_layout: vk.PipelineLayout = undefined,
    graphics_pipeline: vk.Pipeline = undefined,
    swapchain_framebuffers: []vk.Framebuffer = undefined,

    vertex_buffer: vk.Buffer = undefined,
    vertex_buffer_memory: vk.DeviceMemory = undefined,

    command_pool: vk.CommandPool = undefined,
    command_buffers: []vk.CommandBuffer = undefined,
    
    current_frame: u32 = 0,
    image_available_semaphores: []vk.Semaphore = undefined,
    render_finished_semaphores: []vk.Semaphore = undefined,
    in_flight_fences: []vk.Fence = undefined,

    framebuffer_resized: bool = false,

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
        if (debug_mode) try self.setupDebugMessenger();
        try self.createSurface();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();
        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandPool();
        try self.createVertexBuffer();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }
    
    pub fn deinit(self: *@This()) void {
        for (0..self.max_frames_in_flight) |i| {
            self.vkd.destroyFence(self.device, self.in_flight_fences[i], null);
            self.vkd.destroySemaphore(self.device, self.render_finished_semaphores[i], null);
            self.vkd.destroySemaphore(self.device, self.image_available_semaphores[i], null);
        }
        
        self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        self.cleanupSwapchain();

        self.vkd.destroyBuffer(self.device, self.vertex_buffer, null);
        self.vkd.freeMemory(self.device, self.vertex_buffer_memory, null);

        self.vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        self.vkd.destroyRenderPass(self.device, self.render_pass, null);

        if (debug_mode) self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);

        self.vkd.destroyDevice(self.device, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
        std.log.info("Deinitialized Vulkan instance", .{});

        glfw.destroyWindow(self.window);
        std.log.info("Closed GLFW window", .{});
        glfw.terminate();
        std.log.debug("Terminated GLFW", .{});

        self.allocator.free(self.command_buffers);
        self.allocator.free(self.swapchain_image_views);
        self.allocator.free(self.swapchain_images);
        self.instance_extensions.deinit();
    }

    fn mainLoop(self: *@This()) !void {
        while (!glfw.windowShouldClose(self.window)) {

            while (glfw.getWindowAttribute(self.window, glfw.Window.Attribute.iconified)) {
                if (glfw.windowShouldClose(self.window)) break;
                glfw.waitEvents();
            }

            glfw.pollEvents();
            try self.drawFrame();
        }

        try self.vkd.deviceWaitIdle(self.device);
    }

    fn initWindow(self: *@This()) !void {
        try glfw.init();
        std.log.debug("Initialized GLFW", .{});

        glfw.windowHint(glfw.WindowHint.client_api, glfw.ClientApi.no_api);
        //glfw.windowHint(glfw.WindowHint.resizable, false);
        self.window = try glfw.createWindow(@intCast(self.window_width), @intCast(self.window_height), self.app_name, null);

        std.log.info("Created GLFW window", .{});

        self.window.setUserPointer(self);
        _ = glfw.setFramebufferSizeCallback(self.window, framebufferResizedCallback);
    }

    fn framebufferResizedCallback(window: *glfw.Window, width: c_int, height: c_int) callconv(.c) void {
        _ = width;
        _ = height;

        if (window.getUserPointer(@This())) |self| {
            self.framebuffer_resized = true;
        }
    }

    fn getRequiredExtensions(self: *@This()) !void {
        var glfw_extensions: [][*:0]const u8 = undefined;
        glfw_extensions = try glfw.getRequiredInstanceExtensions();

        var extensions = std.ArrayList([*:0]const u8).init(self.allocator);
        try extensions.appendSlice(glfw_extensions);
        if (debug_mode) try extensions.append(vk.extensions.ext_debug_utils.name);

        self.instance_extensions = extensions;
    }

    fn createInstance(self: *@This()) !void {
        self.vkb = try BaseDispatch.load(vk_ctx.glfwGetInstanceProcAddress);

        if (self.enable_validation_layers) {
            _ = try checkValidationLayerSupport(self);
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = self.app_name,
            .application_version = vk.makeApiVersion(0, 1, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(0, 1, 0, 0),
            .api_version = vk.makeApiVersion(0, 1, 3, 0),
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

    fn setupDebugMessenger(self: *@This()) !void {
        if (debug_mode) {
            const create_info = vk.DebugUtilsMessengerCreateInfoEXT{
                .message_severity = .{ .verbose_bit_ext = true, .error_bit_ext = true, .warning_bit_ext = true },
                .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
                .pfn_user_callback = &debugCallback,
            };

            self.debug_messenger = try self.vki.createDebugUtilsMessengerEXT(self.instance, &create_info, null);
        }
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
                    if (cStringEql(val_layer, &ava_layer.layer_name)) {
                        std.debug.print("\t [X] {s}\n", .{ava_layer.layer_name});
                    } else {
                        std.debug.print("\t [ ] {s}\n", .{ava_layer.layer_name});
                    }
                }
            }
        }

        for (validation_layers) |val_layer| {
            var found_layer: bool = false;

            for (available_layers) |ava_layer| {
                if (cStringEql(val_layer, &ava_layer.layer_name)) {
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
                if (cStringEql(dev_extension, &ava_extension.extension_name,)) {
                    found_layer = true;
                }
            }

            if (!found_layer) {
                std.log.err("Device extension \"{s}\" not found", .{dev_extension});
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

    fn cleanupSwapchain(self: *@This()) void {
        for (self.swapchain_framebuffers) |framebuffer| {
            self.vkd.destroyFramebuffer(self.device, framebuffer, null);
        }

        for (self.swapchain_image_views) |image_view| {
            self.vkd.destroyImageView(self.device, image_view, null);
        }

        self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);
    }

    fn recreateSwapchain(self: *@This()) !void {
        std.log.debug("Recreating swapchain", .{});

        try self.vkd.deviceWaitIdle(self.device);

        self.cleanupSwapchain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createFramebuffers();
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
        errdefer self.allocator.free(self.swapchain_images);

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
            glfw.getFramebufferSize(self.window, @ptrCast(@constCast(&self.window_width)), @ptrCast(@constCast(&self.window_height)));

            var actual_extent = vk.Extent2D{
                .width = self.window_width,
                .height = self.window_height,
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

    fn createRenderPass(self: *@This()) !void {
        const color_attachment = vk.AttachmentDescription{
            .format = self.swapchain_image_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };

        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        };

        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_ref),
        };

        const dependency = vk.SubpassDependency{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
        };

        const render_pass_info = vk.RenderPassCreateInfo{
            .s_type = .render_pass_create_info,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_attachment),
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = 1,
            .p_dependencies = @ptrCast(&dependency),
        };

        self.render_pass = try self.vkd.createRenderPass(self.device, &render_pass_info, null);
        std.log.debug("Created render pass", .{});
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

        const binding_description = Vertex.getBindingDesciption();
        const attribute_description = Vertex.getAttributeDescriptions();

        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
            .s_type = .pipeline_vertex_input_state_create_info,
            .vertex_binding_description_count = 1,
            .vertex_attribute_description_count = @intCast(attribute_description.len),
            .p_vertex_binding_descriptions = @ptrCast(&binding_description),
            .p_vertex_attribute_descriptions = attribute_description.ptr,
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .s_type = .pipeline_input_assembly_state_create_info,
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .s_type = .pipeline_viewport_state_create_info,
            .viewport_count = 1,
            .scissor_count = 1,
        };

        const dynamic_states = &[_]vk.DynamicState{
            vk.DynamicState.viewport,
            vk.DynamicState.scissor,
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .s_type = .pipeline_rasterization_state_create_info,
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .line_width = 1.0,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0.0,
            .depth_bias_clamp = 0.0,
            .depth_bias_slope_factor = 0.0,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .s_type = .pipeline_multisample_state_create_info,
            .sample_shading_enable = vk.FALSE,
            .rasterization_samples = .{ .@"1_bit" = true },
            .min_sample_shading = 1.0,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, 
                .b_bit = true, .a_bit = true, },
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .s_type = .pipeline_color_blend_state_create_info,
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_blend_attachment),
            .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
        };

        const dynamic_state = vk.PipelineDynamicStateCreateInfo{
            .s_type = .pipeline_dynamic_state_create_info,
            .dynamic_state_count = @intCast(dynamic_states.len),
            .p_dynamic_states = dynamic_states.ptr,
        };

        const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
            .s_type = .pipeline_layout_create_info,
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        };

        self.pipeline_layout = try self.vkd.createPipelineLayout(self.device, &pipeline_layout_info, null);
        std.log.debug("Created pipeline layout", .{});

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .s_type = .graphics_pipeline_create_info,
            .stage_count = 2,
            .p_stages = @ptrCast(&shader_stages),
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        const result = try self.vkd.createGraphicsPipelines(
            self.device,
            .null_handle,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&self.graphics_pipeline),
        );

        try VkAssert.withMessage(result, "Failed to create graphics pipeline");
        std.log.debug("Created graphics pipeline", .{});
    }

    fn createShaderModule(self: @This(), code: []const u8) !vk.ShaderModule {
        const create_info = vk.ShaderModuleCreateInfo{
            .s_type = .shader_module_create_info,
            .code_size = code.len,
            .p_code = @ptrCast(@alignCast(code.ptr)),
        };
        return try self.vkd.createShaderModule(self.device, &create_info, null);
    }

    fn createFramebuffers(self: *@This()) !void {
        self.swapchain_framebuffers = try self.allocator.alloc(vk.Framebuffer, self.swapchain_image_views.len);
        errdefer self.allocator.free(self.swapchain_framebuffers);

        for (self.swapchain_image_views, 0..) |image_view, i| {
            const attachments = [_]vk.ImageView{
                image_view,
            };

            const framebuffer_info = vk.FramebufferCreateInfo{
                .s_type = .framebuffer_create_info,
                .render_pass = self.render_pass,
                .attachment_count = 1,
                .p_attachments = &attachments,
                .width = self.swapchain_extent.width,
                .height = self.swapchain_extent.height,
                .layers = 1,
            };

            self.swapchain_framebuffers[i] = try self.vkd.createFramebuffer(self.device, &framebuffer_info, null);
        }
        std.log.debug("Created framebuffers", .{});
    }

    fn createCommandPool(self: *@This()) !void {
        const queue_family_indices = try self.findQueueFamilies(self.physical_device);

        const pool_info = vk.CommandPoolCreateInfo{
            .s_type = .command_pool_create_info,
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        };

        self.command_pool = try self.vkd.createCommandPool(self.device, &pool_info, null);
        std.log.debug("Created command pool", .{});
    }

    fn createVertexBuffer(self: *@This()) !void {
        const buffer_info = vk.BufferCreateInfo{
            .s_type = .buffer_create_info,
            .size = @sizeOf(Vertex) * vertices.len,
            .usage = .{ .vertex_buffer_bit = true },
            .sharing_mode = .exclusive,
        };

        self.vertex_buffer = try self.vkd.createBuffer(self.device, &buffer_info, null);
        std.log.debug("Created vertex buffer", .{});

        const mem_requirements = self.vkd.getBufferMemoryRequirements(self.device, self.vertex_buffer);

        const alloc_info = vk.MemoryAllocateInfo{
            .s_type = .memory_allocate_info,
            .allocation_size = mem_requirements.size,
            .memory_type_index = try self.findMemoryType(
                mem_requirements.memory_type_bits,
                vk.MemoryPropertyFlags{ 
                    .host_visible_bit = true,
                    .host_coherent_bit = true, 
                }),
        };

        self.vertex_buffer_memory = try self.vkd.allocateMemory(self.device, &alloc_info, null);
        try self.vkd.bindBufferMemory(self.device, self.vertex_buffer, self.vertex_buffer_memory, 0);

        const data = try self.vkd.mapMemory(self.device, self.vertex_buffer_memory, 0, buffer_info.size, .{});
        std.mem.copyForwards(u8, @as([*]u8, @ptrCast(data.?))[0..buffer_info.size], std.mem.sliceAsBytes(&vertices));
        defer self.vkd.unmapMemory(self.device, self.vertex_buffer_memory);
    }

    fn findMemoryType(self: *@This(), type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        const mem_properties = self.vki.getPhysicalDeviceMemoryProperties(self.physical_device);

        for (0..mem_properties.memory_type_count) |i| {
            if (
                // Some evil magic to make compiler happy
                (type_filter & (@as(u64, 1) << @intCast(i)) != 0)
                and (vk.MemoryPropertyFlags.intersect(mem_properties.memory_types[i].property_flags, properties) == properties)
            ) {
                return @intCast(i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    fn createCommandBuffers(self: *@This()) !void {
        const alloc_info = vk.CommandBufferAllocateInfo{
            .s_type = .command_buffer_allocate_info,
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = self.max_frames_in_flight,
        };

        self.command_buffers = try self.allocator.alloc(vk.CommandBuffer, self.max_frames_in_flight);
        errdefer self.allocator.free(self.command_buffers);

        try self.vkd.allocateCommandBuffers(self.device, &alloc_info, self.command_buffers.ptr);
        std.log.debug("Created command buffers", .{});
    }

    fn recordCommandBuffer(self: *@This(), command_buffer: vk.CommandBuffer, image_index: u32) !void {
        const begin_info = vk.CommandBufferBeginInfo{
            .s_type = .command_buffer_begin_info,
            .flags = .{},
            .p_inheritance_info = null,
        };

        try self.vkd.beginCommandBuffer(command_buffer, &begin_info);

        const clear_color = vk.ClearValue{ .color = .{ .float_32 = [4]f32{ 0.0, 0.0, 0.0, 1.0 } } };

        const render_pass_info = vk.RenderPassBeginInfo{
            .s_type = .render_pass_begin_info,
            .render_pass = self.render_pass,
            .framebuffer = self.swapchain_framebuffers[image_index],
            .render_area = .{
                .offset = .{ .x = 0, .y = 0},
                .extent = self.swapchain_extent,
            },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear_color),
        };

        self.vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");

            self.vkd.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);

            const viewport = vk.Viewport{
                .x = 0.0,
                .y = 0.0,
                .width = @floatFromInt(self.swapchain_extent.width),
                .height = @floatFromInt(self.swapchain_extent.height),
                .min_depth = 0.0,
                .max_depth = 1.0,
            };
            self.vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&viewport));

            const scissor = vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain_extent,
            };
            self.vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&scissor));

            const vertex_buffers = [_]vk.Buffer{self.vertex_buffer};
            const offsets = [_]vk.DeviceSize{0};
            self.vkd.cmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);

            self.vkd.cmdDraw(command_buffer, 3, 1, 0, 0);

        self.vkd.cmdEndRenderPass(command_buffer);

        try self.vkd.endCommandBuffer(command_buffer);
    }

    fn createSyncObjects(self: *@This()) !void {
        const sempahore_info = vk.SemaphoreCreateInfo{
            .s_type = .semaphore_create_info,
        };

        const fence_info = vk.FenceCreateInfo{
            .s_type = .fence_create_info,
            .flags = .{ .signaled_bit = true },
        };

        self.image_available_semaphores = try self.allocator.alloc(vk.Semaphore, self.max_frames_in_flight);
        errdefer self.allocator.free(self.image_available_semaphores);

        self.render_finished_semaphores = try self.allocator.alloc(vk.Semaphore, self.max_frames_in_flight);
        errdefer self.allocator.free(self.render_finished_semaphores);

        self.in_flight_fences = try self.allocator.alloc(vk.Fence, self.max_frames_in_flight);
        errdefer self.allocator.free(self.in_flight_fences);

        for (0..self.max_frames_in_flight) |i| {
            self.image_available_semaphores[i] = try self.vkd.createSemaphore(self.device, &sempahore_info, null);
            self.render_finished_semaphores[i] = try self.vkd.createSemaphore(self.device, &sempahore_info, null);
            self.in_flight_fences[i] = try self.vkd.createFence(self.device, &fence_info, null);
        }

        std.log.debug("Created sync objects", .{});
    }

    fn drawFrame(self: *@This()) !void {
        defer self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;

        var result = try self.vkd.waitForFences(self.device, 1, @ptrCast(&self.in_flight_fences[self.current_frame]), vk.TRUE, std.math.maxInt(u64));
        try VkAssert.withMessage(result, "Waiting for fence failed");

        const next_image = self.vkd.acquireNextImageKHR(
            self.device,
            self.swapchain,
            std.math.maxInt(u64),
            self.image_available_semaphores[self.current_frame],
            .null_handle,
        ) catch |err| {
            switch (err) {
                error.OutOfDateKHR => {
                    try self.recreateSwapchain();
                    return;
                },
                else => return err,
            }
        };

        switch (next_image.result) {
            .success, .suboptimal_khr => {},
            else => return error.FailedToAcquireSwapchainImage,
        }

        try self.vkd.resetFences(self.device, 1, @ptrCast(&self.in_flight_fences[self.current_frame]));

        try self.vkd.resetCommandBuffer(self.command_buffers[self.current_frame], .{});
        try self.recordCommandBuffer(self.command_buffers[self.current_frame], next_image.image_index);

        const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores[self.current_frame]};
        const wait_stages = [_]vk.PipelineStageFlags{ .{ .color_attachment_output_bit = true} };
        const signal_semaphores = [_]vk.Semaphore{self.render_finished_semaphores[self.current_frame]};

        const submit_info = vk.SubmitInfo{
            .s_type = .submit_info,
            .wait_semaphore_count = @intCast(wait_semaphores.len),
            .p_wait_semaphores = &wait_semaphores,
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.command_buffers[self.current_frame]),
            .signal_semaphore_count = @intCast(signal_semaphores.len),
            .p_signal_semaphores = &signal_semaphores,
        };

        try self.vkd.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit_info), self.in_flight_fences[self.current_frame]);

        const swapchains = [_]vk.SwapchainKHR{self.swapchain};

        const present_info = vk.PresentInfoKHR{
            .s_type = .present_info_khr,
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&signal_semaphores),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&swapchains),
            .p_image_indices = @ptrCast(&next_image.image_index),
            .p_results = null,
        };

        result = self.vkd.queuePresentKHR(self.present_queue, &present_info) catch |err| {
            switch (err) {
                error.OutOfDateKHR => {
                    try self.recreateSwapchain();
                    return;
                },
                else => return err,
            }
        };

        switch (result) {
            .success => {},
            .suboptimal_khr => {
                self.framebuffer_resized = false;
                try self.recreateSwapchain();
                return;
            },
            else => return error.FailedToPresentSwapchainImage,
        }

        if (self.framebuffer_resized) {
            self.framebuffer_resized = false;
            try self.recreateSwapchain();
            return;
        }    
    }
};

fn createDebugMessengerCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return .{
        .message_severity = .{ .verbose_bit_ext = true, .error_bit_ext = true, .warning_bit_ext = true },
        .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
        .pfn_user_callback = &debugCallback,
    };
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = p_user_data;
    _ = message_type;
    _ = message_severity;

    if (p_callback_data) |data| {
        std.log.debug("{s}", .{data.p_message.?});
    }

    return vk.TRUE;
}

fn cStringEql(str_1: [*:0]const u8, str_2: [*]const u8) bool {
    var i: usize = 0;
    while (str_1[i] == str_2[i]) : (i += 1) {
        if (str_1[i] == '\x00') return true;
    }
    return false;
}

pub fn main() !void {
    var app = HelloTriangleApplication.init(c_allocator);
    defer app.deinit();
    try app.run();
}
