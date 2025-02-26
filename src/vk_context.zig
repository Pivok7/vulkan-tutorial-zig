const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");

const debug_mode = @import("main.zig").debug_mode;

pub const VkAssert = struct {
    pub fn basic(result: vk.Result) !void {
        switch (result) {
            .success => return,
            else => return error.Unknown,
        }
    }

    pub fn withMessage(result: vk.Result, message: []const u8) !void {
        switch (result) {
            .success => return,
            else => {
                std.log.err("{s} {s}", .{ @tagName(result), message });
                return error.Unknown;
            },
        }
    }
};

/// To construct base, instance and device wrappers for vulkan-zig, you need to pass a list of 'apis' to it.
const apis: []const vk.ApiInfo = &.{
    .{
        .base_commands = .{
            .getInstanceProcAddr = true,
            .createInstance = true,
            .enumerateInstanceLayerProperties = true,
        },
        .instance_commands = .{
            .getDeviceProcAddr = true,
            .destroyInstance = true,
            .createDebugUtilsMessengerEXT = if (debug_mode) true else false,
            .destroyDebugUtilsMessengerEXT = if (debug_mode) true else false,
            .destroySurfaceKHR = true,
            .createDevice = true,
            .enumeratePhysicalDevices = true,
            .enumerateDeviceExtensionProperties = true,
            .getPhysicalDeviceProperties = true,
            .getPhysicalDeviceQueueFamilyProperties = true,
            .getPhysicalDeviceSurfaceSupportKHR = true,
            .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
            .getPhysicalDeviceSurfaceFormatsKHR = true,
            .getPhysicalDeviceSurfacePresentModesKHR = true,
        },
        .device_commands = .{
            .getDeviceQueue = true,
            .destroyDevice = true,
            .createSwapchainKHR = true,
            .getSwapchainImagesKHR = true,
            .destroySwapchainKHR = true,
            .createImageView = true,
            .destroyImageView = true,
            .createShaderModule = true,
            .destroyShaderModule = true,
            .createPipelineLayout = true,
            .destroyPipelineLayout = true,
            .createRenderPass = true,
            .destroyRenderPass = true,
            .createGraphicsPipelines = true,
            .destroyPipeline = true,
            .createFramebuffer = true,
            .destroyFramebuffer = true,
            .createCommandPool = true,
            .destroyCommandPool = true,
            .allocateCommandBuffers = true,
            .beginCommandBuffer = true,
            .cmdBeginRenderPass = true,
            .cmdBindPipeline = true,
            .cmdSetViewport = true,
            .cmdSetScissor = true,
            .cmdDraw = true,
            .cmdEndRenderPass = true,
            .endCommandBuffer = true,
            .createSemaphore = true,
            .destroySemaphore = true,
            .createFence = true,
            .destroyFence = true,
            .waitForFences = true,
            .resetFences = true,
            .acquireNextImageKHR = true,
            .resetCommandBuffer = true,
            .queueSubmit = true,
            .queuePresentKHR = true,
            .deviceWaitIdle = true,
        },
    },
};

/// Next, pass the `apis` to the wrappers to create dispatch tables.
pub const BaseDispatch = vk.BaseWrapper(apis);
pub const InstanceDispatch = vk.InstanceWrapper(apis);
pub const DeviceDispatch = vk.DeviceWrapper(apis);

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
pub extern fn glfwGetPhysicalDevicePresentationSupport(instance: vk.Instance, pdev: vk.PhysicalDevice, queuefamily: u32) c_int;
pub extern fn glfwCreateWindowSurface(instance: vk.Instance, window: *glfw.Window, allocation_callbacks: ?*const vk.AllocationCallbacks, surface: *vk.SurfaceKHR) vk.Result;
