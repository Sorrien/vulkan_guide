use std::ffi::CString;

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{self, Device},
};
use ash_bootstrap::{
    create_logical_device, BootstrapSwapchain, InstanceBuilder, PhysicalDeviceSelector,
    QueueFamilyIndices, SwapchainBuilder, SwapchainSupportDetails,
};
//use gpu_allocator::vulkan::*;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub mod ash_bootstrap;
pub mod debug;

const FRAME_OVERLAP: usize = 2;

pub struct BaseVulkanState {
    pub window: Window,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_indices: QueueFamilyIndices,
    pub graphics_queue: vk::Queue,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub swapchain_support: SwapchainSupportDetails,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub debug_utils: DebugUtils,
    pub msaa_samples: vk::SampleCountFlags,
}

impl BaseVulkanState {
    pub fn new(window: Window, application_title: String) -> Self {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let entry = ash::Entry::linked();
        let instance = InstanceBuilder::new()
            .entry(entry.clone())
            .application_name(application_title)
            .api_version(vk::API_VERSION_1_3)
            .raw_display_handle(window.raw_display_handle())
            .enable_validation_layers(enable_validation_layers)
            .build();

        let (debug_utils, debug_messenger) = crate::debug::debug_utils(&entry, &instance);

        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .expect("failed to create window surface!");

        // Application can't function without geometry shaders or the graphics queue family or anisotropy (we could remove anisotropy)
        let device_features = vk::PhysicalDeviceFeatures::default()
            .geometry_shader(true)
            .sampler_anisotropy(true);
        let features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true);

        let selector = PhysicalDeviceSelector::new(&instance, &surface_loader, &surface);

        let required_extensions = vec![CString::from(Swapchain::NAME)];
        let bootstrap_physical_device = selector
            .set_required_extensions(required_extensions.clone())
            .set_required_features(device_features)
            .set_required_features_12(features12)
            .set_required_features_13(features13)
            .select()
            .expect("failed to select physical device!");

        let device = create_logical_device(
            &instance,
            bootstrap_physical_device.physical_device,
            &bootstrap_physical_device.queue_family_indices,
            required_extensions,
            device_features,
            features12,
            features13,
        )
        .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            device.get_device_queue(
                bootstrap_physical_device
                    .queue_family_indices
                    .graphics_family
                    .unwrap() as u32,
                0,
            )
        };

        Self {
            window,
            instance,
            physical_device: bootstrap_physical_device.physical_device,
            device,
            queue_family_indices: bootstrap_physical_device.queue_family_indices,
            graphics_queue,
            surface_loader,
            surface,
            swapchain_support: bootstrap_physical_device.swapchain_support_details,
            debug_messenger,
            debug_utils,
            msaa_samples: bootstrap_physical_device.max_sample_count,
        }
    }

    pub fn create_swapchain(&self, window_width: u32, window_height: u32) -> BootstrapSwapchain {
        let bootstrap_swapchain = SwapchainBuilder::new(
            self.instance.clone(),
            self.device.clone(),
            self.surface,
            self.swapchain_support.clone(),
            self.queue_family_indices,
        )
        .desired_extent(window_width, window_height)
        .desired_present_mode(vk::PresentModeKHR::FIFO)
        .desired_surface_format(
            vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
        )
        .add_image_usage_flags(vk::ImageUsageFlags::TRANSFER_DST)
        .build();
        bootstrap_swapchain
    }

    pub fn create_command_pool(
        &self,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<vk::CommandPool, vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(self.queue_family_indices.graphics_family.unwrap() as u32);

        unsafe { self.device.create_command_pool(&pool_info, None) }
    }

    pub fn create_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe { self.device.allocate_command_buffers(&alloc_info) }
    }

    pub fn create_frame_data(&self, count: usize) -> Vec<FrameData> {
        (0..count)
            .map(|_| {
                let command_pool = self
                    .create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .expect("failed to create command pool!");
                let command_buffer = self
                    .create_command_buffers(command_pool, 1)
                    .expect("failed to create command buffer!")[0];

                let render_fence = self
                    .create_fence(vk::FenceCreateFlags::SIGNALED)
                    .expect("failed to create render fence!");
                let swapchain_semaphore = self
                    .create_semaphore(vk::SemaphoreCreateFlags::empty())
                    .expect("failed to create swapchain semaphore!");
                let render_semaphore = self
                    .create_semaphore(vk::SemaphoreCreateFlags::empty())
                    .expect("failed to create swapchain semaphore!");

                FrameData {
                    command_pool,
                    command_buffer,
                    render_fence,
                    swapchain_semaphore,
                    render_semaphore,
                }
            })
            .collect::<Vec<_>>()
    }

    pub fn create_fence(&self, flags: vk::FenceCreateFlags) -> Result<vk::Fence, vk::Result> {
        let fence_create_info = vk::FenceCreateInfo::default().flags(flags);
        unsafe { self.device.create_fence(&fence_create_info, None) }
    }

    pub fn create_semaphore(
        &self,
        flags: vk::SemaphoreCreateFlags,
    ) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default().flags(flags);
        unsafe { self.device.create_semaphore(&create_info, None) }
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image(image);

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        unsafe {
            self.device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
    }
}

impl Drop for BaseVulkanState {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            #[cfg(feature = "validation_layers")]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct VulkanEngine {
    base: BaseVulkanState,
    frames: Vec<FrameData>,
    frame_number: usize,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_loader: Swapchain,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl VulkanEngine {
    pub fn new(window: Window, application_title: String) -> Self {
        let base_vulkan_state = BaseVulkanState::new(window, application_title);

        /* let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,  // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        }); */

        let window_size = base_vulkan_state.window.inner_size();
        let window_height = window_size.height;
        let window_width = window_size.width;

        let bootstrap_swapchain = base_vulkan_state.create_swapchain(window_width, window_height);

        Self {
            base: base_vulkan_state,
            frames: vec![],
            frame_number: 0,

            swapchain: bootstrap_swapchain.swapchain,
            swapchain_loader: bootstrap_swapchain.swapchain_loader,
            format: bootstrap_swapchain.format,
            extent: bootstrap_swapchain.extent,
            swapchain_images: bootstrap_swapchain.swapchain_images,
            swapchain_image_views: bootstrap_swapchain.swapchain_image_views,
        }
    }

    pub fn init_commands(&mut self) {
        self.frames = self.base.create_frame_data(FRAME_OVERLAP);
    }

    pub fn init_window(
        width: u32,
        height: u32,
        title: &str,
    ) -> (EventLoop<()>, winit::window::Window) {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)
            .unwrap();

        (event_loop, window)
    }

    pub fn get_current_frame(&self) -> FrameData {
        self.frames[self.frame_number % FRAME_OVERLAP]
    }

    pub fn run(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        self.main_loop(event_loop)
    }

    fn main_loop(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        event_loop.run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe { self.base.device.device_wait_idle() }
                    .expect("failed to wait for idle on exit!");
                elwt.exit()
            }
            Event::AboutToWait => {
                //AboutToWait is the new MainEventsCleared
                self.base.window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id: _,
            } => {
                let size = self.base.window.inner_size();
                //don't attempt to draw a frame in window size is 0zs
                if size.height > 0 && size.width > 0 {
                    self.draw();
                }
            }
            Event::WindowEvent {
                window_id: _,
                event: WindowEvent::Resized(_new_size),
            } => {
                //self.framebuffer_resized = true;
            }
            _ => (),
        })
    }

    pub fn draw(&mut self) {
        let current_frame = self.get_current_frame();
        let fences = [current_frame.render_fence];
        unsafe { self.base.device.wait_for_fences(&fences, true, u64::MAX) }
            .expect("failed to wait for render fence!");
        unsafe { self.base.device.reset_fences(&fences) }.expect("failed to reset render fence!");

        //acquire next swapchain image
        let (swapchain_image_index, _) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                current_frame.swapchain_semaphore,
                vk::Fence::null(),
            )
        }
        .expect("failed to get swapchain image!");

        let cmd = current_frame.command_buffer;

        unsafe {
            self.base
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
        }
        .expect("failed to reset command buffer!");

        unsafe {
            self.base.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("failed to begin command buffer!");

        let swapchain_image = self.swapchain_images[swapchain_image_index as usize];
        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let clear_value = vk::ClearColorValue {
            float32: [0., 0., (self.frame_number as f32 / 120.).sin().abs(), 1.],
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_array_layer(0)
            .layer_count(vk::REMAINING_ARRAY_LAYERS);
        let ranges = [range];
        unsafe {
            self.base.device.cmd_clear_color_image(
                cmd,
                swapchain_image,
                vk::ImageLayout::GENERAL,
                &clear_value,
                &ranges,
            )
        };

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        unsafe { self.base.device.end_command_buffer(cmd) }.expect("failed to end command buffer!");

        //let command_info = Self::command_buffer_submit_info(cmd);
        let command_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(current_frame.command_buffer)
            .device_mask(0);
        /*         let wait_info = Self::semaphore_submit_info(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR,
            current_frame.swapchain_semaphore,
        );
        let signal_info = Self::semaphore_submit_info(
            vk::PipelineStageFlags2::ALL_GRAPHICS,
            current_frame.render_semaphore,
        );
         */
        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(current_frame.swapchain_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)
            .device_index(0)
            .value(1);

        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(current_frame.render_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
            .device_index(0)
            .value(1);
        let command_buffer_infos = [command_info];
        let signal_semaphore_infos = [signal_info];
        let wait_semaphore_infos = [wait_info];

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .command_buffer_infos(&command_buffer_infos);
        //let submit = Self::submit_info(&command_buffer_infos, &signal_infos, &wait_infos);
        let submits = [submit];
        unsafe {
            self.base.device.queue_submit2(
                self.base.graphics_queue,
                &submits,
                current_frame.render_fence,
            )
        }
        .expect("queue command submit failed!");

        let swapchains = [self.swapchain];
        let render_semaphores = [current_frame.render_semaphore];
        let swapchain_image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&render_semaphores)
            .image_indices(&swapchain_image_indices);
        unsafe {
            self.swapchain_loader
                .queue_present(self.base.graphics_queue, &present_info)
        }
        .expect("failed to queue present to swapchain!");

        self.frame_number += 1;
    }

    pub fn semaphore_submit_info(
        stage_mask: vk::PipelineStageFlags2,
        semaphore: vk::Semaphore,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(stage_mask)
            .device_index(0)
            .value(1)
    }

    pub fn command_buffer_submit_info(
        command_buffer: vk::CommandBuffer,
    ) -> vk::CommandBufferSubmitInfo<'static> {
        vk::CommandBufferSubmitInfo::default()
            .command_buffer(command_buffer)
            .device_mask(0)
    }

    /*     pub fn submit_info(
           command_buffer_info: vk::CommandBufferSubmitInfo,
           signal_semaphore_info: vk::SemaphoreSubmitInfo,
           wait_semaphore_info: vk::SemaphoreSubmitInfo,
       ) -> vk::SubmitInfo2<'static> {
           vk::SubmitInfo2::default()
               .wait_semaphore_infos(&[wait_semaphore_info])
               .signal_semaphore_infos(&[signal_semaphore_info])
               .command_buffer_infos(&[command_buffer_info])
       }
    */
    pub fn submit_info<'a>(
        command_buffer_infos: &'a [vk::CommandBufferSubmitInfo<'a>],
        signal_semaphore_infos: &'a [vk::SemaphoreSubmitInfo<'a>],
        wait_semaphore_infos: &'a [vk::SemaphoreSubmitInfo<'a>],
    ) -> vk::SubmitInfo2<'a> {
        vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_semaphore_infos)
            .signal_semaphore_infos(signal_semaphore_infos)
            .command_buffer_infos(command_buffer_infos)
    }

    pub fn destroy_swapchain(&self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };

        for i in 0..self.swapchain_image_views.len() {
            let swapchain_image_view = self.swapchain_image_views[i];
            unsafe {
                self.base
                    .device
                    .destroy_image_view(swapchain_image_view, None)
            };
        }
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.device_wait_idle() }.expect("failed to wait for device idle!");

        for i in 0..self.frames.len() {
            let frame = self.frames[i];
            unsafe {
                self.base
                    .device
                    .destroy_command_pool(frame.command_pool, None);

                self.base.device.destroy_fence(frame.render_fence, None);
                self.base
                    .device
                    .destroy_semaphore(frame.render_semaphore, None);
                self.base
                    .device
                    .destroy_semaphore(frame.swapchain_semaphore, None);
            };
        }

        self.destroy_swapchain();
    }
}

#[derive(Clone, Copy)]
pub struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    render_fence: vk::Fence,
    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
}
