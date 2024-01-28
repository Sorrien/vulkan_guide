use std::{
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::{extensions::khr::Swapchain, vk, Entry};
use ash_bootstrap::{
    Instance, InstanceBuilder, LogicalDevice, PhysicalDeviceSelector, QueueFamilyIndices,
    VulkanSurface,
};
use debug::DebugMessenger;
use gpu_allocator::vulkan::*;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use swapchain::{MySwapchain, SwapchainBuilder, SwapchainSupportDetails};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub mod ash_bootstrap;
pub mod debug;
pub mod swapchain;

const FRAME_OVERLAP: usize = 2;

pub struct BaseVulkanState {
    pub msaa_samples: vk::SampleCountFlags,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_support: SwapchainSupportDetails,
    pub allocator: Arc<Mutex<Allocator>>,
    pub graphics_queue: vk::Queue,
    pub device: Arc<LogicalDevice>,
    pub physical_device: vk::PhysicalDevice,
    pub debug_messenger: DebugMessenger,
    pub surface: Arc<VulkanSurface>,
    pub instance: Arc<Instance>,
    pub entry: Entry,
    pub window: Window,
}

impl BaseVulkanState {
    pub fn new(window: Window, application_title: String) -> Self {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let entry = unsafe { ash::Entry::load() }.expect("vulkan entry failed to load!");
        let instance = InstanceBuilder::new()
            .entry(entry.clone())
            .application_name(application_title)
            .api_version(vk::API_VERSION_1_3)
            .raw_display_handle(window.raw_display_handle())
            .enable_validation_layers(enable_validation_layers)
            .build();

        let debug_messenger =
            DebugMessenger::new(&entry, instance.clone(), enable_validation_layers);

        let surface = VulkanSurface::new(
            &entry,
            instance.clone(),
            window.raw_display_handle(),
            window.raw_window_handle(),
        )
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

        let selector = PhysicalDeviceSelector::new(instance.clone(), surface.clone());

        let required_extensions = vec![CString::from(Swapchain::NAME)];
        let bootstrap_physical_device = selector
            .set_required_extensions(required_extensions.clone())
            .set_required_features(device_features)
            .set_required_features_12(features12)
            .set_required_features_13(features13)
            .select()
            .expect("failed to select physical device!");

        let device = LogicalDevice::new(
            instance.clone(),
            bootstrap_physical_device.physical_device,
            &bootstrap_physical_device.queue_family_indices,
            required_extensions,
            device_features,
            features12,
            features13,
        )
        .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            device.handle.get_device_queue(
                bootstrap_physical_device
                    .queue_family_indices
                    .graphics_family
                    .unwrap() as u32,
                0,
            )
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle.clone(),
            device: device.handle.clone(),
            physical_device: bootstrap_physical_device.physical_device.clone(),
            debug_settings: Default::default(),
            buffer_device_address: true, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })
        .expect("failed to create allocator!");

        Self {
            entry,
            window,
            instance,
            physical_device: bootstrap_physical_device.physical_device,
            device,
            queue_family_indices: bootstrap_physical_device.queue_family_indices,
            graphics_queue,
            surface,
            swapchain_support: bootstrap_physical_device.swapchain_support_details,
            debug_messenger,
            msaa_samples: bootstrap_physical_device.max_sample_count,
            allocator: Arc::new(Mutex::new(allocator)),
        }
    }

    pub fn create_swapchain(&self, window_width: u32, window_height: u32) -> MySwapchain {
        let bootstrap_swapchain = SwapchainBuilder::new(
            self.instance.clone(),
            self.device.clone(),
            self.surface.clone(),
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

        unsafe { self.device.handle.create_command_pool(&pool_info, None) }
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

        unsafe { self.device.handle.allocate_command_buffers(&alloc_info) }
    }

    pub fn create_frame_data(&self, count: usize) -> Vec<Arc<FrameData>> {
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
                FrameData::new(
                    self.device.clone(),
                    command_pool,
                    command_buffer,
                    render_fence,
                    swapchain_semaphore,
                    render_semaphore,
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn create_fence(&self, flags: vk::FenceCreateFlags) -> Result<vk::Fence, vk::Result> {
        let fence_create_info = vk::FenceCreateInfo::default().flags(flags);
        unsafe { self.device.handle.create_fence(&fence_create_info, None) }
    }

    pub fn create_semaphore(
        &self,
        flags: vk::SemaphoreCreateFlags,
    ) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default().flags(flags);
        unsafe { self.device.handle.create_semaphore(&create_info, None) }
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
                .handle
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
    }

    pub fn create_image(
        &mut self,
        img_extent: vk::Extent2D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
    ) -> (vk::Image, Allocation) {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(img_extent.into())
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(num_samples);

        let image = unsafe { self.device.handle.create_image(&image_info, None) }
            .expect("failed to create image!");

        let mem_requirements = unsafe { self.device.handle.get_image_memory_requirements(image) };

        let is_linear = tiling == vk::ImageTiling::LINEAR;
        let image_allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_requirements,
                location: memory_location,
                linear: is_linear,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            })
            .expect("failed to allocate image!");

        unsafe {
            self.device
                .handle
                .bind_image_memory(image, image_allocation.memory(), 0)
        }
        .expect("failed to bind image memory!");

        (image, image_allocation)
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let component_mapping = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(component_mapping)
            .subresource_range(subresource_range);

        let image_view = unsafe {
            self.device
                .handle
                .create_image_view(&image_view_create_info, None)
        }
        .expect("failed to create image view!");
        image_view
    }
}

pub fn find_memory_type(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    for i in 0..(mem_properties.memory_type_count as usize) {
        if (type_filter & (1 << i)) != 0
            && (mem_properties.memory_types[i].property_flags & properties) == properties
        {
            return i as u32;
        }
    }

    panic!("failed to find suitable memory type!");
}

pub struct VulkanEngine {
    frame_number: usize,
    draw_image: AllocatedImage,
    frames: Vec<Arc<FrameData>>,
    pub swapchain: MySwapchain,
    base: BaseVulkanState,
}

impl VulkanEngine {
    pub fn new(window: Window, application_title: String) -> Self {
        let mut base = BaseVulkanState::new(window, application_title);

        let window_size = base.window.inner_size();
        let window_height = window_size.height;
        let window_width = window_size.width;

        let swapchain = base.create_swapchain(window_width, window_height);

        let draw_image_extent = vk::Extent2D {
            width: window_width,
            height: window_height,
        };
        let draw_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let (draw_image, draw_image_allocation) = base.create_image(
            draw_image_extent,
            draw_image_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        );

        let draw_image_view = base.create_image_view(
            draw_image,
            draw_image_format,
            vk::ImageAspectFlags::COLOR,
            1,
        );

        let draw_image_allocated = AllocatedImage {
            image: draw_image,
            image_view: draw_image_view,
            extent: draw_image_extent.into(),
            format: draw_image_format,
            allocation: ManuallyDrop::new(draw_image_allocation),
            device: base.device.clone(),
            allocator: base.allocator.clone(),
        };

        Self {
            base,
            frames: vec![],
            frame_number: 0,
            draw_image: draw_image_allocated,
            swapchain,
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

    pub fn get_current_frame(&self) -> Arc<FrameData> {
        self.frames[self.frame_number % FRAME_OVERLAP].clone()
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
                unsafe { self.base.device.handle.device_wait_idle() }
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
        unsafe {
            self.base
                .device
                .handle
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .expect("failed to wait for render fence!");
        unsafe { self.base.device.handle.reset_fences(&fences) }
            .expect("failed to reset render fence!");

        //acquire next swapchain image
        let (swapchain_image_index, _) = unsafe {
            self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
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
                .handle
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
        }
        .expect("failed to reset command buffer!");

        unsafe {
            self.base.device.handle.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("failed to begin command buffer!");

        let swapchain_image = self.swapchain.swapchain_images[swapchain_image_index as usize];
        self.base.transition_image_layout(
            cmd,
            self.draw_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        self.draw_background(cmd);

        self.base.transition_image_layout(
            cmd,
            self.draw_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        copy_image_to_image(
            self.base.device.clone(),
            cmd,
            self.draw_image.image,
            swapchain_image,
            self.draw_image.extent,
            self.swapchain.extent,
        );

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        unsafe { self.base.device.handle.end_command_buffer(cmd) }
            .expect("failed to end command buffer!");

        let command_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(current_frame.command_buffer)
            .device_mask(0);
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
        let submits = [submit];
        unsafe {
            self.base.device.handle.queue_submit2(
                self.base.graphics_queue,
                &submits,
                current_frame.render_fence,
            )
        }
        .expect("queue command submit failed!");

        let swapchains = [self.swapchain.swapchain];
        let render_semaphores = [current_frame.render_semaphore];
        let swapchain_image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&render_semaphores)
            .image_indices(&swapchain_image_indices);
        unsafe {
            self.swapchain
                .swapchain_loader
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

    pub fn draw_background(&self, cmd: vk::CommandBuffer) {
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
            self.base.device.handle.cmd_clear_color_image(
                cmd,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
                &clear_value,
                &ranges,
            )
        };
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for device idle!");
    }
}

#[derive(Clone)]
pub struct FrameData {
    device: Arc<LogicalDevice>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    render_fence: vk::Fence,
    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
}

impl FrameData {
    pub fn new(
        device: Arc<LogicalDevice>,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        render_fence: vk::Fence,
        swapchain_semaphore: vk::Semaphore,
        render_semaphore: vk::Semaphore,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            command_pool,
            command_buffer,
            render_fence,
            swapchain_semaphore,
            render_semaphore,
        })
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_command_pool(self.command_pool, None);

            self.device.handle.destroy_fence(self.render_fence, None);
            self.device
                .handle
                .destroy_semaphore(self.render_semaphore, None);
            self.device
                .handle
                .destroy_semaphore(self.swapchain_semaphore, None);
        };
    }
}

pub struct AllocatedImage {
    device: Arc<LogicalDevice>,
    allocator: Arc<Mutex<Allocator>>,
    image: vk::Image,
    image_view: vk::ImageView,
    allocation: ManuallyDrop<Allocation>,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_image_view(self.image_view, None);
            self.device.handle.destroy_image(self.image, None);
        }
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        self.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .expect("failed to free memory for allocated image!");
    }
}

pub fn copy_image_to_image(
    device: Arc<LogicalDevice>,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
) {
    let blit_region = vk::ImageBlit2::default()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: src_size.width as i32,
                y: src_size.height as i32,
                z: 1,
            },
        ])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: dst_size.width as i32,
                y: dst_size.height as i32,
                z: 1,
            },
        ])
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        );

    let regions = [blit_region];
    let blit_info = vk::BlitImageInfo2::default()
        .dst_image(dst)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_image(src)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&regions);

    unsafe { device.handle.cmd_blit_image2(cmd, &blit_info) };
}
