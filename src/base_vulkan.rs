use std::{
    ffi::{CStr, CString},
    fs::File,
    path::Path,
    sync::{Arc, Mutex},
};

use ash::{extensions::khr::Swapchain, util::read_spv, vk, Entry};
use ash_bootstrap::{
    Instance, InstanceBuilder, LogicalDevice, PhysicalDeviceSelector, QueueFamilyIndices,
    VulkanSurface,
};
use debug::DebugMessenger;
use gpu_allocator::vulkan::*;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use swapchain::{MySwapchain, SwapchainBuilder, SwapchainSupportDetails};
use winit::window::Window;

use crate::{
    ash_bootstrap, debug,
    descriptors::{Descriptor, DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio},
    swapchain, AllocatedImage,
};

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

    pub fn create_shader_module<P>(&self, path: P) -> vk::ShaderModule
    where
        P: AsRef<Path>,
    {
        let mut spv_file = File::open(path).unwrap();
        let shader_code = read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            self.device
                .handle
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };
        shader_module
    }

    pub fn init_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> Pipeline {
        let gradient_pipeline = self.init_background_pipelines(draw_image_descriptor_layout);
        gradient_pipeline
    }

    pub fn init_background_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> Pipeline {
        let set_layouts = [draw_image_descriptor_layout];
        let compute_layout = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let gradient_pipeline_layout = unsafe {
            self.device
                .handle
                .create_pipeline_layout(&compute_layout, None)
        }
        .expect("failed to create gradient pipeline layout!");

        let compute_draw_shader = self.create_shader_module("shaders/gradient.comp.spv");
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(compute_draw_shader)
            .name(shader_entry_name);

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(gradient_pipeline_layout)
            .stage(stage_info);
        let create_infos = [compute_pipeline_create_info];
        let pipelines = unsafe {
            self.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &create_infos,
                None,
            )
        }
        .expect("failed to create gradient pipeline!");
        let gradient_pipeline = pipelines[0];

        unsafe {
            self.device
                .handle
                .destroy_shader_module(compute_draw_shader, None)
        };

        let pipeline = Pipeline {
            device: self.device.clone(),
            pipeline: gradient_pipeline,
            pipeline_layout: gradient_pipeline_layout,
        };

        pipeline
    }

    pub fn init_descriptors(
        &mut self,
        global_descriptor_allocator: &mut DescriptorAllocator,
        draw_image: &AllocatedImage,
    ) -> Descriptor {
        let draw_image_desc_ty = vk::DescriptorType::STORAGE_IMAGE;
        let sizes = vec![PoolSizeRatio {
            desc_type: draw_image_desc_ty,
            ratio: 1.,
        }];

        global_descriptor_allocator.init_pool(10, sizes);

        let draw_image_descriptor_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, draw_image_desc_ty)
            .build(self.device.clone(), vk::ShaderStageFlags::COMPUTE)
            .expect("failed to create draw image descriptor layout!");

        let draw_image_descriptors = global_descriptor_allocator
            .allocate(vec![draw_image_descriptor_layout])
            .unwrap()[0];

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(draw_image.image_view);

        let image_infos = [image_info];
        let draw_image_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(draw_image_descriptors)
            .descriptor_count(1)
            .descriptor_type(draw_image_desc_ty)
            .image_info(&image_infos);

        let desc_writes = [draw_image_write];
        let desc_copies = [];
        unsafe {
            self.device
                .handle
                .update_descriptor_sets(&desc_writes, &desc_copies)
        }

        Descriptor::new(
            self.device.clone(),
            draw_image_descriptors,
            draw_image_descriptor_layout,
        )
    }

    pub fn init_immediate_command(&self) -> ImmediateCommand {
        let command_pool = self
            .create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .expect("failed to create immediate submit command pool!");

        let command_buffer = self
            .create_command_buffers(command_pool, 1)
            .expect("failed to create immediate submit command buffer!")[0];

        let fence = self
            .create_fence(vk::FenceCreateFlags::SIGNALED)
            .expect("failed to create immediate submit fence!");

        ImmediateCommand::new(self.device.clone(), fence, command_buffer, command_pool)
    }
}

pub struct Pipeline {
    device: Arc<LogicalDevice>,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.handle.destroy_pipeline(self.pipeline, None)
        }
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

#[derive(Clone)]
pub struct FrameData {
    device: Arc<LogicalDevice>,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub render_fence: vk::Fence,
    pub swapchain_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
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

pub struct ImmediateCommand {
    device: Arc<LogicalDevice>,
    pub fence: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
}

impl ImmediateCommand {
    pub fn new(
        device: Arc<LogicalDevice>,
        fence: vk::Fence,
        command_buffer: vk::CommandBuffer,
        command_pool: vk::CommandPool,
    ) -> Self {
        Self {
            device,
            fence,
            command_buffer,
            command_pool,
        }
    }
}

impl Drop for ImmediateCommand {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_fence(self.fence, None);
            self.device
                .handle
                .destroy_command_pool(self.command_pool, None)
        };
    }
}
