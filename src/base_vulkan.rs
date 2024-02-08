use std::{
    ffi::{CStr, CString},
    fs::File,
    mem::size_of,
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
    swapchain, AllocatedImage, ComputeEffect, ComputePushConstants,
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
        let shader_code = read_spv(&mut spv_file).expect("Failed to read shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            self.device
                .handle
                .create_shader_module(&vertex_shader_info, None)
                .expect("shader module error")
        };
        shader_module
    }

    pub fn init_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> (Vec<ComputeEffect>, Arc<PipelineLayout>) {
        self.init_background_pipelines(draw_image_descriptor_layout)
    }

    pub fn init_background_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> (Vec<ComputeEffect>, Arc<PipelineLayout>) {
        let push_constant = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let set_layouts = [draw_image_descriptor_layout];
        let push_constant_ranges = [push_constant];
        let compute_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        /*
        let compute_effect_pipeline_layout = unsafe {
            self.device
                .handle
                .create_pipeline_layout(&compute_layout, None)
        }
        .expect("failed to create gradient pipeline layout!"); */
        let compute_effect_pipeline_layout =
            PipelineLayout::new(self.device.clone(), compute_layout_create_info)
                .expect("failed to create compute effect pipeline layout!");

        let gradient_shader = self.create_shader_module("shaders/gradient.comp.spv");
        let sky_shader = self.create_shader_module("shaders/sky.comp.spv");

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(gradient_shader)
            .name(shader_entry_name);

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(compute_effect_pipeline_layout.handle)
            .stage(stage_info);

        let mut sky_pipeline_create_info = compute_pipeline_create_info.clone();
        sky_pipeline_create_info.stage.module = sky_shader;
        let create_infos = [compute_pipeline_create_info, sky_pipeline_create_info];

        let pipelines = unsafe {
            self.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &create_infos,
                None,
            )
        }
        .expect("failed to create gradient pipeline!");
        let gradient_pipeline = pipelines[0];
        let sky_pipeline = pipelines[1];

        let gradient = ComputeEffect {
            name: String::from("gradient"),
            pipeline: Pipeline::new(
                self.device.clone(),
                gradient_pipeline,
                compute_effect_pipeline_layout.clone(),
            ),
            data: ComputePushConstants {
                data1: glam::Vec4::new(1., 0., 0., 1.),
                data2: glam::Vec4::new(0., 0., 1., 1.),
                data3: glam::Vec4::ZERO,
                data4: glam::Vec4::ZERO,
            },
        };

        let sky = ComputeEffect {
            name: String::from("sky"),
            pipeline: Pipeline::new(
                self.device.clone(),
                sky_pipeline,
                compute_effect_pipeline_layout.clone(),
            ),
            data: ComputePushConstants {
                data1: glam::Vec4::new(0.1, 0.2, 0.4, 0.97),
                data2: glam::Vec4::ZERO,
                data3: glam::Vec4::ZERO,
                data4: glam::Vec4::ZERO,
            },
        };

        unsafe {
            self.device
                .handle
                .destroy_shader_module(gradient_shader, None);
            self.device.handle.destroy_shader_module(sky_shader, None);
        };

        (vec![gradient, sky], compute_effect_pipeline_layout)
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

    pub fn init_triangle_pipeline(
        &self,
        draw_image_format: vk::Format,
    ) -> (Pipeline, Arc<PipelineLayout>) {
        let triangle_vert_shader = self.create_shader_module("shaders/colored_triangle.vert.spv");
        let triangle_frag_shader = self.create_shader_module("shaders/colored_triangle.frag.spv");

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = PipelineLayout::new(self.device.clone(), pipeline_layout_info)
            .expect("failed to create triangle pipeline layout!");

        let vk_pipeline = PipelineBuilder::new(pipeline_layout.clone())
            .set_shaders(triangle_vert_shader, triangle_frag_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .set_multisampling_none()
            .disable_blending()
            .disable_depth_test()
            .set_color_attachment_format(draw_image_format)
            .set_depth_attachment_format(vk::Format::UNDEFINED)
            .build_pipeline(self.device.clone())
            .expect("failed to create triangle pipeline!");
        let pipeline = Pipeline::new(self.device.clone(), vk_pipeline, pipeline_layout.clone());

        unsafe {
            self.device
                .handle
                .destroy_shader_module(triangle_vert_shader, None);
            self.device
                .handle
                .destroy_shader_module(triangle_frag_shader, None);
        }

        (pipeline, pipeline_layout)
    }
}

pub struct Pipeline {
    device: Arc<LogicalDevice>,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: Arc<PipelineLayout>,
}

impl Pipeline {
    pub fn new(
        device: Arc<LogicalDevice>,
        pipeline: vk::Pipeline,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.pipeline, None);
        }
    }
}

pub struct PipelineLayout {
    device: Arc<LogicalDevice>,
    pub handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn new(
        device: Arc<LogicalDevice>,
        create_info: vk::PipelineLayoutCreateInfo,
    ) -> Result<Arc<PipelineLayout>, vk::Result> {
        let result = unsafe { device.handle.create_pipeline_layout(&create_info, None) };
        if let Ok(handle) = result {
            Ok(Arc::new(Self { device, handle }))
        } else {
            Err(result.err().unwrap())
        }
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_pipeline_layout(self.handle, None)
        };
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

pub struct PipelineBuilder<'a> {
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    pipeline_layout: Arc<PipelineLayout>,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    //render_info: vk::PipelineRenderingCreateInfo<'a>,
    color_attachment_format: vk::Format,
    depth_attachment_format: vk::Format,
}

impl PipelineBuilder<'_> {
    pub fn new(pipeline_layout: Arc<PipelineLayout>) -> Self {
        Self {
            shader_stages: vec![],
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default(),
            multisampling: vk::PipelineMultisampleStateCreateInfo::default(),
            pipeline_layout,
            depth_stencil: vk::PipelineDepthStencilStateCreateInfo::default(),
            //render_info: vk::PipelineRenderingCreateInfo::default(),
            color_attachment_format: vk::Format::default(),
            depth_attachment_format: vk::Format::default(),
        }
    }

    pub fn set_shaders(
        mut self,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
    ) -> Self {
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let vertex_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(shader_entry_name);

        let fragment_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(shader_entry_name);
        self.shader_stages.push(vertex_stage);
        self.shader_stages.push(fragment_stage);
        self
    }

    pub fn set_input_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.input_assembly = self
            .input_assembly
            .topology(topology)
            .primitive_restart_enable(false);
        // we are not going to use primitive restart on the entire tutorial so leave
        // it on false
        self
    }

    pub fn set_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.rasterizer = self.rasterizer.polygon_mode(mode).line_width(1.);
        self
    }

    pub fn set_cull_mode(
        mut self,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
    ) -> Self {
        self.rasterizer = self.rasterizer.cull_mode(cull_mode).front_face(front_face);
        self
    }

    pub fn set_multisampling_none(mut self) -> Self {
        self.multisampling = self
            .multisampling
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.)
            .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);
        self
    }

    pub fn disable_blending(mut self) -> Self {
        self.color_blend_attachment = self
            .color_blend_attachment
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        self
    }

    pub fn set_color_attachment_format(mut self, format: vk::Format) -> Self {
        self.color_attachment_format = format;
        self
    }

    pub fn set_depth_attachment_format(mut self, format: vk::Format) -> Self {
        self.depth_attachment_format = format;
        self
    }

    pub fn disable_depth_test(mut self) -> Self {
        self.depth_stencil = self
            .depth_stencil
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::NEVER)
            .stencil_test_enable(false)
            .front(vk::StencilOpState::default())
            .back(vk::StencilOpState::default())
            .min_depth_bounds(0.)
            .max_depth_bounds(1.);
        self
    }

    pub fn build_pipeline(self, device: Arc<LogicalDevice>) -> Result<vk::Pipeline, vk::Result> {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let attachments = [self.color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&state);

        let color_attachment_formats = [self.color_attachment_format];
        let mut render_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(self.depth_attachment_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut render_info)
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipeline_layout.handle)
            .dynamic_state(&dynamic_state);

        let create_infos = [pipeline_info];
        let result = unsafe {
            device
                .handle
                .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
        };

        if let Ok(new_pipelines) = result {
            Ok(new_pipelines[0])
        } else {
            let (_, error) = result.err().unwrap();
            Err(error)
        }
    }
}