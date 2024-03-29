use std::{
    collections::HashMap,
    mem::{size_of, ManuallyDrop},
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

use ash::vk;
use ash_bootstrap::LogicalDevice;
use base_vulkan::{BaseVulkanState, FrameData};
use buffers::{
    copy_buffer_to_image, copy_to_staging_buffer, write_to_cpu_buffer, AllocatedBuffer,
    GPUDrawPushConstants, GPUMeshBuffers, MeshAsset, Vertex,
};
use camera::Camera;
use descriptors::{
    Descriptor, DescriptorAllocator, DescriptorAllocatorGrowable, DescriptorLayout,
    DescriptorLayoutBuilder, DescriptorWriter,
};
use gpu_allocator::{vulkan::*, MemoryLocation};
use hecs::{Entity, World};
use image::RgbaImage;
use pipelines::{Pipeline, PipelineBuilder};
use swapchain::MySwapchain;
use vk_imgui::init_imgui;
use winit::{
    event::{DeviceEvent, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use crate::pipelines::PipelineLayout;

pub mod ash_bootstrap;
pub mod base_vulkan;
pub mod buffers;
pub mod camera;
pub mod debug;
pub mod descriptors;
pub mod loader;
pub mod pipelines;
pub mod swapchain;
pub mod vk_imgui;

const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine {
    camera: Camera,
    loaded_nodes: HashMap<String, Entity>,
    world: World,
    main_draw_context: DrawContext,
    default_data: Option<Arc<MaterialInstance>>,
    metal_rough_material: Option<GLTFMetallicRoughness>,
    default_sampler_linear: Sampler,
    default_sampler_nearest: Sampler,
    white_image: Option<AllocatedImage>,
    grey_image: Option<AllocatedImage>,
    black_image: Option<AllocatedImage>,
    error_checkerboard_image: Option<AllocatedImage>,
    scene_data: GPUSceneData,
    pub draw_extent: vk::Extent2D,
    pub render_scale: f32,
    pub resize_requested: bool,
    current_background_effect: usize,
    frame_number: usize,
    gpu_scene_data_descriptor_layout: DescriptorLayout,
    meshes: Vec<Arc<MeshAsset>>,
    mesh_pipeline: Pipeline,
    mesh_pipeline_layout: Arc<PipelineLayout>,
    single_image_descriptor_layout: DescriptorLayout,
    immediate_command: base_vulkan::ImmediateCommand,
    background_effects: Vec<ComputeEffect>,
    background_effect_pipeline_layout: Arc<PipelineLayout>,
    draw_image_descriptor: Descriptor,
    global_descriptor_allocator: descriptors::DescriptorAllocatorGrowable,
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
    frames: Vec<Arc<Mutex<FrameData>>>,
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
        let draw_image_allocated = base.create_allocated_image(
            draw_image_extent.into(),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        );

        let depth_image_format = vk::Format::D32_SFLOAT;

        let depth_image_allocated = base.create_allocated_image(
            draw_image_extent.into(),
            depth_image_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        );

        let mut global_descriptor_allocator = DescriptorAllocatorGrowable::new(base.device.clone());

        let draw_image_descriptor =
            base.init_descriptors(&mut global_descriptor_allocator, &draw_image_allocated);

        let gpu_scene_data_descriptor_layout = base.init_gpu_scene_descriptor_layout();

        let single_image_descriptor_layout = base.init_single_image_layout();

        let (background_effects, background_effect_pipeline_layout) =
            base.init_pipelines(draw_image_descriptor.layout.handle);

        let (mesh_pipeline, mesh_pipeline_layout) = base.init_mesh_pipeline(
            draw_image_allocated.format,
            depth_image_allocated.format,
            &single_image_descriptor_layout,
        );

        let immediate_command = base.init_immediate_command();
        let meshes = vec![];

        let default_sampler_linear = Sampler::new(vk::Sampler::null(), base.device.clone());
        let default_sampler_nearest = Sampler::new(vk::Sampler::null(), base.device.clone());

        Self {
            base,
            frames: vec![],
            frame_number: 0,
            draw_image: draw_image_allocated,
            swapchain,
            global_descriptor_allocator,
            draw_image_descriptor,
            background_effects,
            background_effect_pipeline_layout,
            immediate_command,
            current_background_effect: 0,
            mesh_pipeline,
            mesh_pipeline_layout,
            meshes,
            depth_image: depth_image_allocated,
            resize_requested: false,
            draw_extent: draw_image_extent,
            render_scale: 1.,
            gpu_scene_data_descriptor_layout,
            scene_data: GPUSceneData::default(),
            white_image: None,
            grey_image: None,
            black_image: None,
            error_checkerboard_image: None,
            default_sampler_linear,
            default_sampler_nearest,
            single_image_descriptor_layout,
            default_data: None,
            metal_rough_material: None,
            main_draw_context: DrawContext::new(),
            world: World::new(),
            loaded_nodes: HashMap::new(),
            camera: Camera::new(glam::Vec3::new(0., 0., 5.), 0., 0.),
        }
    }

    pub fn init_pipelines(&mut self) {
        let material_constants_buffer = self.base.create_buffer(
            "gltf material constants",
            size_of::<MaterialConstants>(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        );

        self.metal_rough_material =
            Some(GLTFMetallicRoughness::new(&self, material_constants_buffer));
    }

    pub fn init_default_data(&mut self) {
        let white: [u8; 4] = [255, 255, 255, 255];

        /*         let white_image_handle =
        self.allocated_image_arena
            .insert(self.create_allocated_texture_image(
                &[white],
                vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::SAMPLED,
                MemoryLocation::GpuOnly,
                1,
                vk::SampleCountFlags::TYPE_1,
            )); */

        self.white_image = Some(self.create_allocated_texture_image(
            &[white],
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        ));

        let grey: [u8; 4] = [128, 128, 128, 255];
        self.grey_image = Some(self.create_allocated_texture_image(
            &[grey],
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        ));

        let black: [u8; 4] = [0, 0, 0, 255];
        self.black_image = Some(self.create_allocated_texture_image(
            &[black],
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        ));

        let mut my_image = RgbaImage::new(16, 16);
        let magenta = [255, 0, 255, 255];
        for (x, y, pixel) in my_image.enumerate_pixels_mut() {
            pixel.0 = if (x + y) % 2 == 0 { magenta } else { black }
        }

        self.error_checkerboard_image = Some(self.create_allocated_texture_image(
            &my_image.into_raw(),
            vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        ));

        let nearest_sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        self.default_sampler_nearest.handle = unsafe {
            self.base
                .device
                .handle
                .create_sampler(&nearest_sampler_create_info, None)
        }
        .expect("failed to create nearest default sampler!");

        let linear_sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        self.default_sampler_linear.handle = unsafe {
            self.base
                .device
                .handle
                .create_sampler(&linear_sampler_create_info, None)
        }
        .expect("failed to create nearest linear sampler!");

        if let Some(ref mut metal_rough_material) = &mut self.metal_rough_material {
            let material_constants = MaterialConstants {
                colorFactors: glam::Vec4::new(1., 1., 1., 1.),
                metal_rough_factors: glam::Vec4::new(1., 0.5, 0., 0.),
                extra: [glam::Vec4::ZERO; 14],
            };

            write_to_cpu_buffer(
                &material_constants,
                &mut metal_rough_material.material_constants_buffer,
            );

            let white_image = self.white_image.as_ref().unwrap();
            let material_resources = MaterialResources {
                color_image: white_image,
                color_sampler: &self.default_sampler_linear,
                metal_rough_image: white_image,
                metal_rough_sampler: &self.default_sampler_linear,
                data_buffer: metal_rough_material.material_constants_buffer.buffer,
                data_buffer_offset: 0,
            };

            let default_material = Arc::new(metal_rough_material.write_material(
                self.base.device.clone(),
                MaterialPass::MainColor,
                &material_resources,
                &mut self.global_descriptor_allocator,
            ));

            self.default_data = Some(default_material.clone());

            if let Some(meshes) =
                loader::load_gltf_meshes(self, "assets/basicmesh.glb", default_material.clone())
            {
                self.meshes = meshes
                    .into_iter()
                    .map(|mesh| Arc::new(mesh))
                    .collect::<Vec<_>>();
            }

            for mesh in &self.meshes {
                let mesh_entity = self.world.spawn((
                    TransformComponent {
                        transform: glam::Mat4::IDENTITY,
                    },
                    MeshHandleComponent { mesh: mesh.clone() },
                ));

                self.loaded_nodes.insert(mesh.name.clone(), mesh_entity);
            }
        } else {
            panic!("Missing metal rough material!");
        }
    }

    pub fn update_scene(&mut self) {
        //evaluate_relative_transforms(&mut self.world);

        self.camera.update();

        self.main_draw_context.opaque_surfaces.clear();

        let suzanne_entity = self.loaded_nodes["Suzanne"];

        let model = glam::Mat4::IDENTITY * glam::Mat4::from_axis_angle(glam::Vec3::Y, 0.);

        draw_entity(
            &mut self.world,
            suzanne_entity,
            &model,
            &mut self.main_draw_context,
        );

        let cube_entity = self.loaded_nodes["Cube"];

        for x in -3..3 {
            draw_entity(
                &mut self.world,
                cube_entity,
                &(glam::Mat4::from_scale(glam::Vec3::new(0.2, 0.2, 0.2))
                    * glam::Mat4::from_translation(glam::Vec3::new(x as f32, -5., 0.))),
                &mut self.main_draw_context,
            );
        }

        self.scene_data.view = self.camera.get_view_matrix();

        self.scene_data.proj = glam::Mat4::perspective_rh(
            70.,
            self.draw_extent.width as f32 / self.draw_extent.height as f32,
            10000.,
            0.1,
        );
        self.scene_data.proj.y_axis *= -1.;

        self.scene_data.viewproj = self.scene_data.proj * self.scene_data.view;

        self.scene_data.ambientColor = glam::Vec4::ONE;
        self.scene_data.sunlightColor = glam::Vec4::ONE;
        self.scene_data.sunlightDirection = glam::Vec4::new(0., 1., 0.5, 1.);
    }

    pub fn init_imgui(
        &self,
    ) -> (
        imgui::Context,
        imgui_winit_support::WinitPlatform,
        imgui_rs_vulkan_renderer::Renderer,
    ) {
        init_imgui(
            &self.base.window,
            &self.base.allocator,
            &self.base.device,
            self.base.graphics_queue,
            self.immediate_command.command_pool,
            self.swapchain.format,
        )
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

    pub fn get_current_frame(&self) -> Arc<Mutex<FrameData>> {
        self.frames[self.frame_number % FRAME_OVERLAP].clone()
    }

    pub fn run(
        &mut self,
        event_loop: EventLoop<()>,
        imgui: imgui::Context,
        platform: imgui_winit_support::WinitPlatform,
        imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
    ) -> Result<(), winit::error::EventLoopError> {
        self.main_loop(event_loop, imgui, platform, imgui_renderer)
    }

    fn main_loop(
        &mut self,
        event_loop: EventLoop<()>,
        mut imgui: imgui::Context,
        mut platform: imgui_winit_support::WinitPlatform,
        mut imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
    ) -> Result<(), winit::error::EventLoopError> {
        let mut last_frame = Instant::now();

        event_loop.run(move |event, elwt| {
            platform.handle_event(imgui.io_mut(), &self.base.window, &event);

            match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();

                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    self.resize_requested = true;
                }
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

                    if self.resize_requested {
                        if size.width > 0 && size.height > 0 {
                            //println!("resize requested!");
                            self.resize_swapchain();
                        } else {
                            return;
                        }
                    }
                    platform
                        .prepare_frame(imgui.io_mut(), &self.base.window)
                        .expect("failed to prepare frame!");
                    let ui = imgui.frame();

                    ui.window("background")
                        .size([500.0, 200.0], imgui::Condition::FirstUseEver)
                        .build(|| {
                            ui.slider("Render Scale", 0.3, 1., &mut self.render_scale);
                            let len = self.background_effects.len();
                            let selected =
                                &mut self.background_effects[self.current_background_effect];
                            ui.text(format!("Selected effect: {}", selected.name));
                            ui.slider(
                                "Effect Index",
                                0,
                                len - 1,
                                &mut self.current_background_effect,
                            );
                            ui.input_float4("data1", &mut selected.data.data1).build();
                            ui.input_float4("data2", &mut selected.data.data2).build();
                            ui.input_float4("data3", &mut selected.data.data3).build();
                            ui.input_float4("data4", &mut selected.data.data4).build();
                        });

                    platform.prepare_render(ui, &self.base.window);
                    let draw_data = imgui.render();

                    //don't attempt to draw a frame in window size is 0
                    if size.height > 0 && size.width > 0 {
                        self.draw(draw_data, &mut imgui_renderer);
                    }
                }
                Event::WindowEvent {
                    window_id: _,
                    event: WindowEvent::Focused(is_focused),
                } => {
                    //stop_rendering = !is_focused;
                }
                Event::WindowEvent {
                    window_id: _,
                    event: WindowEvent::Resized(_new_size),
                } => {
                    //self.framebuffer_resized = true;
                }
                Event::DeviceEvent {
                    device_id,
                    event:
                        DeviceEvent::MouseMotion {
                            delta: (delta_x, delta_y),
                        },
                } => {
                    self.camera
                        .process_mouse_input_event(delta_x as f32, delta_y as f32);
                }
                Event::WindowEvent {
                    window_id,
                    event:
                        WindowEvent::KeyboardInput {
                            device_id,
                            event,
                            is_synthetic,
                        },
                } => {
                    self.camera.process_keyboard_input_event(event);
                }
                _ => (),
            }
        })
    }

    pub fn draw(
        &mut self,
        draw_data: &imgui::DrawData,
        imgui_renderer: &mut imgui_rs_vulkan_renderer::Renderer,
    ) {
        self.update_scene();

        let current_frame = self.get_current_frame();
        let mut current_frame = current_frame.lock().unwrap();
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

        current_frame.frame_descriptors.destroy_pools();
        current_frame.frame_descriptors.clear_pools();

        let swapchain = &self.swapchain;

        self.draw_extent.height = (swapchain.extent.height.min(self.draw_image.extent.height)
            as f32
            * self.render_scale) as u32;
        self.draw_extent.width = (swapchain.extent.width.min(self.draw_image.extent.width) as f32
            * self.render_scale) as u32;

        //acquire next swapchain image
        let (swapchain_image_index, _) = unsafe {
            let result = swapchain.swapchain_loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                current_frame.swapchain_semaphore,
                vk::Fence::null(),
            );

            match result {
                Ok((image_index, was_next_image_acquired)) => {
                    (image_index, was_next_image_acquired)
                }
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        self.resize_requested = true;
                        return;
                    }
                    _ => panic!("failed to acquire next swapchain image!"),
                },
            }
        };

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

        let swapchain_image = swapchain.swapchain_images[swapchain_image_index as usize];
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
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        self.base.transition_image_layout(
            cmd,
            self.depth_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        self.draw_geometry(cmd, &mut current_frame);

        self.base.transition_image_layout(
            cmd,
            self.draw_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
            self.draw_extent.into(),
            swapchain.extent.into(),
        );

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        self.draw_imgui(
            cmd,
            draw_data,
            imgui_renderer,
            swapchain.swapchain_image_views[swapchain_image_index as usize],
        );

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

        let swapchains = [swapchain.swapchain];
        let render_semaphores = [current_frame.render_semaphore];
        let swapchain_image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&render_semaphores)
            .image_indices(&swapchain_image_indices);

        let present_result = unsafe {
            swapchain
                .swapchain_loader
                .queue_present(self.base.graphics_queue, &present_info)
        };
        match present_result {
            Ok(_) => (),
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                    self.resize_requested = true;
                }
                _ => panic!("failed to present swap chain image!"),
            },
        }

        self.frame_number += 1;
    }

    fn resize_swapchain(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for idle when recreating swapchain!");

        let window_size = self.base.window.inner_size();
        let window_height = window_size.height;
        let window_width = window_size.width;

        self.swapchain.destroy();
        self.swapchain = self.base.create_swapchain(window_width, window_height);
        self.resize_requested = false;
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
        let cur_effect = &self.background_effects[self.current_background_effect];

        unsafe {
            self.base.device.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cur_effect.pipeline.pipeline,
            )
        }
        let descriptor_sets = [self.draw_image_descriptor.set];
        unsafe {
            self.base.device.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cur_effect.pipeline.pipeline_layout.handle,
                0,
                &descriptor_sets,
                &[],
            )
        };

        cmd_push_constants(
            self.base.device.clone(),
            cmd,
            cur_effect.pipeline.pipeline_layout.clone(),
            cur_effect.data,
            vk::ShaderStageFlags::COMPUTE,
            0,
        );

        unsafe {
            self.base.device.handle.cmd_dispatch(
                cmd,
                (self.draw_extent.width as f32 / 16.).ceil() as u32,
                (self.draw_extent.height as f32 / 16.).ceil() as u32,
                1,
            );
        }
    }

    pub fn draw_imgui(
        &self,
        cmd: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        imgui_renderer: &mut imgui_rs_vulkan_renderer::Renderer,
        target_image_view: vk::ImageView,
    ) {
        let color_attachment =
            Self::attachment_info(target_image_view, None, vk::ImageLayout::GENERAL);
        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.swapchain.extent.into())
            .color_attachments(&color_attachments)
            .flags(vk::RenderingFlags::CONTENTS_INLINE_EXT)
            .layer_count(1);

        unsafe {
            self.base
                .device
                .handle
                .cmd_begin_rendering(cmd, &rendering_info)
        };

        imgui_renderer
            .cmd_draw(cmd, draw_data)
            .expect("failed to draw imgui data!");

        unsafe { self.base.device.handle.cmd_end_rendering(cmd) };
    }

    fn attachment_info(
        view: vk::ImageView,
        clear: Option<vk::ClearValue>,
        layout: vk::ImageLayout,
    ) -> vk::RenderingAttachmentInfo<'static> {
        let load_op = if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        };
        let mut result = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(layout)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE);

        if let Some(clear) = clear {
            result = result.clear_value(clear);
        }

        result
    }

    pub fn immediate_submit<F: FnOnce(vk::CommandBuffer)>(&self, f: F) {
        let fences = [self.immediate_command.fence];
        let cmd = self.immediate_command.command_buffer;
        let device = &self.base.device.handle;

        unsafe { device.reset_fences(&fences) }.expect("failed to reset immediate submit fence!");
        unsafe { device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .expect("failed to reset imm submit cmd buffer!");

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("failed to end imm submit cmd buffer!");

        f(cmd);

        unsafe { device.end_command_buffer(cmd) }.expect("failed to end imm submit cmd buffer!");

        let cmd_info = Self::command_buffer_submit_info(cmd);
        let cmd_infos = [cmd_info];
        let submit = Self::submit_info(&cmd_infos, &[], &[]);

        //we may want to find a different queue than graphics for this if possible
        let submits = [submit];
        unsafe {
            device.queue_submit2(
                self.base.graphics_queue,
                &submits,
                self.immediate_command.fence,
            )
        }
        .expect("failed to submit imm cmd!");

        let fences = [self.immediate_command.fence];
        unsafe { device.wait_for_fences(&fences, true, u64::MAX) }
            .expect("failed to wait for imm submit fence!");
    }

    fn draw_geometry(&self, cmd: vk::CommandBuffer, frame: &mut MutexGuard<'_, FrameData>) {
        write_to_cpu_buffer(&self.scene_data, &mut frame.gpu_scene_data_buffer);

        let global_descriptor = frame
            .frame_descriptors
            .allocate(self.gpu_scene_data_descriptor_layout.handle);
        {
            let mut descriptor_writer = DescriptorWriter::new();
            descriptor_writer.write_buffer(
                0,
                frame.gpu_scene_data_buffer.buffer,
                size_of::<GPUSceneData>() as u64,
                0,
                descriptors::BufferDescriptorType::UniformBuffer,
            );
            descriptor_writer.update_set(self.base.device.clone(), global_descriptor);
        }

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.draw_image.image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut depth_clear_value = vk::ClearValue::default();
        depth_clear_value.depth_stencil.depth = 0.;
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_image.image_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value);

        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.draw_extent.into())
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment)
            .layer_count(1);

        unsafe {
            self.base
                .device
                .handle
                .cmd_begin_rendering(cmd, &rendering_info)
        };

        let viewport = vk::Viewport::default()
            .x(0.)
            .y(0.)
            .width(self.draw_extent.width as f32)
            .height(self.draw_extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);
        let viewports = [viewport];
        unsafe { self.base.device.handle.cmd_set_viewport(cmd, 0, &viewports) };

        let scissor = vk::Rect2D::default().extent(self.draw_extent);
        let scissors = [scissor];
        unsafe { self.base.device.handle.cmd_set_scissor(cmd, 0, &scissors) };

        for draw in &self.main_draw_context.opaque_surfaces {
            unsafe {
                self.base.device.handle.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.material.pipeline.pipeline,
                );
            };
            let draw_pipeline_desc_sets_1 = [global_descriptor];
            let draw_pipeline_desc_sets_2 = [draw.material.material_set];
            unsafe {
                self.base.device.handle.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.material.pipeline.pipeline_layout.handle,
                    0,
                    &draw_pipeline_desc_sets_1,
                    &[],
                );
                self.base.device.handle.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.material.pipeline.pipeline_layout.handle,
                    1,
                    &draw_pipeline_desc_sets_2,
                    &[],
                );
            };

            let draw_push_constants =
                GPUDrawPushConstants::new(draw.transform, draw.vertex_buffer_address);

            cmd_push_constants(
                self.base.device.clone(),
                cmd,
                draw.material.pipeline.pipeline_layout.clone(),
                draw_push_constants,
                vk::ShaderStageFlags::VERTEX,
                0,
            );

            unsafe {
                self.base.device.handle.cmd_bind_index_buffer(
                    cmd,
                    draw.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                )
            };

            unsafe {
                self.base.device.handle.cmd_draw_indexed(
                    cmd,
                    draw.index_count,
                    1,
                    draw.first_index,
                    0,
                    0,
                )
            };
        }

        unsafe { self.base.device.handle.cmd_end_rendering(cmd) };
    }

    fn upload_mesh(&self, indices: Vec<u32>, vertices: Vec<Vertex>) -> GPUMeshBuffers {
        let vertex_buffer_size = vertices.len() * size_of::<Vertex>();
        let index_buffer_size = indices.len() * size_of::<u32>();

        let vertex_buffer = self.base.create_buffer(
            "vertex buffer",
            vertex_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        );

        let info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let vertex_buffer_address: vk::DeviceAddress =
            unsafe { self.base.device.handle.get_buffer_device_address(&info) };

        let index_buffer = self.base.create_buffer(
            "index buffer",
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );

        let new_surface = GPUMeshBuffers {
            index_buffer,
            vertex_buffer,
            vertex_buffer_address,
        };

        let vertex_staging_buffer = self.base.create_buffer(
            "vertex_staging_buffer",
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        copy_to_staging_buffer(&vertex_staging_buffer, vertex_buffer_size as u64, &vertices);

        let index_staging_buffer = self.base.create_buffer(
            "index_staging_buffer_staging_buffer",
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        copy_to_staging_buffer(&index_staging_buffer, index_buffer_size as u64, &indices);

        self.immediate_submit(|cmd| {
            let vertex_copy = vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(0)
                .size(vertex_buffer_size as u64);
            let vertex_regions = [vertex_copy];
            unsafe {
                self.base.device.handle.cmd_copy_buffer(
                    cmd,
                    vertex_staging_buffer.buffer,
                    new_surface.vertex_buffer.buffer,
                    &vertex_regions,
                )
            };

            let index_copy = vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(0)
                .size(index_buffer_size as u64);
            let index_regions = [index_copy];
            unsafe {
                self.base.device.handle.cmd_copy_buffer(
                    cmd,
                    index_staging_buffer.buffer,
                    new_surface.index_buffer.buffer,
                    &index_regions,
                )
            };
        });

        new_surface
    }

    pub fn create_allocated_texture_image<T>(
        &mut self,
        data: &[T],
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
    ) -> AllocatedImage
    where
        T: Copy,
    {
        let data_size = size_of::<T>() * data.len(); //extent.depth * extent.width * extent.height;
        let upload_buffer = self.base.create_buffer(
            "texture upload buffer",
            data_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        copy_to_staging_buffer(&upload_buffer, data_size as u64, &data);

        let allocated_image = self.base.create_allocated_image(
            extent,
            format,
            tiling,
            usage | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            memory_location,
            mip_levels,
            num_samples,
        );

        self.immediate_submit(|cmd| {
            self.base.transition_image_layout(
                cmd,
                allocated_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            copy_buffer_to_image(
                cmd,
                self.base.device.clone(),
                upload_buffer.buffer,
                allocated_image.image,
                extent,
            );

            self.base.transition_image_layout(
                cmd,
                allocated_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        allocated_image
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for device idle!");
    }
}

pub struct Sampler {
    device: Arc<LogicalDevice>,
    pub handle: vk::Sampler,
}

impl Sampler {
    fn new(handle: vk::Sampler, device: Arc<LogicalDevice>) -> Self {
        Self { handle, device }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None) };
    }
}

pub struct AllocatedImage {
    device: Arc<LogicalDevice>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: ManuallyDrop<Allocation>,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
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
    src_size: vk::Extent3D,
    dst_size: vk::Extent3D,
) {
    let blit_region = vk::ImageBlit2::default()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: src_size.width as i32,
                y: src_size.height as i32,
                z: src_size.depth as i32,
            },
        ])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: dst_size.width as i32,
                y: dst_size.height as i32,
                z: dst_size.depth as i32,
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

#[derive(Copy, Clone)]
pub struct ComputePushConstants {
    pub data1: glam::Vec4,
    pub data2: glam::Vec4,
    pub data3: glam::Vec4,
    pub data4: glam::Vec4,
}

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}

pub struct ComputeEffect {
    pub name: String,
    pub pipeline: Pipeline,
    pub data: ComputePushConstants,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GPUSceneData {
    view: glam::Mat4,
    proj: glam::Mat4,
    viewproj: glam::Mat4,
    ambientColor: glam::Vec4,
    sunlightDirection: glam::Vec4,
    sunlightColor: glam::Vec4,
}

pub struct RenderObject {
    index_count: u32,
    first_index: u32,
    index_buffer: vk::Buffer,

    material: Arc<MaterialInstance>,
    transform: glam::Mat4,
    vertex_buffer_address: vk::DeviceAddress,
}

pub struct DrawContext {
    opaque_surfaces: Vec<RenderObject>,
}

impl DrawContext {
    pub fn new() -> Self {
        Self {
            opaque_surfaces: Vec::new(),
        }
    }
}

pub struct MaterialInstance {
    pipeline: Arc<Pipeline>,
    material_set: vk::DescriptorSet,
    pass_type: MaterialPass,
}

#[derive(PartialEq, Clone, Copy)]
pub enum MaterialPass {
    MainColor = 1,
    Transparent = 2,
    Other = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct MaterialConstants {
    colorFactors: glam::Vec4,
    metal_rough_factors: glam::Vec4,
    //padding, we need it anyway for uniform buffers
    extra: [glam::Vec4; 14],
}

pub struct MaterialResources<'a> {
    color_image: &'a AllocatedImage,
    color_sampler: &'a Sampler,
    metal_rough_image: &'a AllocatedImage,
    metal_rough_sampler: &'a Sampler,
    data_buffer: vk::Buffer,
    data_buffer_offset: u64,
}

pub struct GLTFMetallicRoughness {
    opaque_pipeline: Arc<Pipeline>,
    transparent_pipeline: Arc<Pipeline>,
    material_layout: DescriptorLayout,
    material_constants_buffer: AllocatedBuffer,
}

impl GLTFMetallicRoughness {
    pub fn new(engine: &VulkanEngine, material_constants_buffer: AllocatedBuffer) -> Self {
        let material_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                engine.base.device.clone(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )
            .expect("failed to create material set layout");

        let (opaque_pipeline, transparent_pipeline) =
            Self::build_pipelines(engine, material_layout);

        Self {
            opaque_pipeline: Arc::new(opaque_pipeline),
            transparent_pipeline: Arc::new(transparent_pipeline),
            material_layout: DescriptorLayout::new(engine.base.device.clone(), material_layout),
            material_constants_buffer,
        }
    }

    pub fn build_pipelines(
        engine: &VulkanEngine,
        material_layout: vk::DescriptorSetLayout,
    ) -> (Pipeline, Pipeline) {
        let mesh_frag_shader = engine
            .base
            .create_shader_module("shaders/mesh.frag.spv")
            .expect("failed to load shader module!");
        let mesh_vert_shader = engine
            .base
            .create_shader_module("shaders/mesh.vert.spv")
            .expect("failed to load shader module!");

        let matrix_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let layouts = [
            engine.gpu_scene_data_descriptor_layout.handle,
            material_layout,
        ];

        let push_constant_ranges = [matrix_range];
        let mesh_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&layouts);

        let new_layout = PipelineLayout::new(engine.base.device.clone(), mesh_layout_info)
            .expect("failed to create pipeline layout!");

        let pipeline_builder = PipelineBuilder::new(new_layout)
            .set_shaders(mesh_vert_shader, mesh_frag_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .set_multisampling_none()
            .set_color_attachment_format(engine.draw_image.format)
            .set_depth_attachment_format(engine.depth_image.format);

        let pipeline_builder_transparent = pipeline_builder
            .clone()
            .enable_blending_additive()
            .enable_depth_test(false, vk::CompareOp::GREATER_OR_EQUAL);

        let pipeline_builder_opaque = pipeline_builder
            .disable_blending()
            .enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL);

        let opaque_pipeline = pipeline_builder_opaque
            .build_pipeline(engine.base.device.clone())
            .expect("failed to build opaque pipeline!");

        let transparent_pipeline = pipeline_builder_transparent
            .build_pipeline(engine.base.device.clone())
            .expect("failed to build transparent pipeline!");

        unsafe {
            engine
                .base
                .device
                .handle
                .destroy_shader_module(mesh_vert_shader, None);
            engine
                .base
                .device
                .handle
                .destroy_shader_module(mesh_frag_shader, None);
        }

        (opaque_pipeline, transparent_pipeline)
    }

    pub fn clear_resources() {}

    pub fn write_material(
        &self,
        device: Arc<LogicalDevice>,
        pass: MaterialPass,
        resources: &MaterialResources,
        descriptor_allocator: &mut DescriptorAllocatorGrowable,
    ) -> MaterialInstance {
        let pipeline = if pass == MaterialPass::Transparent {
            self.transparent_pipeline.clone()
        } else {
            self.opaque_pipeline.clone()
        };

        let material_set = descriptor_allocator.allocate(self.material_layout.handle);

        let mut desc_writer = DescriptorWriter::new();
        desc_writer.write_buffer(
            0,
            resources.data_buffer,
            size_of::<MaterialConstants>() as u64,
            resources.data_buffer_offset,
            descriptors::BufferDescriptorType::UniformBuffer,
        );
        desc_writer.write_image(
            1,
            resources.color_image.image_view,
            resources.color_sampler.handle,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        desc_writer.write_image(
            2,
            resources.metal_rough_image.image_view,
            resources.metal_rough_sampler.handle,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        desc_writer.update_set(device.clone(), material_set);

        let mat_data = MaterialInstance {
            pipeline,
            material_set,
            pass_type: pass,
        };

        mat_data
    }
}

fn evaluate_relative_transforms(world: &mut World) {
    let mut parents = world.query::<&ParentComponent>();
    let parents = parents.view();

    let mut roots = world
        .query::<&TransformComponent>()
        .without::<&ParentComponent>();
    let roots = roots.view();

    for (_entity, (parent, absolute)) in world
        .query::<(&ParentComponent, &mut TransformComponent)>()
        .iter()
    {
        let mut relative = parent.local_transform;
        let mut ancestor = parent.parent;
        while let Some(next) = parents.get(ancestor) {
            relative = next.local_transform * relative;
            ancestor = next.parent;
        }
        absolute.transform = roots.get(ancestor).unwrap().transform * relative;
    }
}

fn draw_all(world: &mut World, top_matrix: &glam::Mat4, context: &mut DrawContext) {
    for (_entity, (mesh_component, transform_component)) in world
        .query::<(&MeshHandleComponent, &TransformComponent)>()
        .iter()
    {
        let node_matrix = *top_matrix * transform_component.transform;
        context.opaque_surfaces.append(&mut mesh_to_render_object(
            node_matrix,
            &mesh_component.mesh,
        ));
    }
}

fn draw_entity(
    world: &mut World,
    draw_entity: Entity,
    top_matrix: &glam::Mat4,
    context: &mut DrawContext,
) {
    if let Some((_entity, (mesh_component, transform_component))) = world
        .query::<(&MeshHandleComponent, &TransformComponent)>()
        .iter()
        .find(|(entity, _)| *entity == draw_entity)
    {
        let node_matrix = *top_matrix * transform_component.transform;
        /*         for s in &mesh_component.mesh.surfaces {
            let def = RenderObject {
                index_count: s.count as u32,
                first_index: s.start_index as u32,
                index_buffer: mesh_component.mesh.mesh_buffers.index_buffer.buffer,
                material: s.material.clone(),
                transform: node_matrix,
                vertex_buffer_address: mesh_component.mesh.mesh_buffers.vertex_buffer_address,
            };
            context.opaque_surfaces.push(def);
        } */
        context.opaque_surfaces.append(&mut mesh_to_render_object(
            node_matrix,
            &mesh_component.mesh,
        ));
    }
}

fn mesh_to_render_object(
    node_matrix: glam::Mat4,
    mesh_asset: &Arc<MeshAsset>,
) -> Vec<RenderObject> {
    let mut surfaces = vec![];
    for s in &mesh_asset.surfaces {
        let def = RenderObject {
            index_count: s.count as u32,
            first_index: s.start_index as u32,
            index_buffer: mesh_asset.mesh_buffers.index_buffer.buffer,
            material: s.material.clone(),
            transform: node_matrix,
            vertex_buffer_address: mesh_asset.mesh_buffers.vertex_buffer_address,
        };
        surfaces.push(def);
    }
    surfaces
}

#[derive(Debug, Copy, Clone, Default)]
pub struct TransformComponent {
    transform: glam::Mat4,
}

pub struct ParentComponent {
    parent: Entity,
    local_transform: glam::Mat4,
}

pub struct MeshHandleComponent {
    mesh: Arc<MeshAsset>,
}

pub fn cmd_push_constants<T>(
    device: Arc<LogicalDevice>,
    cmd: vk::CommandBuffer,
    pipeline_layout: Arc<PipelineLayout>,
    push_constants: T,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
) {
    let pc = [push_constants];
    unsafe {
        let push = any_as_u8_slice(&pc);
        device
            .handle
            .cmd_push_constants(cmd, pipeline_layout.handle, stage_flags, offset, &push)
    };
}
