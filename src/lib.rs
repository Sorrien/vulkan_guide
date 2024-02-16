use std::{
    mem::{size_of, ManuallyDrop},
    sync::{Arc, Mutex},
    time::Instant,
};

use ash::vk;
use ash_bootstrap::LogicalDevice;
use base_vulkan::{BaseVulkanState, FrameData};
use buffers::{copy_to_staging_buffer, GPUDrawPushConstants, GPUMeshBuffers, MeshAsset, Vertex};
use descriptors::{Descriptor, DescriptorAllocator};
use gpu_allocator::{vulkan::*, MemoryLocation};
use pipelines::Pipeline;
use swapchain::MySwapchain;
use vk_imgui::init_imgui;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::pipelines::PipelineLayout;

pub mod ash_bootstrap;
pub mod base_vulkan;
pub mod buffers;
pub mod debug;
pub mod descriptors;
pub mod loader;
pub mod pipelines;
pub mod swapchain;
pub mod vk_imgui;

const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine {
    pub draw_extent: vk::Extent2D,
    pub render_scale: f32,
    pub resize_requested: bool,
    current_background_effect: usize,
    frame_number: usize,
    meshes: Vec<MeshAsset>,
    mesh_pipeline: Pipeline,
    mesh_pipeline_layout: Arc<PipelineLayout>,
    immediate_command: base_vulkan::ImmediateCommand,
    background_effects: Vec<ComputeEffect>,
    background_effect_pipeline_layout: Arc<PipelineLayout>,
    draw_image_descriptor: Descriptor,
    global_descriptor_allocator: descriptors::DescriptorAllocator,
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
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

        let depth_image_format = vk::Format::D32_SFLOAT;
        let (depth_image, depth_image_allocation) = base.create_image(
            draw_image_extent,
            depth_image_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            1,
            vk::SampleCountFlags::TYPE_1,
        );

        let depth_image_view = base.create_image_view(
            depth_image,
            depth_image_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        );

        let depth_image_allocated = AllocatedImage {
            image: depth_image,
            image_view: depth_image_view,
            extent: draw_image_extent.into(),
            format: depth_image_format,
            allocation: ManuallyDrop::new(depth_image_allocation),
            device: base.device.clone(),
            allocator: base.allocator.clone(),
        };

        let mut global_descriptor_allocator = DescriptorAllocator::new(base.device.clone());

        let draw_image_descriptor =
            base.init_descriptors(&mut global_descriptor_allocator, &draw_image_allocated);

        let (background_effects, background_effect_pipeline_layout) =
            base.init_pipelines(draw_image_descriptor.layout);

        let (mesh_pipeline, mesh_pipeline_layout) =
            base.init_mesh_pipeline(draw_image_allocated.format, depth_image_allocated.format);

        let immediate_command = base.init_immediate_command();
        let meshes = vec![];

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
        }
    }

    pub fn init_default_data(&mut self) {
        if let Some(mut meshes) = loader::load_gltf_meshes(self, "assets/basicmesh.glb") {
            self.meshes.append(&mut meshes);
        }
        /*
        let vertices = vec![
            Vertex::new(glam::vec3(-0.5, -0.5, 0.), glam::vec4(1., 0., 0., 1.)),
            Vertex::new(glam::vec3(0.5, -0.5, 0.), glam::vec4(0.0, 1.0, 0.0, 1.)),
            Vertex::new(glam::vec3(0.5, 0.5, 0.), glam::vec4(0., 0., 1., 1.)),
            Vertex::new(glam::vec3(-0.5, 0.5, 0.), glam::vec4(1.0, 1.0, 1.0, 1.)),
        ];
        let indices = vec![0, 1, 2, 2, 3, 0];
        let rectangle = self.upload_mesh(indices, vertices);
        self.meshes.push(rectangle);
         */
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

    pub fn get_current_frame(&self) -> Arc<FrameData> {
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
                _ => (),
            }
        })
    }

    pub fn draw(
        &mut self,
        draw_data: &imgui::DrawData,
        imgui_renderer: &mut imgui_rs_vulkan_renderer::Renderer,
    ) {
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

        self.draw_geometry(cmd);

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
            self.draw_extent,
            swapchain.extent,
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
        let dynamic_offsets = [];
        unsafe {
            self.base.device.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cur_effect.pipeline.pipeline_layout.handle,
                0,
                &descriptor_sets,
                &dynamic_offsets,
            )
        };

        let pc = &cur_effect.data;

        unsafe {
            let push = any_as_u8_slice(pc);
            self.base.device.handle.cmd_push_constants(
                cmd,
                cur_effect.pipeline.pipeline_layout.handle,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            )
        };

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

    fn draw_geometry(&self, cmd: vk::CommandBuffer) {
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

        unsafe {
            self.base.device.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline.pipeline,
            )
        };

        //for mesh in &self.meshes {
        let mesh = &self.meshes[2];

        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0., 0., -5.),
            glam::Vec3::new(0., 0., 0.),
            glam::Vec3::Y,
        );
        let mut proj = glam::Mat4::perspective_rh_gl(
            70.,
            self.draw_extent.width as f32 / self.draw_extent.height as f32,
            0.1,
            10000.,
        );
        proj.y_axis *= -1.;

        let draw_push_constants =
            GPUDrawPushConstants::new(proj * view, mesh.mesh_buffers.vertex_buffer_address);

        let pc = [draw_push_constants];
        unsafe {
            let push = any_as_u8_slice(&pc);
            self.base.device.handle.cmd_push_constants(
                cmd,
                self.mesh_pipeline_layout.handle,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push,
            )
        };

        unsafe {
            self.base.device.handle.cmd_bind_index_buffer(
                cmd,
                mesh.mesh_buffers.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            )
        };

        unsafe {
            self.base.device.handle.cmd_draw_indexed(
                cmd,
                mesh.surfaces[0].count as u32,
                1,
                mesh.surfaces[0].start_index as u32,
                0,
                0,
            )
        };
        //}

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
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for device idle!");
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
