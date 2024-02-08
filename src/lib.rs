use std::{
    ffi::CStr,
    mem::{size_of, ManuallyDrop},
    sync::{Arc, Mutex},
    time::Instant,
};

use ash::vk;
use ash_bootstrap::LogicalDevice;
use base_vulkan::{BaseVulkanState, FrameData, Pipeline};
use descriptors::{Descriptor, DescriptorAllocator, PoolSizeRatio};
use gpu_allocator::vulkan::*;
use imgui_rs_vulkan_renderer::DynamicRendering;
use swapchain::MySwapchain;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub mod ash_bootstrap;
pub mod base_vulkan;
pub mod debug;
pub mod descriptors;
pub mod swapchain;

const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine {
    current_background_effect: usize,
    frame_number: usize,
    //imgui_context: ImguiContext,
    triangle_pipeline: Pipeline,
    triangle_pipeline_layout: Arc<base_vulkan::PipelineLayout>,
    immediate_command: base_vulkan::ImmediateCommand,
    background_effects: Vec<ComputeEffect>,
    background_effect_pipeline_layout: Arc<base_vulkan::PipelineLayout>,
    draw_image_descriptor: Descriptor,
    global_descriptor_allocator: descriptors::DescriptorAllocator,
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

        let mut global_descriptor_allocator = DescriptorAllocator::new(base.device.clone());

        let draw_image_descriptor =
            base.init_descriptors(&mut global_descriptor_allocator, &draw_image_allocated);

        let (background_effects, background_effect_pipeline_layout) =
            base.init_pipelines(draw_image_descriptor.layout);

        let (triangle_pipeline, triangle_pipeline_layout) =
            base.init_triangle_pipeline(draw_image_allocated.format);

        let immediate_command = base.init_immediate_command();

        /*         let imgui_context = ImguiContext::new(
            &base.window,
            &base.allocator,
            &base.device,
            base.graphics_queue,
            immediate_command.command_pool,
            swapchain.format,
        ); */

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
            triangle_pipeline,
            triangle_pipeline_layout, //imgui_context,
        }
    }

    /* pub fn init_imgui(&self) -> ImguiContext {
        ImguiContext::new(
            &self.base.window,
            &self.base.allocator,
            &self.base.device,
            self.base.graphics_queue,
            self.immediate_command.command_pool,
            self.swapchain.format,
        )
    } */

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
        let mut stop_rendering = false;

        let mut value = 0;
        let choices = ["test test this is 1", "test test this is 2"];

        event_loop.run(move |event, elwt| {
            platform.handle_event(imgui.io_mut(), &self.base.window, &event);

            match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();

                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
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
                    platform
                        .prepare_frame(imgui.io_mut(), &self.base.window)
                        .expect("failed to prepare frame!");
                    let ui = imgui.frame();

                    ui.window("background")
                        .size([500.0, 200.0], imgui::Condition::FirstUseEver)
                        .build(|| {
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
                    if size.height > 0 && size.width > 0 && !stop_rendering {
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
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
            self.draw_image.extent,
            self.swapchain.extent,
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
            self.swapchain.swapchain_image_views[swapchain_image_index as usize],
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
                (self.draw_image.extent.width as f32 / 16.).ceil() as u32,
                (self.draw_image.extent.height as f32 / 16.).ceil() as u32,
                1,
            );
        }
    }

    pub fn draw_imgui(
        &mut self,
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
        unsafe { self.base.device.handle.reset_fences(&fences) }
            .expect("failed to reset immediate submit fence!");
        unsafe {
            self.base.device.handle.reset_command_buffer(
                self.immediate_command.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .expect("failed to reset imm submit cmd buffer!");

        let cmd = self.immediate_command.command_buffer;
        let device = &self.base.device.handle;

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

        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.swapchain.extent.into())
            .color_attachments(&color_attachments)
            .layer_count(1);

        unsafe {
            self.base
                .device
                .handle
                .cmd_begin_rendering(cmd, &rendering_info)
        };

        unsafe {
            self.base.device.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.triangle_pipeline.pipeline,
            )
        };

        let viewport = vk::Viewport::default()
            .x(0.)
            .y(0.)
            .width(self.draw_image.extent.width as f32)
            .height(self.draw_image.extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);
        let viewports = [viewport];
        unsafe { self.base.device.handle.cmd_set_viewport(cmd, 0, &viewports) };

        let scissor = vk::Rect2D::default().extent(self.draw_image.extent);
        let scissors = [scissor];
        unsafe { self.base.device.handle.cmd_set_scissor(cmd, 0, &scissors) };

        unsafe { self.base.device.handle.cmd_draw(cmd, 3, 1, 0, 0) };

        unsafe { self.base.device.handle.cmd_end_rendering(cmd) };
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for device idle!");
    }
}

pub struct ImguiContext {
    pub renderer: imgui_rs_vulkan_renderer::Renderer,
    pub platform: imgui_winit_support::WinitPlatform,
    pub imgui: imgui::Context,
}

impl ImguiContext {
    pub fn new(
        window: &Window,
        allocator: &Arc<Mutex<Allocator>>,
        device: &Arc<LogicalDevice>,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
        format: vk::Format,
    ) -> Self {
        let mut imgui = imgui::Context::create();

        imgui.set_ini_filename(None);

        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        let dpi_mode = imgui_winit_support::HiDpiMode::Default;

        platform.attach_window(imgui.io_mut(), window, dpi_mode);
        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

        let renderer = imgui_rs_vulkan_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            device.handle.clone(),
            graphics_queue,
            command_pool,
            DynamicRendering {
                color_attachment_format: format,
                depth_attachment_format: None,
            },
            &mut imgui,
            Some(imgui_rs_vulkan_renderer::Options {
                in_flight_frames: FRAME_OVERLAP,
                ..Default::default()
            }),
        )
        .unwrap();

        Self {
            renderer,
            platform,
            imgui,
        }
    }
}

pub fn init_imgui(
    window: &Window,
    allocator: &Arc<Mutex<Allocator>>,
    device: &Arc<LogicalDevice>,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    format: vk::Format,
) -> (
    imgui::Context,
    imgui_winit_support::WinitPlatform,
    imgui_rs_vulkan_renderer::Renderer,
) {
    let mut imgui = imgui::Context::create();

    imgui.set_ini_filename(None);

    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    let dpi_mode = imgui_winit_support::HiDpiMode::Default;

    platform.attach_window(imgui.io_mut(), window, dpi_mode);
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

    let renderer = imgui_rs_vulkan_renderer::Renderer::with_gpu_allocator(
        allocator.clone(),
        device.handle.clone(),
        graphics_queue,
        command_pool,
        DynamicRendering {
            color_attachment_format: format,
            depth_attachment_format: None,
        },
        &mut imgui,
        Some(imgui_rs_vulkan_renderer::Options {
            in_flight_frames: FRAME_OVERLAP,
            ..Default::default()
        }),
    )
    .unwrap();

    (imgui, platform, renderer)
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
