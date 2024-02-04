use vulkan_guide::VulkanEngine;

fn main() -> Result<(), winit::error::EventLoopError> {
    let title = String::from("Example Vulkan Application");
    let (event_loop, window) = VulkanEngine::init_window(800, 600, &title);

    let mut vulkan_engine = VulkanEngine::new(window, title);
    vulkan_engine.init_commands();
    //vulkan_engine.init_descriptors();
    let (imgui, platform, imgui_renderer) = vulkan_engine.init_imgui();
    vulkan_engine.run(event_loop, imgui, platform, imgui_renderer)
}
