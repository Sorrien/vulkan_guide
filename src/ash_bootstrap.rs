use std::ffi::{c_char, c_void, CStr, CString};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{self, PhysicalDeviceFeatures2},
};
use raw_window_handle::RawDisplayHandle;

use crate::debug;

//use gpu_allocator::vulkan::*;
pub struct InstanceBuilder {
    entry: Option<ash::Entry>,
    enable_validation_layers: bool,
    raw_display_handle: Option<RawDisplayHandle>,
    api_version: u32,
    application_name: String,
    application_version: u32,
    engine_name: String,
    engine_version: u32,
}

impl InstanceBuilder {
    pub fn new() -> Self {
        Self {
            entry: None,
            enable_validation_layers: false,
            raw_display_handle: None,
            application_name: String::from("Vulkan Application"),
            application_version: vk::make_api_version(0, 1, 0, 0),
            engine_name: String::from("No Engine"),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
        }
    }

    pub fn entry(mut self, entry: ash::Entry) -> Self {
        self.entry = Some(entry);
        self
    }

    pub fn enable_validation_layers(mut self, enable_validation_layers: bool) -> Self {
        self.enable_validation_layers = enable_validation_layers;
        self
    }

    pub fn raw_display_handle(mut self, raw_display_handle: RawDisplayHandle) -> Self {
        self.raw_display_handle = Some(raw_display_handle);
        self
    }

    pub fn api_version(mut self, api_version: u32) -> Self {
        self.api_version = api_version;
        self
    }

    pub fn application_name(mut self, application_name: String) -> Self {
        self.application_name = application_name;
        self
    }

    pub fn application_version(mut self, application_version: u32) -> Self {
        self.application_version = application_version;
        self
    }

    pub fn engine_name(mut self, engine_name: String) -> Self {
        self.engine_name = engine_name;
        self
    }

    pub fn engine_version(mut self, engine_version: u32) -> Self {
        self.engine_version = engine_version;
        self
    }

    pub fn build(self) -> ash::Instance {
        let app_name_cstring = CString::new(self.application_name.as_str()).unwrap();
        let app_name = app_name_cstring.as_c_str();
        let engine_name_cstring = CString::new(self.engine_name.as_str()).unwrap();
        let engine_name = engine_name_cstring.as_c_str();

        let entry = &self.entry.expect("no entry was provided!");

        let mut required_extension_names = vec![];

        if let Some(raw_display_handle) = self.raw_display_handle {
            let mut window_extension_names =
                ash_window::enumerate_required_extensions(raw_display_handle)
                    .unwrap()
                    .to_vec();

            required_extension_names.append(&mut window_extension_names);
        }

        if self.enable_validation_layers {
            required_extension_names.push(DebugUtils::NAME.as_ptr());

            println!("Validation Layers enabled!");
        }

        let extension_properties = unsafe { entry.enumerate_instance_extension_properties(None) }
            .expect("failed to enumerate instance extension props!");

        println!("Enabled extensions:");
        for extension_name in required_extension_names.iter() {
            let str = unsafe { CStr::from_ptr(*extension_name) }
                .to_str()
                .expect("failed to get ext name str");

            if extension_properties.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required extensions were not available!");
            }
        }
        println!("");

        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(self.application_version)
            .engine_name(engine_name)
            .engine_version(self.engine_version)
            .api_version(self.api_version);

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);

        let debug_info = debug::create_debug_info();
        if self.enable_validation_layers {
            Self::check_validation_layer_support(entry, &layers_names_raw);
            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next =
                &debug_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);

        let debug_info = debug::create_debug_info();
        if self.enable_validation_layers {
            Self::check_validation_layer_support(entry, &layers_names_raw);
            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next =
                &debug_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error!")
        };
        instance
    }

    fn check_validation_layer_support(entry: &ash::Entry, layers_names_raw: &Vec<*const c_char>) {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("failed to get available layers!");

        for name in layers_names_raw.iter() {
            let str = unsafe { CStr::from_ptr(*name) }
                .to_str()
                .expect("failed to get layer name str");

            if available_layers.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required layers were not available!");
            }
        }
    }
}

pub struct PhysicalDeviceSelector<'a> {
    instance: &'a ash::Instance,
    surface_loader: &'a Surface,
    surface: &'a vk::SurfaceKHR,
    criteria: SelectionCriteria<'a>,
}

pub struct SelectionCriteria<'a> {
    required_extensions: Vec<CString>,
    features: vk::PhysicalDeviceFeatures,
    features12: vk::PhysicalDeviceVulkan12Features<'a>,
    features13: vk::PhysicalDeviceVulkan13Features<'a>,
}

impl SelectionCriteria<'_> {
    pub fn new() -> Self {
        Self {
            required_extensions: vec![],
            features: vk::PhysicalDeviceFeatures::default(),
            features12: vk::PhysicalDeviceVulkan12Features::default(),
            features13: vk::PhysicalDeviceVulkan13Features::default(),
        }
    }
}

impl<'a> PhysicalDeviceSelector<'a> {
    pub fn new(
        instance: &'a ash::Instance,
        surface_loader: &'a Surface,
        surface: &'a vk::SurfaceKHR,
    ) -> PhysicalDeviceSelector<'a> {
        Self {
            instance,
            surface_loader,
            surface,
            criteria: SelectionCriteria::new(),
        }
    }

    pub fn set_required_extensions(mut self, required_extensions: Vec<CString>) -> Self {
        self.criteria.required_extensions = required_extensions;
        self
    }

    pub fn set_required_features(mut self, required_features: vk::PhysicalDeviceFeatures) -> Self {
        self.criteria.features = required_features;
        self
    }

    pub fn set_required_features_12(
        mut self,
        required_features: vk::PhysicalDeviceVulkan12Features<'a>,
    ) -> Self {
        self.criteria.features12 = required_features;
        self
    }

    pub fn set_required_features_13(
        mut self,
        required_features: vk::PhysicalDeviceVulkan13Features<'a>,
    ) -> Self {
        self.criteria.features13 = required_features;
        self
    }

    pub fn select(&self) -> Option<BootstrapPhysicalDevice> {
        let physical_devices = unsafe { self.instance.enumerate_physical_devices() }
            .expect("failed to find physical devices!");

        if physical_devices.len() == 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        let mut scored_devices = physical_devices
            .iter()
            .filter_map(|device| {
                if let Some((score, physical_device)) = self.rate_device(device) {
                    Some((score, physical_device))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored_devices.sort_by(|(a, _), (b, _)| a.cmp(b));

        if let Some((_, device)) = scored_devices.last() {
            Some(device.clone())
        } else {
            None
        }
    }

    fn rate_device(&self, device: &vk::PhysicalDevice) -> Option<(u32, BootstrapPhysicalDevice)> {
        let mut device_features_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut device_features_12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut device_features_12)
            .push_next(&mut device_features_13);
        unsafe {
            self.instance
                .get_physical_device_features2(*device, &mut physical_device_features)
        };
        let features = physical_device_features.features;

        let has_required_features = self.check_for_required_features(features)
            && self.check_for_required_features_12(device_features_12)
            && self.check_for_required_features_13(device_features_13);

        if !has_required_features {
            println!("failed to get required features!");
            return None;
        }

        let has_required_extensions = self.check_for_required_extensions(*device);

        if !has_required_extensions {
            println!("failed to get required extensions!");
            return None;
        }

        let mut score = 0;

        let properties = unsafe { self.instance.get_physical_device_properties(*device) };

        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            // Discrete GPUs have a significant performance advantage
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += properties.limits.max_image_dimension2_d;

        let physical_device_properties =
            unsafe { self.instance.get_physical_device_properties(*device) };
        let device_name =
            unsafe { CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_str() }
                .unwrap();
        let queue_families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(*device)
        };

        let queue_family_indices =
            QueueFamilyIndices::new(&queue_families, self.surface_loader, device, self.surface);
        if queue_family_indices.graphics_family.is_some() {
            let swapchain_support_details =
                SwapchainSupportDetails::new(device, self.surface_loader, self.surface);

            if swapchain_support_details.formats.len() > 0
                && swapchain_support_details.present_modes.len() > 0
            {
                let msaa_samples = Self::get_max_usable_sample_count(physical_device_properties);

                Some((
                    score,
                    BootstrapPhysicalDevice {
                        physical_device: *device,
                        physical_device_properties,
                        swapchain_support_details,
                        queue_family_indices,
                        max_sample_count: msaa_samples,
                    },
                ))
            } else {
                println!(
                    "failed to find swapchain format or present mode on physical device! {}",
                    device_name
                );
                None
            }
        } else {
            println!(
                "failed to find graphics family queue on physical device! {}",
                device_name
            );
            None
        }
    }

    pub fn get_max_usable_sample_count(
        physical_device_properties: vk::PhysicalDeviceProperties,
    ) -> vk::SampleCountFlags {
        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        let sorted_desired_sample_counts = [
            vk::SampleCountFlags::TYPE_64,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ];

        let mut allowed_sample_counts = sorted_desired_sample_counts
            .iter()
            .filter(|desired_sample_count| counts.contains(**desired_sample_count));

        if let Some(sample_count) = allowed_sample_counts.next() {
            *sample_count
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn check_for_required_extensions(&self, device: vk::PhysicalDevice) -> bool {
        let device_extension_properties =
            unsafe { self.instance.enumerate_device_extension_properties(device) }
                .expect("failed to get device extension properties!");
        let mut device_extension_names = device_extension_properties
            .iter()
            .map(|prop| unsafe { CString::from(CStr::from_ptr(prop.extension_name.as_ptr())) });
        for required_extension in &self.criteria.required_extensions {
            if device_extension_names
                .find(|extension| *extension == *required_extension)
                .is_none()
            {
                println!(
                    "couldn't find extension {}",
                    required_extension.to_str().unwrap()
                );
                return false;
            } else {
                println!("found extension {}", required_extension.to_str().unwrap());
            }
        }

        return true;
    }

    fn check_for_required_features(&self, device_features: vk::PhysicalDeviceFeatures) -> bool {
        let criteria_features = self.criteria.features;
        if criteria_features.geometry_shader == 1 && device_features.geometry_shader == 0 {
            false
        } else if criteria_features.sampler_anisotropy == 1
            && device_features.sampler_anisotropy == 0
        {
            false
        } else if criteria_features.alpha_to_one == 1 && device_features.alpha_to_one == 0 {
            false
        } else if criteria_features.depth_bias_clamp == 1 && device_features.depth_bias_clamp == 0 {
            false
        } else if criteria_features.depth_bounds == 1 && device_features.depth_bounds == 0 {
            false
        } else if criteria_features.depth_clamp == 1 && device_features.depth_clamp == 0 {
            false
        } else if criteria_features.draw_indirect_first_instance == 1
            && device_features.draw_indirect_first_instance == 0
        {
            false
        } else {
            true
        }
    }

    fn check_for_required_features_12(
        &self,
        device_features: vk::PhysicalDeviceVulkan12Features,
    ) -> bool {
        let criteria_features = self.criteria.features12;
        if criteria_features.buffer_device_address == 1
            && device_features.buffer_device_address == 0
        {
            false
        } else if criteria_features.descriptor_indexing == 1
            && device_features.descriptor_indexing == 0
        {
            false
        } else {
            true
        }
    }

    fn check_for_required_features_13(
        &self,
        device_features: vk::PhysicalDeviceVulkan13Features,
    ) -> bool {
        let criteria_features = self.criteria.features13;
        if criteria_features.dynamic_rendering == 1 && device_features.dynamic_rendering == 0 {
            false
        } else if criteria_features.synchronization2 == 1 && device_features.synchronization2 == 0 {
            false
        } else {
            true
        }
    }
}

#[derive(Clone)]
pub struct BootstrapPhysicalDevice {
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_support_details: SwapchainSupportDetails,
    pub max_sample_count: vk::SampleCountFlags,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &QueueFamilyIndices,
    required_extensions: Vec<CString>,
    device_features: vk::PhysicalDeviceFeatures,
    mut device_features12: vk::PhysicalDeviceVulkan12Features,
    mut device_features13: vk::PhysicalDeviceVulkan13Features,
) -> Result<ash::Device, vk::Result> {
    let queue_create_infos = [queue_family_indices.graphics_family.unwrap() as u32].map(|i| {
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(i)
            .queue_priorities(&[1.])
    });

    let device_extension_names_raw = required_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_features(&device_features)
        .enabled_extension_names(&device_extension_names_raw)
        .push_next(&mut device_features12)
        .push_next(&mut device_features13);

    unsafe { instance.create_device(physical_device, &device_create_info, None) }
}

#[derive(Clone, Copy)]
pub struct QueueFamilyIndices {
    pub graphics_family: Option<usize>,
    pub present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new(
        queue_families: &Vec<vk::QueueFamilyProperties>,
        surface_loader: &Surface,
        pdevice: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        let graphics_family_index = if let Some((graphics_family_index, _)) =
            queue_families.iter().enumerate().find(|(_, queue)| {
                queue.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) {
            Some(graphics_family_index)
        } else {
            None
        };

        let present_family_index = if let Some((present_family_index, _)) =
            queue_families.iter().enumerate().find(|(i, _)| {
                unsafe {
                    surface_loader
                        .get_physical_device_surface_support(*pdevice, *i as u32, *surface)
                }
                .unwrap()
            }) {
            Some(present_family_index)
        } else {
            None
        };

        Self {
            graphics_family: graphics_family_index,
            present_family: present_family_index,
        }
    }
}

#[derive(Clone)]
pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: &vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(*device, *surface) }
                .expect("failed to get surface capabilites!");
        let formats =
            unsafe { surface_loader.get_physical_device_surface_formats(*device, *surface) }
                .expect("failed to get device surface formats!");
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(*device, *surface) }
                .expect("failed to get device surface present modes!");

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }
}

pub struct SwapchainBuilder {
    swapchain_support: SwapchainSupportDetails,
    queue_family_indices: QueueFamilyIndices,
    instance: ash::Instance,
    device: ash::Device,
    surface: vk::SurfaceKHR,
    window_width: u32,
    window_height: u32,
    desired_present_mode: vk::PresentModeKHR,
    desired_surface_format: vk::SurfaceFormatKHR,
    swapchain_image_usage_flags: vk::ImageUsageFlags,
}

impl SwapchainBuilder {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        surface: vk::SurfaceKHR,
        swapchain_support: SwapchainSupportDetails,
        queue_family_indices: QueueFamilyIndices,
    ) -> Self {
        Self {
            swapchain_support,
            queue_family_indices,
            instance,
            device,
            surface,
            window_width: 800,
            window_height: 600,
            desired_present_mode: vk::PresentModeKHR::MAILBOX,
            desired_surface_format: vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_SRGB)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            swapchain_image_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        }
    }

    pub fn desired_extent(mut self, window_width: u32, window_height: u32) -> Self {
        self.window_height = window_height;
        self.window_width = window_width;
        self
    }

    pub fn desired_present_mode(mut self, desired_present_mode: vk::PresentModeKHR) -> Self {
        self.desired_present_mode = desired_present_mode;
        self
    }

    pub fn desired_surface_format(mut self, desired_surface_format: vk::SurfaceFormatKHR) -> Self {
        self.desired_surface_format = desired_surface_format;
        self
    }

    pub fn add_image_usage_flags(mut self, image_usage_flags: vk::ImageUsageFlags) -> Self {
        self.swapchain_image_usage_flags |= image_usage_flags;
        self
    }

    pub fn build(self) -> BootstrapSwapchain {
        let (swapchain, swapchain_loader, format, extent) = self
            .create_swapchain(self.window_width, self.window_height)
            .expect("failed to create swapchain!");

        let swapchain_images = Self::create_swapchain_images(swapchain, swapchain_loader.clone())
            .expect("failed to get swapchain images!");

        let swapchain_image_views = self.create_swapchain_image_views(&swapchain_images, format);

        BootstrapSwapchain {
            swapchain,
            swapchain_loader,
            format,
            extent,
            swapchain_images,
            swapchain_image_views,
        }
    }
    fn create_swapchain(
        &self,
        window_width: u32,
        window_height: u32,
    ) -> Result<(vk::SwapchainKHR, Swapchain, vk::Format, vk::Extent2D), vk::Result> {
        let surface_format = self
            .choose_swapchain_surface_format(&self.swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode =
            self.choose_swapchain_present_mode(&self.swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(
            &self.swapchain_support.capabilities,
            window_width,
            window_height,
        );

        let image_count = self.swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if self.swapchain_support.capabilities.max_image_count > 0
            && image_count > self.swapchain_support.capabilities.max_image_count
        {
            self.swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) = if self.queue_family_indices.graphics_family
            != self.queue_family_indices.present_family
        {
            (
                vk::SharingMode::CONCURRENT,
                vec![
                    self.queue_family_indices.graphics_family.unwrap() as u32,
                    self.queue_family_indices.present_family.unwrap() as u32,
                ],
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, vec![])
        };

        let swapchain_loader = Swapchain::new(&self.instance, &self.device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(self.swapchain_image_usage_flags)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(self.swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        Ok((swapchain, swapchain_loader, surface_format.format, extent))
    }

    fn choose_swapchain_surface_format<'a>(
        &'a self,
        available_formats: &'a Vec<vk::SurfaceFormatKHR>,
    ) -> Option<&vk::SurfaceFormatKHR> {
        if let Some(desired_format) = available_formats.iter().find(|surface_format| {
            surface_format.color_space == self.desired_surface_format.color_space
                && surface_format.format == self.desired_surface_format.format
        }) {
            Some(desired_format)
        } else {
            available_formats.first()
        }
    }

    fn choose_swapchain_present_mode(
        &self,
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        let desired_mode = self.desired_present_mode;
        let is_desired_mode_available = available_present_modes
            .iter()
            .any(|present_mode| *present_mode == desired_mode);
        if is_desired_mode_available {
            desired_mode
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> vk::Extent2D {
        match capabilities.current_extent.width {
            //the max value of u32 is a special value to indicate that we must choose a resolution with the current min and max extents
            //should look into how DPI scaling is handled by winit and if this is the pixel extent or if this includes dpi scaling.
            u32::MAX => vk::Extent2D {
                width: window_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            },
            _ => capabilities.current_extent,
        }
    }

    fn create_swapchain_images(
        swapchain: vk::SwapchainKHR,
        swapchain_loader: Swapchain,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        unsafe { swapchain_loader.get_swapchain_images(swapchain) }
    }

    fn create_swapchain_image_views(
        &self,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: vk::Format,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                self.create_image_view(
                    *image,
                    swapchain_image_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_image_view(
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

        let image_view = unsafe { self.device.create_image_view(&image_view_create_info, None) }
            .expect("failed to create image view!");
        image_view
    }
}

pub struct BootstrapSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_loader: Swapchain,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}
