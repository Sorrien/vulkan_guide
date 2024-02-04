use std::sync::Arc;

use ash::vk;

use crate::ash_bootstrap::LogicalDevice;

pub struct Descriptor {
    device: Arc<LogicalDevice>,
    pub set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
}

impl Descriptor {
    pub fn new(
        device: Arc<LogicalDevice>,
        set: vk::DescriptorSet,
        layout: vk::DescriptorSetLayout,
    ) -> Self {
        Self {
            device,
            set,
            layout,
        }
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.layout, None)
        };
    }
}

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}
impl DescriptorLayoutBuilder<'_> {
    pub fn new() -> Self {
        Self { bindings: vec![] }
    }

    pub fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        let new_bind = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_count(1)
            .descriptor_type(descriptor_type);
        self.bindings.push(new_bind);
        self
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build(
        mut self,
        device: Arc<LogicalDevice>,
        shader_stages: vk::ShaderStageFlags,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        self.bindings.iter_mut().for_each(|binding| {
            binding.stage_flags |= shader_stages;
        });

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::empty());

        unsafe { device.handle.create_descriptor_set_layout(&info, None) }
    }
}

pub struct DescriptorAllocator {
    device: Arc<LogicalDevice>,
    pool: Option<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    pub fn new(device: Arc<LogicalDevice>) -> Self {
        Self { device, pool: None }
    }

    pub fn init_pool(&mut self, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) {
        let pool_sizes = pool_ratios
            .iter()
            .map(|pool_ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(pool_ratio.desc_type)
                    .descriptor_count((pool_ratio.ratio * (max_sets as f32)) as u32)
            })
            .collect::<Vec<_>>();

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        self.pool = Some(
            unsafe {
                self.device
                    .handle
                    .create_descriptor_pool(&create_info, None)
            }
            .expect("failed to create descriptor pool!"),
        );
    }

    pub fn clear_descriptors(&mut self) {
        if let Some(pool) = self.pool {
            unsafe {
                self.device
                    .handle
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("failed to reset descriptor pool!");
        }
    }

    pub fn destroy_pool(&mut self) {
        if let Some(pool) = self.pool {
            unsafe { self.device.handle.destroy_descriptor_pool(pool, None) };
        }
    }

    pub fn allocate(
        &self,
        layouts: Vec<vk::DescriptorSetLayout>,
    ) -> Option<Vec<vk::DescriptorSet>> {
        if let Some(pool) = self.pool {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            Some(
                unsafe { self.device.handle.allocate_descriptor_sets(&alloc_info) }
                    .expect("failed to allocate descriptor sets!"),
            )
        } else {
            None
        }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        if let Some(pool) = self.pool {
            unsafe { self.device.handle.destroy_descriptor_pool(pool, None) }
        }
    }
}

pub struct PoolSizeRatio {
    pub desc_type: vk::DescriptorType,
    pub ratio: f32,
}
