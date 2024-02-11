use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::{util::Align, vk};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::ash_bootstrap::LogicalDevice;

pub struct AllocatedBuffer {
    device: Arc<LogicalDevice>,
    allocator: Arc<Mutex<Allocator>>,
    pub buffer: vk::Buffer,
    pub allocation: ManuallyDrop<Allocation>,
}

impl AllocatedBuffer {
    pub fn new(
        device: Arc<LogicalDevice>,
        allocator: Arc<Mutex<Allocator>>,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buffer = unsafe { device.handle.create_buffer(&buffer_info, None) }
            .expect("failed to create buffer!");

        let requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("failed to allocate buffer!");

        unsafe {
            device
                .handle
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .expect("failed to bind buffer!");

        Self {
            device,
            allocator,
            buffer,
            allocation: ManuallyDrop::new(allocation),
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        self.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .expect("failed to free memory for allocated image!");
        unsafe {
            self.device.handle.destroy_buffer(self.buffer, None);
        }
    }
}

pub fn copy_to_staging_buffer<DataType: std::marker::Copy>(
    staging_buffer: &AllocatedBuffer,
    size: vk::DeviceSize,
    data: &Vec<DataType>,
) {
    let ptr = unsafe { staging_buffer.allocation.mapped_ptr().unwrap().as_mut() };
    let mut align =
        unsafe { Align::new(ptr, std::mem::align_of::<DataType>() as u64, size as u64) };
    align.copy_from_slice(&data);
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    position: glam::Vec3,
    uv_x: f32,
    normal: glam::Vec3,
    uv_y: f32,
    color: glam::Vec4,
}

impl Vertex {
    pub fn new(position: glam::Vec3, color: glam::Vec4) -> Self {
        Self {
            position,
            uv_x: 0.,
            normal: glam::Vec3::ZERO,
            uv_y: 0.,
            color,
        }
    }
}

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}
pub use offset_of;

pub struct GPUMeshBuffers {
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

#[derive(Clone, Copy)]
pub struct GPUDrawPushConstants {
    world_matrix: glam::Mat4,
    vertex_buffer: vk::DeviceAddress,
}

impl GPUDrawPushConstants {
    pub fn new(world_matrix: glam::Mat4, vertex_buffer: vk::DeviceAddress) -> Self {
        Self {
            world_matrix,
            vertex_buffer,
        }
    }
}
