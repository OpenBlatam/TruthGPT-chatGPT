// The runtime for GPU runs on production code in tensorflow
//https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/common_runtime
// the action is use the api solana and put the login compiler
// and the the tensor a create the login compiler


// [NODE builder components ( )] computes a DFS in the graphs of the AI (sub instructions or links)
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, Arc};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemLoc {
    CPU,
    GPU,
}

#[derive(Debug, Clone)]
pub struct MemDesc {
    pub loc: MemLoc,
    pub dev_index: usize,
    pub gpu_registered: bool,
    pub nic_registered: bool,
}

impl Default for MemDesc {
    fn default() -> Self {
        Self {
            loc: MemLoc::CPU,
            dev_index: 0,
            gpu_registered: false,
            nic_registered: false,
        }
    }
}

impl MemDesc {
    pub fn debug_string(&self) -> String {
        format!(
            "MemDesc {{ loc: {:?}, dev_index: {}, gpu_registered: {}, nic_registered: {} }}",
            self.loc, self.dev_index, self.gpu_registered, self.nic_registered
        )
    }
}

pub trait Allocator: Send + Sync {
    fn name(&self) -> &'static str;
    fn allocate_raw(&self, alignment: usize, size: usize) -> *mut u8;
    fn deallocate_raw(&self, ptr: *mut u8);
    fn tracks_allocation_sizes(&self) -> bool;
    fn requested_size(&self, ptr: *const u8) -> usize;
    fn allocated_size(&self, ptr: *const u8) -> usize;
}

type Visitor = Arc<dyn Fn(*mut u8, usize) + Send + Sync>;

pub struct ProcessState {
    numa_enabled: bool,
    cpu_allocators: Mutex<Vec<Arc<dyn Allocator>>>,
    cpu_alloc_visitors: Mutex<Vec<Visitor>>,
    cpu_free_visitors: Mutex<Vec<Visitor>>,
    mem_desc_map: Mutex<HashMap<*const u8, MemDesc>>,
    cpu_allocators_cache: [OnceLock<Arc<dyn Allocator>>; 8],
}

impl ProcessState {
    fn new() -> Self {
        Self {
            numa_enabled: false,
            cpu_allocators: Mutex::new(Vec::new()),
            cpu_alloc_visitors: Mutex::new(Vec::new()),
            cpu_free_visitors: Mutex::new(Vec::new()),
            mem_desc_map: Mutex::new(HashMap::new()),
            cpu_allocators_cache: Default::default(),
        }
    }

    pub fn singleton() -> &'static Self {
        static INSTANCE: OnceLock<ProcessState> = OnceLock::new();
        INSTANCE.get_or_init(|| ProcessState::new())
    }

    pub fn enable_numa(&mut self) {
        self.numa_enabled = true;
    }

    pub fn add_cpu_alloc_visitor(&self, visitor: Visitor) {
        self.cpu_alloc_visitors.lock().unwrap().push(visitor);
    }

    pub fn add_cpu_free_visitor(&self, visitor: Visitor) {
        self.cpu_free_visitors.lock().unwrap().push(visitor);
    }

    pub fn ptr_type(&self, ptr: *const u8) -> MemDesc {
        self.mem_desc_map
            .lock()
            .unwrap()
            .get(&ptr)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_cpu_allocator(&self, numa_node: usize) -> Arc<dyn Allocator> {
        let index = if numa_node == usize::MAX { 0 } else { numa_node };
        if let Some(cached) = self.cpu_allocators_cache.get(index).and_then(|c| c.get()) {
            return cached.clone();
        }

        let allocators = self.cpu_allocators.lock().unwrap();
        let allocator = allocators
            .get(index)
            .expect("Allocator not found for NUMA node")
            .clone();

        self.cpu_allocators_cache[index]
            .set(allocator.clone())
            .ok();

        allocator
    }

    pub fn record_allocation(&self, ptr: *mut u8, desc: MemDesc) {
        self.mem_desc_map.lock().unwrap().insert(ptr, desc);
    }

    pub fn record_deallocation(&self, ptr: *mut u8) {
        self.mem_desc_map.lock().unwrap().remove(&ptr);
    }
}
