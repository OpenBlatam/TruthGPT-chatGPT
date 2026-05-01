//! Paged Attention for TruthGPT
//!
//! Memory-efficient attention implementation using paging, inspired by vLLM.
//! Enables efficient handling of variable-length sequences and memory sharing.
//!
//! ## Features
//!
//! - Block-based KV cache management
//! - Copy-on-write for shared prefixes
//! - Memory pooling and defragmentation
//! - Support for prefix caching

use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Block size for paged attention (tokens per block)
pub const BLOCK_SIZE: usize = 16;

/// Number of cache blocks per GPU
pub const DEFAULT_NUM_BLOCKS: usize = 8192;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Physical block ID
pub type PhysicalBlockId = u32;

/// Logical block ID (per sequence)
pub type LogicalBlockId = u32;

/// Sequence ID
pub type SeqId = u64;

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICAL BLOCK
// ═══════════════════════════════════════════════════════════════════════════════

/// Physical block storing KV cache data
#[derive(Debug)]
pub struct PhysicalBlock {
    /// Block ID
    pub id: PhysicalBlockId,
    /// Reference count (for copy-on-write)
    pub ref_count: AtomicU32,
    /// Number of tokens stored
    pub num_tokens: AtomicUsize,
    /// Hash for prefix matching (optional)
    pub hash: Option<u64>,
}

impl PhysicalBlock {
    fn new(id: PhysicalBlockId) -> Self {
        Self {
            id,
            ref_count: AtomicU32::new(0),
            num_tokens: AtomicUsize::new(0),
            hash: None,
        }
    }

    fn is_free(&self) -> bool {
        self.ref_count.load(Ordering::Relaxed) == 0
    }

    fn incr_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    fn decr_ref(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::Relaxed) == 1
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLOCK TABLE
// ═══════════════════════════════════════════════════════════════════════════════

/// Block table mapping logical to physical blocks for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Sequence ID
    pub seq_id: SeqId,
    /// Mapping from logical to physical block IDs
    pub blocks: Vec<PhysicalBlockId>,
    /// Number of tokens in the sequence
    pub num_tokens: usize,
}

impl BlockTable {
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            blocks: Vec::new(),
            num_tokens: 0,
        }
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get physical block for a token position
    pub fn get_block(&self, position: usize) -> Option<PhysicalBlockId> {
        let logical_idx = position / BLOCK_SIZE;
        self.blocks.get(logical_idx).copied()
    }

    /// Add a new block
    pub fn push_block(&mut self, physical_id: PhysicalBlockId) {
        self.blocks.push(physical_id);
    }

    /// Check if last block has space
    pub fn last_block_has_space(&self) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        self.num_tokens % BLOCK_SIZE != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLOCK ALLOCATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Block allocator for managing physical blocks
pub struct BlockAllocator {
    /// All physical blocks
    blocks: Vec<Arc<PhysicalBlock>>,
    /// Free block queue
    free_blocks: Mutex<VecDeque<PhysicalBlockId>>,
    /// Number of free blocks
    num_free: AtomicUsize,
    /// Total number of blocks
    num_total: usize,
}

impl BlockAllocator {
    /// Create new allocator with given number of blocks
    pub fn new(num_blocks: usize) -> Self {
        let blocks: Vec<_> = (0..num_blocks)
            .map(|i| Arc::new(PhysicalBlock::new(i as PhysicalBlockId)))
            .collect();

        let free_blocks: VecDeque<_> = (0..num_blocks as PhysicalBlockId).collect();

        Self {
            blocks,
            free_blocks: Mutex::new(free_blocks),
            num_free: AtomicUsize::new(num_blocks),
            num_total: num_blocks,
        }
    }

    /// Allocate a single block
    pub fn allocate(&self) -> Option<PhysicalBlockId> {
        let mut free = self.free_blocks.lock();
        if let Some(block_id) = free.pop_front() {
            self.num_free.fetch_sub(1, Ordering::Relaxed);
            self.blocks[block_id as usize].incr_ref();
            Some(block_id)
        } else {
            None
        }
    }

    /// Allocate multiple blocks
    pub fn allocate_many(&self, count: usize) -> Option<Vec<PhysicalBlockId>> {
        let mut free = self.free_blocks.lock();
        
        if free.len() < count {
            return None;
        }

        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(block_id) = free.pop_front() {
                self.blocks[block_id as usize].incr_ref();
                allocated.push(block_id);
            }
        }
        
        self.num_free.fetch_sub(count, Ordering::Relaxed);
        Some(allocated)
    }

    /// Free a block
    pub fn free(&self, block_id: PhysicalBlockId) {
        let block = &self.blocks[block_id as usize];
        
        // Decrement reference count
        if block.decr_ref() {
            // Last reference - return to free list
            let mut free = self.free_blocks.lock();
            free.push_back(block_id);
            self.num_free.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment reference count (for copy-on-write)
    pub fn incr_ref(&self, block_id: PhysicalBlockId) {
        self.blocks[block_id as usize].incr_ref();
    }

    /// Get reference count
    pub fn ref_count(&self, block_id: PhysicalBlockId) -> u32 {
        self.blocks[block_id as usize].ref_count.load(Ordering::Relaxed)
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.num_free.load(Ordering::Relaxed)
    }

    /// Get total number of blocks
    pub fn num_total_blocks(&self) -> usize {
        self.num_total
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f32 {
        let free = self.num_free.load(Ordering::Relaxed);
        (self.num_total - free) as f32 / self.num_total as f32
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLOCK MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Block manager for handling multiple sequences
pub struct BlockManager {
    allocator: Arc<BlockAllocator>,
    /// Block tables per sequence
    block_tables: RwLock<HashMap<SeqId, BlockTable>>,
    /// Prefix cache: hash -> (block_ids, ref_count)
    prefix_cache: RwLock<HashMap<u64, (Vec<PhysicalBlockId>, u32)>>,
    /// Enable prefix caching
    enable_prefix_caching: bool,
}

impl BlockManager {
    /// Create new block manager
    pub fn new(num_blocks: usize, enable_prefix_caching: bool) -> Self {
        Self {
            allocator: Arc::new(BlockAllocator::new(num_blocks)),
            block_tables: RwLock::new(HashMap::new()),
            prefix_cache: RwLock::new(HashMap::new()),
            enable_prefix_caching,
        }
    }

    /// Allocate blocks for a new sequence
    pub fn allocate_sequence(&self, seq_id: SeqId, num_tokens: usize) -> Result<(), String> {
        let num_blocks = num_tokens.div_ceil(BLOCK_SIZE);
        
        let blocks = self.allocator.allocate_many(num_blocks)
            .ok_or_else(|| "Not enough free blocks".to_string())?;

        let mut table = BlockTable::new(seq_id);
        table.blocks = blocks;
        table.num_tokens = num_tokens;

        self.block_tables.write().insert(seq_id, table);
        Ok(())
    }

    /// Append tokens to a sequence
    pub fn append_tokens(&self, seq_id: SeqId, num_new_tokens: usize) -> Result<(), String> {
        let mut tables = self.block_tables.write();
        let table = tables.get_mut(&seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let old_num_tokens = table.num_tokens;
        let new_num_tokens = old_num_tokens + num_new_tokens;
        
        let old_num_blocks = old_num_tokens.div_ceil(BLOCK_SIZE);
        let new_num_blocks = new_num_tokens.div_ceil(BLOCK_SIZE);

        // Allocate additional blocks if needed
        if new_num_blocks > old_num_blocks {
            let num_to_allocate = new_num_blocks - old_num_blocks;
            let new_blocks = self.allocator.allocate_many(num_to_allocate)
                .ok_or_else(|| "Not enough free blocks".to_string())?;
            
            table.blocks.extend(new_blocks);
        }

        table.num_tokens = new_num_tokens;
        Ok(())
    }

    /// Free a sequence
    pub fn free_sequence(&self, seq_id: SeqId) {
        if let Some(table) = self.block_tables.write().remove(&seq_id) {
            for block_id in table.blocks {
                self.allocator.free(block_id);
            }
        }
    }

    /// Fork a sequence (copy-on-write)
    pub fn fork_sequence(&self, parent_seq_id: SeqId, child_seq_id: SeqId) -> Result<(), String> {
        let tables = self.block_tables.read();
        let parent = tables.get(&parent_seq_id)
            .ok_or_else(|| format!("Parent sequence {} not found", parent_seq_id))?;

        // Create child with same blocks (reference counting)
        let mut child = BlockTable::new(child_seq_id);
        child.blocks = parent.blocks.clone();
        child.num_tokens = parent.num_tokens;

        // Increment reference counts
        for &block_id in &child.blocks {
            self.allocator.incr_ref(block_id);
        }

        drop(tables);
        self.block_tables.write().insert(child_seq_id, child);
        
        Ok(())
    }

    /// Get block table for a sequence
    pub fn get_block_table(&self, seq_id: SeqId) -> Option<BlockTable> {
        self.block_tables.read().get(&seq_id).cloned()
    }

    /// Check if blocks are available
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        let num_blocks = num_tokens.div_ceil(BLOCK_SIZE);
        self.allocator.num_free_blocks() >= num_blocks
    }

    /// Get statistics
    pub fn stats(&self) -> BlockManagerStats {
        BlockManagerStats {
            total_blocks: self.allocator.num_total_blocks(),
            free_blocks: self.allocator.num_free_blocks(),
            num_sequences: self.block_tables.read().len(),
            utilization: self.allocator.utilization(),
        }
    }
}

/// Block manager statistics
#[derive(Debug, Clone)]
pub struct BlockManagerStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub num_sequences: usize,
    pub utilization: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGED ATTENTION METADATA
// ═══════════════════════════════════════════════════════════════════════════════

/// Metadata for paged attention kernel
#[derive(Debug, Clone)]
pub struct PagedAttentionMetadata {
    /// Block tables for all sequences in batch [batch_size, max_blocks]
    pub block_tables: Vec<Vec<PhysicalBlockId>>,
    /// Context lengths for each sequence
    pub context_lens: Vec<usize>,
    /// Maximum context length in batch
    pub max_context_len: usize,
    /// Slot mapping for prefill [total_tokens]
    pub slot_mapping: Vec<i32>,
}

impl PagedAttentionMetadata {
    pub fn new() -> Self {
        Self {
            block_tables: Vec::new(),
            context_lens: Vec::new(),
            max_context_len: 0,
            slot_mapping: Vec::new(),
        }
    }

    /// Build metadata from block manager and sequence IDs
    pub fn build(
        block_manager: &BlockManager,
        seq_ids: &[SeqId],
        is_prefill: bool,
    ) -> Self {
        let mut metadata = Self::new();

        for &seq_id in seq_ids {
            if let Some(table) = block_manager.get_block_table(seq_id) {
                metadata.block_tables.push(table.blocks.clone());
                metadata.context_lens.push(table.num_tokens);
                metadata.max_context_len = metadata.max_context_len.max(table.num_tokens);

                if is_prefill {
                    // Build slot mapping for prefill
                    for pos in 0..table.num_tokens {
                        let block_idx = pos / BLOCK_SIZE;
                        let block_offset = pos % BLOCK_SIZE;
                        let physical_block = table.blocks[block_idx];
                        let slot = physical_block as i32 * BLOCK_SIZE as i32 + block_offset as i32;
                        metadata.slot_mapping.push(slot);
                    }
                }
            }
        }

        metadata
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator() {
        let allocator = BlockAllocator::new(100);
        
        assert_eq!(allocator.num_free_blocks(), 100);
        
        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free_blocks(), 99);
        
        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 100);
    }

    #[test]
    fn test_block_allocator_many() {
        let allocator = BlockAllocator::new(100);
        
        let blocks = allocator.allocate_many(10).unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(allocator.num_free_blocks(), 90);
        
        for block in blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.num_free_blocks(), 100);
    }

    #[test]
    fn test_block_manager_allocation() {
        let manager = BlockManager::new(100, false);
        
        // Allocate sequence with 32 tokens (2 blocks)
        manager.allocate_sequence(1, 32).unwrap();
        
        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.free_blocks, 98);
    }

    #[test]
    fn test_block_manager_append() {
        let manager = BlockManager::new(100, false);
        
        manager.allocate_sequence(1, 10).unwrap();
        let stats1 = manager.stats();
        
        // Append more tokens (should stay in same block)
        manager.append_tokens(1, 5).unwrap();
        let stats2 = manager.stats();
        assert_eq!(stats1.free_blocks, stats2.free_blocks);
        
        // Append more to cross block boundary
        manager.append_tokens(1, 10).unwrap();
        let stats3 = manager.stats();
        assert!(stats3.free_blocks < stats2.free_blocks);
    }

    #[test]
    fn test_fork_sequence() {
        let manager = BlockManager::new(100, false);
        
        manager.allocate_sequence(1, 32).unwrap();
        let free_before = manager.stats().free_blocks;
        
        // Fork should use same blocks (CoW)
        manager.fork_sequence(1, 2).unwrap();
        let free_after = manager.stats().free_blocks;
        
        // No new blocks allocated
        assert_eq!(free_before, free_after);
        
        // Both sequences exist
        assert_eq!(manager.stats().num_sequences, 2);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(1);
        table.push_block(0);
        table.push_block(1);
        table.num_tokens = 20;
        
        assert_eq!(table.get_block(0), Some(0));
        assert_eq!(table.get_block(15), Some(0));
        assert_eq!(table.get_block(16), Some(1));
        assert!(table.last_block_has_space());
    }
}

