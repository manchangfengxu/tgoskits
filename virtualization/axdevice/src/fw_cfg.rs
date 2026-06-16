// Copyright 2025 The Axvisor Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::{collections::BTreeMap, string::String, vec, vec::Vec};

use ax_errno::{AxResult, ax_err};
use ax_kspin::SpinNoIrq as Mutex;
use axaddrspace::GuestMemoryAccessor;
use axdevice_base::{AccessWidth, BaseDeviceOps, Port, PortRange};
use axvm_types::{EmulatedDeviceType, GuestPhysAddr};

const FW_CFG_IO_SELECTOR: u16 = 0x510;
/// Port number for the fw_cfg data register.
pub const FW_CFG_IO_DATA: u16 = 0x511;
const FW_CFG_IO_DMA_ADDRESS: u16 = 0x514;

const QEMU_FW_CFG_FNAME_SIZE: usize = 56;

const QEMU_FW_CFG_ITEM_SIGNATURE: u16 = 0x0000;
const QEMU_FW_CFG_ITEM_INTERFACE_VERSION: u16 = 0x0001;
const QEMU_FW_CFG_ITEM_SMP_CPU_COUNT: u16 = 0x0005;
const QEMU_FW_CFG_ITEM_FILE_DIR: u16 = 0x0019;
const QEMU_FW_CFG_ITEM_ETC_E820: u16 = 0x8000;
const FW_CFG_FILE_FIRST: u16 = 0x0020;

const FW_CFG_F_DMA: u32 = 1 << 1;

/// DMA control bits used when executing a fw_cfg DMA transaction.
pub const FW_CFG_DMA_CTL_ERROR: u32 = 1 << 0;
pub const FW_CFG_DMA_CTL_READ: u32 = 1 << 1;
pub const FW_CFG_DMA_CTL_SKIP: u32 = 1 << 2;
pub const FW_CFG_DMA_CTL_SELECT: u32 = 1 << 3;
pub const FW_CFG_DMA_CTL_WRITE: u32 = 1 << 4;

/// Backing data for a fw_cfg item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FwCfgContent {
    Bytes(Vec<u8>),
    U32(u32),
}

impl FwCfgContent {
    fn len(&self) -> AxResult<usize> {
        Ok(match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::U32(_) => size_of::<u32>(),
        })
    }

    fn read_bytes(&self, offset: usize, size: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(size);
        match self {
            Self::Bytes(bytes) => {
                let end = (offset + size).min(bytes.len());
                if offset < end {
                    out.extend_from_slice(&bytes[offset..end]);
                }
            }
            Self::U32(value) => {
                let bytes = value.to_le_bytes();
                let end = (offset + size).min(bytes.len());
                if offset < end {
                    out.extend_from_slice(&bytes[offset..end]);
                }
            }
        }
        out.resize(size, 0);
        out
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FwCfgFileItem {
    selector: u16,
    name: String,
    content: FwCfgContent,
}

/// Parsed fw_cfg DMA descriptor.
///
/// The descriptor fields are big-endian, matching the QEMU fw_cfg DMA ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FwCfgDmaRequest {
    pub control: u32,
    pub length: usize,
    pub address: GuestPhysAddr,
}

impl FwCfgDmaRequest {
    pub fn parse(descriptor: [u8; 16]) -> Self {
        let control = u32::from_be_bytes(descriptor[0..4].try_into().unwrap());
        let length = u32::from_be_bytes(descriptor[4..8].try_into().unwrap()) as usize;
        let address = u64::from_be_bytes(descriptor[8..16].try_into().unwrap()) as usize;
        Self {
            control,
            length,
            address: GuestPhysAddr::from(address),
        }
    }
}

struct FwCfgInner {
    selector: u16,
    offset: usize,
    dma_bytes: [u8; 8],
    dma_bytes_written: usize,
    pending_dma: Option<GuestPhysAddr>,
    known_items: BTreeMap<u16, FwCfgContent>,
    file_items: Vec<FwCfgFileItem>,
}

impl FwCfgInner {
    fn new() -> Self {
        Self {
            selector: 0,
            offset: 0,
            dma_bytes: [0; 8],
            dma_bytes_written: 0,
            pending_dma: None,
            known_items: BTreeMap::new(),
            file_items: Vec::new(),
        }
    }

    fn configure(&mut self, memory_regions: &[(u64, u64)], cpu_count: usize) {
        self.known_items.clear();
        self.file_items.clear();
        self.known_items.insert(
            QEMU_FW_CFG_ITEM_SIGNATURE,
            FwCfgContent::Bytes(b"QEMU".to_vec()),
        );
        self.known_items.insert(
            QEMU_FW_CFG_ITEM_INTERFACE_VERSION,
            FwCfgContent::U32(FW_CFG_F_DMA),
        );
        self.known_items.insert(
            QEMU_FW_CFG_ITEM_SMP_CPU_COUNT,
            FwCfgContent::Bytes((cpu_count as u16).to_le_bytes().to_vec()),
        );
        self.file_items.push(FwCfgFileItem {
            selector: QEMU_FW_CFG_ITEM_ETC_E820,
            name: String::from("etc/e820"),
            content: FwCfgContent::Bytes(build_e820(memory_regions)),
        });
        self.rebuild_file_dir();
    }

    fn add_file_item(&mut self, name: &str, content: FwCfgContent) -> AxResult<u16> {
        if name.len() >= QEMU_FW_CFG_FNAME_SIZE {
            return ax_err!(InvalidInput, "fw_cfg file name is too long");
        }
        let selector = self.allocate_file_selector()?;
        self.file_items.push(FwCfgFileItem {
            selector,
            name: String::from(name),
            content,
        });
        self.rebuild_file_dir();
        Ok(selector)
    }

    fn allocate_file_selector(&self) -> AxResult<u16> {
        for selector in FW_CFG_FILE_FIRST..QEMU_FW_CFG_ITEM_ETC_E820 {
            if !self.file_items.iter().any(|item| item.selector == selector) {
                return Ok(selector);
            }
        }
        ax_err!(InvalidInput, "fw_cfg file selector space is exhausted")
    }

    fn rebuild_file_dir(&mut self) {
        let dir = self.build_file_dir();
        self.known_items
            .insert(QEMU_FW_CFG_ITEM_FILE_DIR, FwCfgContent::Bytes(dir));
    }

    fn build_file_dir(&self) -> Vec<u8> {
        let mut dir = Vec::new();
        append_u32_be(&mut dir, self.file_items.len() as u32);
        for item in &self.file_items {
            append_u32_be(&mut dir, item.content.len().unwrap_or(0) as u32);
            append_u16_be(&mut dir, item.selector);
            append_u16_be(&mut dir, 0);
            let mut name_bytes = [0u8; QEMU_FW_CFG_FNAME_SIZE];
            let bytes = item.name.as_bytes();
            name_bytes[..bytes.len()].copy_from_slice(bytes);
            dir.extend_from_slice(&name_bytes);
        }
        dir
    }

    fn select(&mut self, selector: u16) {
        self.selector = selector;
        self.offset = 0;
        debug!("fw_cfg select item={selector:#x}");
    }

    fn read_port(&mut self, width: AccessWidth) -> usize {
        let mut value = 0usize;
        let bytes = self.read_bytes(width.size());
        for (index, byte) in bytes.iter().enumerate() {
            value |= (*byte as usize) << (index * 8);
        }
        value
    }

    fn read_bytes(&mut self, size: usize) -> Vec<u8> {
        let offset = self.offset;
        if let Some(item) = self.selected_content() {
            let bytes = item.read_bytes(offset, size);
            let item_len = item.len().unwrap_or(0);
            self.offset = (offset + size).min(item_len);
            bytes
        } else {
            vec![0; size]
        }
    }

    /// Writes `data` into the currently-selected item at the current offset.
    /// Used by fw_cfg DMA WRITE control.
    fn write_bytes(&mut self, data: &[u8]) {
        // TODO: implement item mutation once fw_cfg WRITE semantics are needed.
        debug!("fw_cfg DMA WRITE: {} bytes (no-op)", data.len());
    }

    fn skip_bytes(&mut self, size: usize) {
        self.offset += size;
        if let Some(item) = self.selected_content() {
            self.offset = self.offset.min(item.len().unwrap_or(0));
        }
    }

    fn selected_content(&self) -> Option<&FwCfgContent> {
        self.known_items.get(&self.selector).or_else(|| {
            self.file_items
                .iter()
                .find(|item| item.selector == self.selector)
                .map(|item| &item.content)
        })
    }

    fn write_dma_address_part(&mut self, value: usize, high_half: bool) {
        let part = u32::from_be(value as u32);
        if high_half {
            self.dma_bytes[0..4].copy_from_slice(&part.to_be_bytes());
            self.dma_bytes_written = 4;
        } else {
            self.dma_bytes[4..8].copy_from_slice(&part.to_be_bytes());
            if self.dma_bytes_written == 4 {
                let high = u32::from_be_bytes(self.dma_bytes[0..4].try_into().unwrap()) as u64;
                let low = u32::from_be_bytes(self.dma_bytes[4..8].try_into().unwrap()) as u64;
                self.pending_dma = Some(GuestPhysAddr::from(((high << 32) | low) as usize));
            }
            self.dma_bytes_written = 0;
        }
    }

    fn take_pending_dma(&mut self) -> Option<GuestPhysAddr> {
        self.pending_dma.take()
    }
}

/// QEMU fw_cfg device providing platform configuration to OVMF guests.
///
/// Exposes selector (0x510), data (0x511), and DMA address ports (0x514 / 0x518)
/// through [`BaseDeviceOps<PortRange>`]. DMA-address writes queue a descriptor
/// GPA; `execute_dma` consumes it with a caller-provided guest-memory
/// accessor. String I/O reads use `read_string_bytes`.
pub struct FwCfgDevice {
    inner: Mutex<FwCfgInner>,
}

impl FwCfgDevice {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FwCfgInner::new()),
        }
    }

    /// Populates the fw_cfg item set from the VM memory map and CPU count.
    pub fn configure(&self, memory_regions: &[(u64, u64)], cpu_count: usize) {
        self.inner.lock().configure(memory_regions, cpu_count);
    }

    /// Returns and clears a pending DMA descriptor GPA, if any.
    ///
    /// `AxVmDevices` uses this before calling `execute_dma`.
    pub fn take_pending_dma(&self) -> Option<GuestPhysAddr> {
        self.inner.lock().take_pending_dma()
    }

    /// Reads `count` bytes from the currently-selected item for string I/O.
    pub fn read_string_bytes(&self, count: usize) -> Vec<u8> {
        self.inner.lock().read_bytes(count)
    }

    /// Executes a pending DMA request whose descriptor is at `dma_gpa`.
    ///
    /// Reads the 16-byte DMA descriptor from guest memory via `mem`, performs
    /// the requested operation (select / read / write / skip), and writes the
    /// status word back to the descriptor. The inner lock is held only during
    /// brief item-state operations, not during guest memory transfers.
    pub fn execute_dma<M: GuestMemoryAccessor>(&self, dma_gpa: GuestPhysAddr, mem: &M) -> AxResult {
        let mut descriptor = [0u8; 16];
        mem.read_buffer(dma_gpa, &mut descriptor)?;
        let request = FwCfgDmaRequest::parse(descriptor);

        if request.control & FW_CFG_DMA_CTL_SELECT != 0 {
            self.inner.lock().select((request.control >> 16) as u16);
        }

        if request.control & FW_CFG_DMA_CTL_READ != 0 {
            let bytes = self.inner.lock().read_bytes(request.length);
            mem.write_buffer(request.address, &bytes)?;
        } else if request.control & FW_CFG_DMA_CTL_WRITE != 0 {
            let mut bytes = alloc::vec![0u8; request.length];
            mem.read_buffer(request.address, &mut bytes)?;
            self.inner.lock().write_bytes(&bytes);
        } else if request.control & FW_CFG_DMA_CTL_SKIP != 0 {
            self.inner.lock().skip_bytes(request.length);
        }

        let status: u32 = if request.control & FW_CFG_DMA_CTL_ERROR != 0 {
            FW_CFG_DMA_CTL_ERROR
        } else {
            0
        };
        mem.write_buffer(dma_gpa, &status.to_be_bytes())
    }

    /// Adds a named fw_cfg file item and returns its selector.
    pub fn add_file_item(&self, name: &str, content: FwCfgContent) -> AxResult<u16> {
        self.inner.lock().add_file_item(name, content)
    }
}

impl Default for FwCfgDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseDeviceOps<PortRange> for FwCfgDevice {
    fn emu_type(&self) -> EmulatedDeviceType {
        EmulatedDeviceType::X86FwCfg
    }

    fn address_range(&self) -> PortRange {
        PortRange::new(
            Port::new(FW_CFG_IO_SELECTOR),
            Port::new(FW_CFG_IO_DMA_ADDRESS + 4),
        )
    }

    fn handle_read(&self, addr: Port, width: AccessWidth) -> AxResult<usize> {
        match addr.number() {
            FW_CFG_IO_DATA => Ok(self.inner.lock().read_port(width)),
            _ => Ok(0),
        }
    }

    fn handle_write(&self, addr: Port, width: AccessWidth, val: usize) -> AxResult {
        let mut g = self.inner.lock();
        match addr.number() {
            FW_CFG_IO_SELECTOR if width == AccessWidth::Word => {
                g.select(val as u16);
            }
            FW_CFG_IO_DMA_ADDRESS | 0x518 if width == AccessWidth::Dword => {
                g.write_dma_address_part(val, addr.number() == FW_CFG_IO_DMA_ADDRESS);
            }
            _ => {}
        }
        Ok(())
    }
}

fn build_e820(memory_regions: &[(u64, u64)]) -> Vec<u8> {
    let mut e820 = Vec::new();
    for &(gpa, size) in memory_regions {
        if gpa >= 0x1_0000_0000 {
            continue;
        }
        let end = (gpa + size).min(0x1_0000_0000);
        append_u64_le(&mut e820, gpa);
        append_u64_le(&mut e820, end - gpa);
        append_u32_le(&mut e820, 1);
    }
    e820
}

fn append_u16_be(buffer: &mut Vec<u8>, value: u16) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

fn append_u32_be(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

fn append_u32_le(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn append_u64_le(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use axdevice_base::AccessWidth;

    use super::{
        FwCfgContent, FwCfgDmaRequest, FwCfgInner, QEMU_FW_CFG_ITEM_FILE_DIR,
        QEMU_FW_CFG_ITEM_INTERFACE_VERSION, QEMU_FW_CFG_ITEM_SIGNATURE,
    };

    #[test]
    fn known_item_reads_reset_offset_on_select() {
        let mut fw_cfg = FwCfgInner::new();
        fw_cfg.configure(&[], 1);

        fw_cfg.select(QEMU_FW_CFG_ITEM_SIGNATURE);
        assert_eq!(fw_cfg.read_port(AccessWidth::Word), 0x4551);
        fw_cfg.select(QEMU_FW_CFG_ITEM_SIGNATURE);
        assert_eq!(fw_cfg.read_port(AccessWidth::Word), 0x4551);
    }

    #[test]
    fn content_model_supports_bytes_and_u32() {
        let bytes = FwCfgContent::Bytes(b"ab".to_vec());
        let number = FwCfgContent::U32(0x1122_3344);

        assert_eq!(bytes.len().unwrap(), 2);
        assert_eq!(number.len().unwrap(), 4);
    }

    #[test]
    fn selected_item_reads_data_and_zero_fills_after_end() {
        let mut fw_cfg = FwCfgInner::new();
        fw_cfg.configure(&[], 1);

        fw_cfg.select(QEMU_FW_CFG_ITEM_INTERFACE_VERSION);

        let first = fw_cfg.read_port(AccessWidth::Dword);
        let second = fw_cfg.read_port(AccessWidth::Dword);

        assert_eq!(first, 2);
        assert_eq!(second, 0);
    }

    #[test]
    fn file_directory_lists_e820_file() {
        let mut fw_cfg = FwCfgInner::new();
        fw_cfg.configure(&[(0, 0x1000)], 1);

        fw_cfg.select(QEMU_FW_CFG_ITEM_FILE_DIR);
        let dir = fw_cfg.read_bytes(64);

        assert_eq!(u32::from_be_bytes(dir[0..4].try_into().unwrap()), 1);
        assert!(
            dir[8..]
                .windows(b"etc/e820".len())
                .any(|w| w == b"etc/e820")
        );
    }

    #[test]
    fn add_file_item_updates_directory_and_allocates_selectors() {
        let mut fw_cfg = FwCfgInner::new();
        fw_cfg.configure(&[], 1);

        let selector = fw_cfg
            .add_file_item(
                "opt/org.test/custom",
                FwCfgContent::Bytes(b"hello".to_vec()),
            )
            .unwrap();

        assert_eq!(selector, 0x20);
        fw_cfg.select(QEMU_FW_CFG_ITEM_FILE_DIR);
        let dir = fw_cfg.read_bytes(128);
        assert!(
            dir.windows(b"opt/org.test/custom".len())
                .any(|w| w == b"opt/org.test/custom")
        );

        fw_cfg.select(selector);
        assert_eq!(fw_cfg.read_bytes(8)[0..5], *b"hello");
    }

    #[test]
    fn dma_address_write_sets_one_pending_descriptor() {
        let mut fw_cfg = FwCfgInner::new();

        fw_cfg.write_dma_address_part(0x1234, true);
        assert!(fw_cfg.take_pending_dma().is_none());

        fw_cfg.write_dma_address_part(0x5678, false);

        let expected = ((u32::from_be(0x1234) as u64) << 32) | u32::from_be(0x5678) as u64;
        assert_eq!(
            fw_cfg.take_pending_dma().map(|addr| addr.as_usize()),
            Some(expected as usize)
        );
        assert!(fw_cfg.take_pending_dma().is_none());
    }

    #[test]
    fn dma_request_parses_big_endian_descriptor() {
        let mut descriptor = [0u8; 16];
        descriptor[0..4].copy_from_slice(&0x0008_0002u32.to_be_bytes());
        descriptor[4..8].copy_from_slice(&0x0000_1000u32.to_be_bytes());
        descriptor[8..16].copy_from_slice(&0x1234_5678_9abc_def0u64.to_be_bytes());

        let request = FwCfgDmaRequest::parse(descriptor);

        assert_eq!(request.control, 0x0008_0002);
        assert_eq!(request.length, 0x1000);
        assert_eq!(request.address.as_usize(), 0x1234_5678_9abc_def0usize);
    }
}
