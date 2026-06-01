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

use alloc::{boxed::Box, collections::BTreeMap, format, sync::Arc, vec, vec::Vec};
use core::{alloc::Layout, fmt};

use ax_cpumask::CpuMask;
use ax_errno::{AxError, AxResult, ax_err, ax_err_type};
use ax_kspin::SpinNoIrq as Mutex;
use ax_memory_addr::{align_down_4k, align_up_4k};
use axaddrspace::{AddrSpace, MappingFlags};
use axdevice::{AxVmDeviceConfig, AxVmDevices};
use axdevice_base::{AccessWidth, Port};
use axvcpu::{AxVCpu, AxVCpuExitReason};
#[cfg(target_arch = "x86_64")]
use axvm_types::EmulatedDeviceType;
use axvm_types::{GuestPhysAddr, HostPhysAddr, HostVirtAddr};
use spin::Once;
#[cfg(target_arch = "x86_64")]
use x86_64::instructions::port::Port as X86Port;
#[cfg(all(target_arch = "x86_64", feature = "vmx"))]
use x86_vcpu::{X86_APIC_ACCESS_GPA, x86_apic_access_page_addr};

#[cfg(not(target_arch = "x86_64"))]
use crate::vcpu::AxVCpuCreateConfig;
#[cfg(target_arch = "aarch64")]
use crate::vcpu::get_sysreg_device;
use crate::{
    config::{AxVMConfig, PhysCpuList, VMInterruptMode},
    host::paging::{HostPagingHandler, virt_to_phys},
    vcpu::AxArchVCpuImpl,
};

const VM_ASPACE_BASE: usize = 0x0;
const VM_ASPACE_SIZE: usize = 0x7fff_ffff_f000;

#[cfg(target_arch = "x86_64")]
const FW_CFG_IO_SELECTOR: u16 = 0x510;
#[cfg(target_arch = "x86_64")]
const FW_CFG_IO_DATA: u16 = 0x511;
#[cfg(target_arch = "x86_64")]
const FW_CFG_IO_DMA_ADDRESS: u16 = 0x514;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_FNAME_SIZE: usize = 56;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_ITEM_SIGNATURE: u16 = 0x0000;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_ITEM_INTERFACE_VERSION: u16 = 0x0001;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_ITEM_SMP_CPU_COUNT: u16 = 0x0005;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_ITEM_FILE_DIR: u16 = 0x0019;
#[cfg(target_arch = "x86_64")]
const QEMU_FW_CFG_ITEM_ETC_E820: u16 = 0x8000;
#[cfg(target_arch = "x86_64")]
const FW_CFG_F_DMA: u32 = 1 << 1;
#[cfg(target_arch = "x86_64")]
const FW_CFG_DMA_CTL_ERROR: u32 = 1 << 0;
#[cfg(target_arch = "x86_64")]
const FW_CFG_DMA_CTL_READ: u32 = 1 << 1;
#[cfg(target_arch = "x86_64")]
const FW_CFG_DMA_CTL_SKIP: u32 = 1 << 2;
#[cfg(target_arch = "x86_64")]
const FW_CFG_DMA_CTL_SELECT: u32 = 1 << 3;
#[cfg(target_arch = "x86_64")]
const FW_CFG_DMA_CTL_WRITE: u32 = 1 << 4;
#[cfg(target_arch = "x86_64")]
const OVMF_VIRTIO_BLK_IO_BASE: u16 = 0x6000;
#[cfg(target_arch = "x86_64")]
const OVMF_VIRTIO_BLK_IO_SIZE: u16 = 0x80;
#[cfg(target_arch = "x86_64")]
const OVMF_VIRTIO_BLK_QUEUE_PFN: u16 = OVMF_VIRTIO_BLK_IO_BASE + 0x08;
#[cfg(target_arch = "x86_64")]
const OVMF_VIRTIO_BLK_QUEUE_NOTIFY: u16 = OVMF_VIRTIO_BLK_IO_BASE + 0x10;
#[cfg(target_arch = "x86_64")]
const ACPI_PM_IO_BASE: u16 = 0x600;
#[cfg(target_arch = "x86_64")]
const ACPI_PM_IO_SIZE: u16 = 0x10;

/// A vCPU with architecture-independent interface.
type VCpu = AxVCpu<AxArchVCpuImpl>;
/// A reference to a vCPU.
pub type AxVCpuRef = Arc<VCpu>;
/// A reference to a VM.
pub type AxVMRef = Arc<AxVM>;

fn width_mask(width: AccessWidth) -> usize {
    match width {
        AccessWidth::Byte => 0xff,
        AccessWidth::Word => 0xffff,
        AccessWidth::Dword => 0xffff_ffff,
        AccessWidth::Qword => usize::MAX,
    }
}

fn sign_extend_value(value: usize, width: AccessWidth) -> usize {
    match width {
        AccessWidth::Byte => (value as i8) as isize as usize,
        AccessWidth::Word => (value as i16) as isize as usize,
        AccessWidth::Dword => (value as i32) as isize as usize,
        AccessWidth::Qword => value,
    }
}

struct AxVMInnerConst {
    phys_cpu_ls: PhysCpuList,
    vcpu_list: Box<[AxVCpuRef]>,
    devices: AxVmDevices,
}

unsafe impl Send for AxVMInnerConst {}
unsafe impl Sync for AxVMInnerConst {}

/// Represents a memory region in a virtual machine.
#[derive(Debug, Clone)]
pub struct VMMemoryRegion {
    /// Guest physical address.
    pub gpa: GuestPhysAddr,
    /// Host virtual address.
    pub hva: HostVirtAddr,
    /// Memory layout of the region.
    pub layout: Layout,
    /// Whether this region was allocated by the allocator and needs to be deallocated
    pub needs_dealloc: bool,
}

impl VMMemoryRegion {
    /// Returns the size of the memory region.
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the host physical address backing this guest memory region.
    pub fn host_paddr(&self) -> HostPhysAddr {
        virt_to_phys(self.hva)
    }

    /// Returns `true` if the guest physical address is identical to the host physical address.
    pub fn is_identical(&self) -> bool {
        self.gpa.as_usize() == self.host_paddr().as_usize()
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Default)]
struct OvmfVirtioBlkIoState {
    queue_pfn: u32,
    queue_size: u16,
    translated_queue_pfn: u32,
}

struct AxVMInnerMut {
    // Todo: use more efficient lock.
    address_space: AddrSpace<HostPagingHandler>,
    memory_regions: Vec<VMMemoryRegion>,
    config: AxVMConfig,
    vm_status: VMStatus,
    #[cfg(target_arch = "x86_64")]
    fw_cfg: FwCfgState,
    #[cfg(target_arch = "x86_64")]
    ovmf_virtio_blk: OvmfVirtioBlkIoState,
}

#[cfg(target_arch = "x86_64")]
struct FwCfgState {
    selector: u16,
    offset: usize,
    dma_address: u64,
    dma_bytes: [u8; 8],
    dma_bytes_written: usize,
    items: BTreeMap<u16, Vec<u8>>,
}

#[cfg(target_arch = "x86_64")]
impl FwCfgState {
    fn new() -> Self {
        Self {
            selector: 0,
            offset: 0,
            dma_address: 0,
            dma_bytes: [0; 8],
            dma_bytes_written: 0,
            items: BTreeMap::new(),
        }
    }

    fn configure(&mut self, memory_regions: &[VMMemoryRegion], cpu_count: usize) {
        self.items.clear();
        self.items
            .insert(QEMU_FW_CFG_ITEM_SIGNATURE, b"QEMU".to_vec());
        self.items.insert(
            QEMU_FW_CFG_ITEM_INTERFACE_VERSION,
            FW_CFG_F_DMA.to_le_bytes().to_vec(),
        );
        self.items.insert(
            QEMU_FW_CFG_ITEM_SMP_CPU_COUNT,
            (cpu_count as u16).to_le_bytes().to_vec(),
        );
        self.items
            .insert(QEMU_FW_CFG_ITEM_ETC_E820, Self::build_e820(memory_regions));
        self.items
            .insert(QEMU_FW_CFG_ITEM_FILE_DIR, self.build_file_dir());
    }

    fn build_e820(memory_regions: &[VMMemoryRegion]) -> Vec<u8> {
        let mut e820 = Vec::new();
        for region in memory_regions {
            let gpa = region.gpa.as_usize();
            if gpa >= 0x100000000 {
                continue;
            }
            let end = (gpa + region.size()).min(0x100000000);
            append_u64_le(&mut e820, gpa as u64);
            append_u64_le(&mut e820, (end - gpa) as u64);
            append_u32_le(&mut e820, 1);
        }
        e820
    }

    fn build_file_dir(&self) -> Vec<u8> {
        let files = [("etc/e820", QEMU_FW_CFG_ITEM_ETC_E820)];
        let mut dir = Vec::new();
        append_u32_be(&mut dir, files.len() as u32);
        for (name, selector) in files {
            let data = self.items.get(&selector).expect("fw_cfg file item missing");
            append_u32_be(&mut dir, data.len() as u32);
            append_u16_be(&mut dir, selector);
            append_u16_be(&mut dir, 0);
            let mut name_bytes = [0u8; QEMU_FW_CFG_FNAME_SIZE];
            let bytes = name.as_bytes();
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
        let mut out = Vec::with_capacity(size);
        if let Some(item) = self.items.get(&self.selector) {
            let end = (self.offset + size).min(item.len());
            out.extend_from_slice(&item[self.offset..end]);
            self.offset = end;
        }
        out.resize(size, 0);
        out
    }

    fn skip_bytes(&mut self, size: usize) {
        self.offset += size;
        if let Some(item) = self.items.get(&self.selector) {
            self.offset = self.offset.min(item.len());
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn append_u16_be(buffer: &mut Vec<u8>, value: u16) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

#[cfg(target_arch = "x86_64")]
fn append_u32_be(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

#[cfg(target_arch = "x86_64")]
fn append_u32_le(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

#[cfg(target_arch = "x86_64")]
fn append_u64_le(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// VM status enumeration representing the lifecycle states of a virtual machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMStatus {
    /// VM is being created/loaded
    Loading,
    /// VM is loaded but not yet started
    Loaded,
    /// VM is currently running
    Running,
    /// VM is suspended (paused but can be resumed)
    Suspended,
    /// VM is in the process of shutting down
    Stopping,
    /// VM is stopped
    Stopped,
}

impl VMStatus {
    /// Get status as a string (lowercase)
    pub fn as_str(&self) -> &'static str {
        match self {
            VMStatus::Loading => "loading",
            VMStatus::Loaded => "loaded",
            VMStatus::Running => "running",
            VMStatus::Suspended => "suspended",
            VMStatus::Stopping => "stopping",
            VMStatus::Stopped => "stopped",
        }
    }

    /// Get status with emoji icon
    pub fn as_str_with_icon(&self) -> &'static str {
        match self {
            VMStatus::Loading => "🔄 loading",
            VMStatus::Loaded => "📦 loaded",
            VMStatus::Running => "🚀 running",
            VMStatus::Suspended => "🛑 suspended",
            VMStatus::Stopping => "⏹️ stopping",
            VMStatus::Stopped => "💤 stopped",
        }
    }
}

impl fmt::Display for VMStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

const TEMP_MAX_VCPU_NUM: usize = 64;

/// A Virtual Machine.
pub struct AxVM {
    id: usize,
    inner_const: Once<AxVMInnerConst>,
    inner_mut: Mutex<AxVMInnerMut>,
    debugcon_buf: Mutex<Vec<u8>>,
}

impl AxVM {
    /// Creates a new VM with the given configuration.
    /// Returns an error if the configuration is invalid.
    /// The VM is not started until `boot` is called.
    pub fn new(config: AxVMConfig) -> AxResult<AxVMRef> {
        let address_space = AddrSpace::new_empty(
            crate::vcpu::max_guest_page_table_levels(),
            GuestPhysAddr::from(VM_ASPACE_BASE),
            VM_ASPACE_SIZE,
        )?;

        let result = Arc::new(Self {
            id: config.id(),
            inner_const: Once::new(),
            inner_mut: Mutex::new(AxVMInnerMut {
                address_space,
                config,
                memory_regions: Vec::new(),
                vm_status: VMStatus::Loading,
                #[cfg(target_arch = "x86_64")]
                fw_cfg: FwCfgState::new(),
                #[cfg(target_arch = "x86_64")]
                ovmf_virtio_blk: OvmfVirtioBlkIoState::default(),
            }),
            debugcon_buf: Mutex::new(Vec::new()),
        });

        info!("VM created: id={}", result.id());

        Ok(result)
    }

    /// Returns the VM id.
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the configured VM interrupt mode.
    pub fn interrupt_mode(&self) -> VMInterruptMode {
        self.inner_mut.lock().config.interrupt_mode()
    }

    /// Sets up the VM before booting.
    pub fn init(&self) -> AxResult {
        let mut inner_mut = self.inner_mut.lock();

        let dtb_addr = inner_mut.config.image_config().dtb_load_gpa;
        let vcpu_id_pcpu_sets = inner_mut.config.phys_cpu_ls.get_vcpu_affinities_pcpu_ids();
        let fw_cfg_cpu_count = vcpu_id_pcpu_sets.len();

        info!("dtb_load_gpa: {dtb_addr:?}");
        debug!("id: {}, VCpuIdPCpuSets: {vcpu_id_pcpu_sets:#x?}", self.id());

        let mut vcpu_list = Vec::with_capacity(vcpu_id_pcpu_sets.len());
        for (vcpu_id, phys_cpu_set, _pcpu_id) in vcpu_id_pcpu_sets {
            #[cfg(target_arch = "aarch64")]
            let arch_config = AxVCpuCreateConfig {
                mpidr_el1: _pcpu_id as _,
                dtb_addr: dtb_addr.unwrap_or_default().as_usize(),
            };
            #[cfg(target_arch = "riscv64")]
            let arch_config = AxVCpuCreateConfig {
                hart_id: vcpu_id as _,
                dtb_addr: dtb_addr.unwrap_or_default().as_usize(),
            };
            #[cfg(target_arch = "loongarch64")]
            let arch_config = AxVCpuCreateConfig {
                cpu_id: vcpu_id,
                dtb_addr: dtb_addr.unwrap_or_default().as_usize(),
            };

            // FIXME: VCpu is neither `Send` nor `Sync` by design, check whether
            // 1. we should make it `Send` and `Sync`, or
            // 2. we can guarantee that no cross-thread access is performed
            #[allow(clippy::arc_with_non_send_sync)]
            vcpu_list.push(Arc::new(VCpu::new(
                self.id(),
                vcpu_id,
                0, // Currently not used.
                phys_cpu_set,
                #[cfg(target_arch = "aarch64")]
                arch_config,
                #[cfg(target_arch = "loongarch64")]
                arch_config,
                #[cfg(target_arch = "riscv64")]
                arch_config,
                #[cfg(target_arch = "x86_64")]
                (),
            )?));
        }

        #[cfg(target_arch = "x86_64")]
        {
            let memory_regions = inner_mut.memory_regions.clone();
            inner_mut
                .fw_cfg
                .configure(&memory_regions, fw_cfg_cpu_count);
        }

        #[cfg(target_arch = "x86_64")]
        {
            inner_mut.address_space.map_linear(
                GuestPhysAddr::from(0xfee0_0000),
                crate::vcpu::EmulatedLocalApic::virtual_apic_access_addr(),
                0x1000,
                MappingFlags::DEVICE
                    | MappingFlags::READ
                    | MappingFlags::WRITE
                    | MappingFlags::USER,
            )?;
        }

        let mut pt_dev_region = Vec::new();
        for pt_device in inner_mut.config.pass_through_devices() {
            trace!(
                "PT dev {:?} region: [{:#x}~{:#x}] -> [{:#x}~{:#x}]",
                pt_device.name,
                pt_device.base_gpa,
                pt_device.base_gpa + pt_device.length,
                pt_device.base_hpa,
                pt_device.base_hpa + pt_device.length
            );
            // Align the base address and length to 4K boundaries.
            pt_dev_region.push((
                align_down_4k(pt_device.base_gpa),
                align_up_4k(pt_device.length),
            ));
        }

        for pt_addr in inner_mut.config.pass_through_addresses() {
            debug!(
                "PT addr region: [{:#x}~{:#x}]",
                pt_addr.base_gpa,
                pt_addr.base_gpa + pt_addr.length,
            );
            // Align the base address and length to 4K boundaries.
            pt_dev_region.push((align_down_4k(pt_addr.base_gpa), align_up_4k(pt_addr.length)));
        }

        pt_dev_region.sort_by_key(|(gpa, _)| *gpa);

        // Merge overlapping regions.
        let pt_dev_region =
            pt_dev_region
                .into_iter()
                .fold(Vec::<(usize, usize)>::new(), |mut acc, (gpa, len)| {
                    if let Some(last) = acc.last_mut() {
                        if last.0 + last.1 >= gpa {
                            // Merge with the last region.
                            last.1 = (last.0 + last.1).max(gpa + len) - last.0;
                        } else {
                            acc.push((gpa, len));
                        }
                    } else {
                        acc.push((gpa, len));
                    }
                    acc
                });

        for (gpa, len) in &pt_dev_region {
            inner_mut.address_space.map_linear(
                GuestPhysAddr::from(*gpa),
                HostPhysAddr::from(*gpa),
                *len,
                MappingFlags::DEVICE
                    | MappingFlags::READ
                    | MappingFlags::WRITE
                    | MappingFlags::USER,
            )?;
        }

        #[cfg(all(target_arch = "x86_64", feature = "vmx"))]
        inner_mut.address_space.map_linear(
            GuestPhysAddr::from(X86_APIC_ACCESS_GPA),
            x86_apic_access_page_addr(),
            ax_memory_addr::PAGE_SIZE_4K,
            MappingFlags::DEVICE | MappingFlags::READ | MappingFlags::WRITE,
        )?;

        #[cfg_attr(not(target_arch = "aarch64"), expect(unused_mut))]
        let mut devices = axdevice::AxVmDevices::new(AxVmDeviceConfig {
            emu_configs: inner_mut.config.emu_devices().to_vec(),
        });

        #[cfg(target_arch = "aarch64")]
        {
            let passthrough = inner_mut.config.interrupt_mode() == VMInterruptMode::Passthrough;
            if passthrough {
                let spis = inner_mut.config.pass_through_spis();
                let cpu_id = self.id() - 1; // FIXME: get the real CPU id.
                let mut gicd_found = false;

                for device in devices.iter_mmio_dev() {
                    if let Some(result) = axdevice_base::map_device_of_type(
                        device,
                        |gicd: &arm_vgic::v3::vgicd::VGicD| {
                            debug!("VGicD found, assigning SPIs...");

                            for spi in spis {
                                gicd.assign_irq(*spi + 32, cpu_id, (0, 0, 0, cpu_id as _))
                            }

                            AxResult::Ok(())
                        },
                    ) {
                        result?;
                        gicd_found = true;
                        break;
                    }
                }

                if !gicd_found {
                    warn!("Failed to assign SPIs: No VGicD found in device list");
                }
            } else {
                // non-passthrough mode, we need to set up the virtual timer.
                //
                // FIXME: maybe let `axdevice` handle this automatically?
                // how to let `axdevice` know whether the VM is in passthrough mode or not?
                for dev in get_sysreg_device() {
                    devices.add_sys_reg_dev(dev);
                }
            }
        }

        self.inner_const.call_once(|| AxVMInnerConst {
            phys_cpu_ls: inner_mut.config.phys_cpu_ls.clone(),
            vcpu_list: vcpu_list.into_boxed_slice(),
            devices,
        });

        // Setup VCpus.
        for vcpu in self.vcpu_list() {
            #[cfg(target_arch = "aarch64")]
            let setup_config = {
                let passthrough = inner_mut.config.interrupt_mode() == VMInterruptMode::Passthrough;
                crate::vcpu::AxVCpuSetupConfig {
                    passthrough_interrupt: passthrough,
                    passthrough_timer: passthrough,
                }
            };
            #[cfg(target_arch = "loongarch64")]
            let setup_config = {
                let passthrough = inner_mut.config.interrupt_mode() == VMInterruptMode::Passthrough;
                crate::vcpu::AxVCpuSetupConfig {
                    passthrough_interrupt: passthrough,
                    passthrough_timer: passthrough,
                }
            };
            #[cfg(not(any(
                target_arch = "aarch64",
                target_arch = "loongarch64",
                target_arch = "x86_64"
            )))]
            #[allow(clippy::let_unit_value)]
            let setup_config = <AxArchVCpuImpl as axvcpu::AxArchVCpu>::SetupConfig::default();
            #[cfg(target_arch = "x86_64")]
            let setup_config = crate::vcpu::AxVCpuSetupConfig {
                emulate_com1: inner_mut
                    .config
                    .emu_devices()
                    .iter()
                    .any(|dev| dev.emu_type == EmulatedDeviceType::Console),
            };

            let entry = if vcpu.id() == 0 {
                inner_mut.config.bsp_entry()
            } else {
                inner_mut.config.ap_entry()
            };

            debug!("Setting up vCPU[{}] entry at {:#x}", vcpu.id(), entry);

            vcpu.setup(
                entry,
                inner_mut.address_space.page_table_root(),
                setup_config,
            )?;
        }
        info!("VM setup: id={}", self.id());
        Ok(())
    }

    /// Sets the VM status.
    pub fn set_vm_status(&self, status: VMStatus) {
        let mut inner_mut = self.inner_mut.lock();
        inner_mut.vm_status = status;
    }

    /// Returns the current VM status.
    pub fn vm_status(&self) -> VMStatus {
        let inner_mut = self.inner_mut.lock();
        inner_mut.vm_status
    }

    /// Retrieves the vCPU corresponding to the given vcpu_id for the VM.
    /// Returns None if the vCPU does not exist.
    #[inline]
    pub fn vcpu(&self, vcpu_id: usize) -> Option<AxVCpuRef> {
        self.vcpu_list().get(vcpu_id).cloned()
    }

    /// Returns the number of vCPUs corresponding to the VM.
    #[inline]
    pub fn vcpu_num(&self) -> usize {
        self.inner_const().vcpu_list.len()
    }

    fn inner_const(&self) -> &AxVMInnerConst {
        self.inner_const
            .get()
            .expect("VM inner_const not initialized")
    }

    /// Returns a reference to the list of vCPUs corresponding to the VM.
    #[inline]
    pub fn vcpu_list(&self) -> &[AxVCpuRef] {
        &self.inner_const().vcpu_list
    }

    /// Returns the base address of the two-stage address translation page table for the VM.
    pub fn ept_root(&self) -> HostPhysAddr {
        self.inner_mut.lock().address_space.page_table_root()
    }

    /// Returns to the VM's configuration.
    pub fn with_config<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut AxVMConfig) -> R,
    {
        let mut g = self.inner_mut.lock();
        f(&mut g.config)
    }

    /// Returns guest VM image load region in `Vec<&'static mut [u8]>`,
    /// according to the given `image_load_gpa` and `image_size.
    /// `Vec<&'static mut [u8]>` is a series of (HVA) address segments,
    /// which may correspond to non-contiguous physical addresses,
    ///
    /// FIXME:
    /// Find a more elegant way to manage potentially non-contiguous physical memory
    ///         instead of `Vec<&'static mut [u8]>`.
    pub fn get_image_load_region(
        &self,
        image_load_gpa: GuestPhysAddr,
        image_size: usize,
    ) -> AxResult<Vec<&'static mut [u8]>> {
        let g = self.inner_mut.lock();
        let image_load_hva = g
            .address_space
            .translated_byte_buffer(image_load_gpa, image_size)
            .expect("Failed to translate kernel image load address");
        Ok(image_load_hva)
    }

    /// Boots the VM by transitioning to Running state.
    pub fn boot(&self) -> AxResult {
        if self.running() {
            ax_err!(BadState, format!("VM[{}] is already running", self.id()))
        } else {
            info!("Booting VM[{}]", self.id());
            self.set_vm_status(VMStatus::Running);
            Ok(())
        }
    }

    /// Returns if the VM is running.
    pub fn running(&self) -> bool {
        self.vm_status() == VMStatus::Running
    }

    /// Returns if the VM is shutting down (in Stopping state).
    pub fn stopping(&self) -> bool {
        self.vm_status() == VMStatus::Stopping
    }

    /// Returns if the VM is suspended.
    pub fn suspending(&self) -> bool {
        self.vm_status() == VMStatus::Suspended
    }

    /// Returns if the VM is stopped.
    pub fn stopped(&self) -> bool {
        self.vm_status() == VMStatus::Stopped
    }

    /// Shuts down the VM by transitioning to Stopping state.
    ///
    /// This method sets the VM status to Stopping, which signals all vCPUs to exit.
    /// Currently, the "re-init" process of the VM is not implemented. Therefore, a VM can only be
    /// booted once. And after the VM is shut down, it cannot be booted again.
    pub fn shutdown(&self) -> AxResult {
        if self.stopping() {
            ax_err!(BadState, format!("VM[{}] is already stopping", self.id()))
        } else if self.stopped() {
            ax_err!(BadState, format!("VM[{}] is already stopped", self.id()))
        } else {
            info!("Shutting down VM[{}]", self.id());
            self.set_vm_status(VMStatus::Stopping);
            Ok(())
        }
    }

    // TODO: implement suspend/resume.
    // TODO: implement re-init.

    /// Returns this VM's emulated devices.
    pub fn get_devices(&self) -> &AxVmDevices {
        &self.inner_const().devices
    }

    /// Run a vCPU according to the given vcpu_id.
    ///
    /// ## Arguments
    /// * `vcpu_id` - the id of the vCPU to run.
    ///
    /// ## Returns
    /// * `AxVCpuExitReason` - the exit reason of the vCPU, wrapped in an `AxResult`.
    pub fn run_vcpu(&self, vcpu_id: usize) -> AxResult<AxVCpuExitReason> {
        let vcpu = self
            .vcpu(vcpu_id)
            .ok_or_else(|| ax_err_type!(InvalidInput, "Invalid vcpu_id"))?;

        vcpu.bind()?;

        let exit_reason = vcpu.with_current_cpu_set(|| -> AxResult<AxVCpuExitReason> {
            loop {
                crate::runtime::vcpus::inject_pending_interrupts(self.id(), vcpu_id, &vcpu);

                let exit_reason = vcpu.run()?;
                trace!("{exit_reason:#x?}");
                match exit_reason {
                    AxVCpuExitReason::MmioRead {
                        addr,
                        width,
                        reg,
                        reg_width,
                        signed_ext,
                    } => {
                        let raw = self.get_devices().handle_mmio_read(addr, width)?;
                        let masked = raw & width_mask(width);
                        let val = if signed_ext {
                            sign_extend_value(masked, width)
                        } else {
                            masked & width_mask(reg_width)
                        };
                        vcpu.set_gpr(reg, val);
                    }
                    AxVCpuExitReason::MmioWrite { addr, width, data } => {
                        self.get_devices()
                            .handle_mmio_write(addr, width, data as usize)?;
                    }
                    AxVCpuExitReason::IoRead { port, width } => {
                        let val = {
                            #[cfg(target_arch = "x86_64")]
                            if port.number() == 0x402 {
                                0xE9 // OVMF debugcon 默认读回标志
                            } else if let Some(val) = self.handle_fw_cfg_io_read(port, width)? {
                                val
                            } else if let Some(val) =
                                self.handle_ovmf_virtio_blk_io_read(port, width)?
                            {
                                val
                            } else if let Some(val) = self.handle_acpi_pm_io_read(port, width)? {
                                val
                            } else {
                                self.get_devices().handle_port_read(port, width)?
                            }

                            #[cfg(not(target_arch = "x86_64"))]
                            self.get_devices().handle_port_read(port, width)?
                        };

                        #[cfg(not(target_arch = "riscv64"))]
                        vcpu.set_gpr(0, val);

                        #[cfg(target_arch = "riscv64")]
                        vcpu.set_gpr(riscv_vcpu::GprIndex::A0 as usize, val);
                    }
                    AxVCpuExitReason::IoWrite { port, width, data } => {
                        #[cfg(target_arch = "x86_64")]
                        if port.number() == 0x402 {
                            self.debugcon_write_bytes(&[data as u8]);
                        } else if self.handle_fw_cfg_io_write(port, width, data as usize)? {
                            // 成功由 fw_cfg 处理
                        } else if self.handle_ovmf_virtio_blk_io_write(
                            port,
                            width,
                            data as usize,
                        )? {
                            // 成功由 ovmf_virtio_blk 处理
                        } else if self.handle_acpi_pm_io_write(port, width, data as usize)? {
                            // 成功由 acpi_pm 处理
                        } else {
                            self.get_devices()
                                .handle_port_write(port, width, data as usize)?;
                        }

                        #[cfg(not(target_arch = "x86_64"))]
                        self.get_devices()
                            .handle_port_write(port, width, data as usize)?;
                    }
                    AxVCpuExitReason::IoStringRead {
                        port,
                        width: _,
                        dst_gpa,
                        count,
                    } => {
                        #[cfg(target_arch = "x86_64")]
                        if port.number() == 0x402 {
                            let fill = alloc::vec![0xE9u8; count];
                            self.write_guest_bytes(dst_gpa, &fill)?;
                        } else {
                            warn!("Unhandled string I/O read from port {:#x}", port.number());
                        }

                        #[cfg(not(target_arch = "x86_64"))]
                        warn!("Unhandled string I/O read from port {:#x}", port.number());
                    }
                    AxVCpuExitReason::IoStringWrite {
                        port,
                        width: _,
                        src_gpa,
                        count,
                    } => {
                        #[cfg(target_arch = "x86_64")]
                        if port.number() == 0x402 {
                            let data = self.read_guest_bytes(src_gpa, count)?;
                            self.debugcon_write_bytes(&data);
                        } else {
                            warn!("Unhandled string I/O write to port {:#x}", port.number());
                        }

                        #[cfg(not(target_arch = "x86_64"))]
                        warn!("Unhandled string I/O write to port {:#x}", port.number());
                    }
                    AxVCpuExitReason::SysRegRead { addr, reg } => {
                        let val = self
                            .get_devices()
                            .handle_sys_reg_read(addr, AccessWidth::Qword)?;
                        vcpu.set_gpr(reg, val);
                    }
                    AxVCpuExitReason::SysRegWrite { addr, value } => {
                        self.get_devices().handle_sys_reg_write(
                            addr,
                            AccessWidth::Qword,
                            value as usize,
                        )?;
                    }
                    AxVCpuExitReason::NestedPageFault { addr, access_flags } => {
                        if !self.handle_nested_page_fault(addr, access_flags) {
                            break Ok(AxVCpuExitReason::NestedPageFault { addr, access_flags });
                        }
                    }
                    exit_reason => break Ok(exit_reason),
                }
            }
        })?;

        vcpu.unbind()?;
        Ok(exit_reason)
    }

    fn handle_nested_page_fault(&self, addr: GuestPhysAddr, access_flags: MappingFlags) -> bool {
        let mut guard = self.inner_mut.lock();
        let handled = guard.address_space.handle_page_fault(addr, access_flags);
        Self::debug_nested_page_fault(self.id(), &guard, addr, access_flags, handled);
        handled
    }

    fn debug_nested_page_fault(
        vm_id: usize,
        inner: &AxVMInnerMut,
        addr: GuestPhysAddr,
        access_flags: MappingFlags,
        handled: bool,
    ) {
        let root = inner.address_space.page_table_root();
        match inner.address_space.page_table().query(addr) {
            Ok((hpa, flags, size)) => {
                if handled {
                    debug!(
                        "VM[{}] stage2 query hit: gpa={:#x} -> hpa={:#x}, access={:?}, \
                         pte_flags={:?}, page_size={:?}, root={:#x}",
                        vm_id,
                        addr.as_usize(),
                        hpa.as_usize(),
                        access_flags,
                        flags,
                        size,
                        root.as_usize()
                    );
                } else {
                    warn!(
                        "VM[{}] stage2 query hit: gpa={:#x} -> hpa={:#x}, access={:?}, \
                         pte_flags={:?}, page_size={:?}, root={:#x}",
                        vm_id,
                        addr.as_usize(),
                        hpa.as_usize(),
                        access_flags,
                        flags,
                        size,
                        root.as_usize()
                    );
                }
            }
            Err(err) => {
                if handled {
                    debug!(
                        "VM[{}] stage2 query miss: gpa={:#x}, access={:?}, err={:?}, root={:#x}",
                        vm_id,
                        addr.as_usize(),
                        access_flags,
                        err,
                        root.as_usize()
                    );
                } else {
                    warn!(
                        "VM[{}] stage2 query miss: gpa={:#x}, access={:?}, err={:?}, root={:#x}",
                        vm_id,
                        addr.as_usize(),
                        access_flags,
                        err,
                        root.as_usize()
                    );
                }
            }
        }

        let translate = inner.address_space.translate(addr);
        if handled {
            debug!(
                "VM[{}] stage2 translate: gpa={:#x} -> {:?}",
                vm_id,
                addr.as_usize(),
                translate
            );
        } else {
            warn!(
                "VM[{}] stage2 translate: gpa={:#x} -> {:?}",
                vm_id,
                addr.as_usize(),
                translate
            );
        }

        for (idx, region) in inner.memory_regions.iter().enumerate() {
            let start = region.gpa.as_usize();
            let end = start + region.size();
            if (start..end).contains(&addr.as_usize()) {
                if handled {
                    debug!(
                        "VM[{}] stage2 region hit[{}]: gpa=[{:#x},{:#x}) hva={:#x} hpa={:#x} \
                         size={:#x} identical={}",
                        vm_id,
                        idx,
                        start,
                        end,
                        region.hva.as_usize(),
                        region.host_paddr().as_usize(),
                        region.size(),
                        region.is_identical()
                    );
                } else {
                    warn!(
                        "VM[{}] stage2 region hit[{}]: gpa=[{:#x},{:#x}) hva={:#x} hpa={:#x} \
                         size={:#x} identical={}",
                        vm_id,
                        idx,
                        start,
                        end,
                        region.hva.as_usize(),
                        region.host_paddr().as_usize(),
                        region.size(),
                        region.is_identical()
                    );
                }
            }
        }
    }

    /// Injects an interrupt to the vCPU.
    pub fn inject_interrupt_to_vcpu(
        &self,
        targets: CpuMask<TEMP_MAX_VCPU_NUM>,
        irq: usize,
    ) -> AxResult {
        for vcpu in self.vcpu_list() {
            if targets.get(vcpu.id()) {
                crate::runtime::vcpus::queue_interrupt(self.id(), vcpu.id(), irq)?;
            }
        }
        Ok(())
    }

    /// Returns vCpu id list and its corresponding pCpu affinity list, as well as its physical id.
    /// If the pCpu affinity is None, it means the vCpu will be allocated to any available pCpu randomly.
    /// if the pCPU id is not provided, the vCpu's physical id will be set as vCpu id.
    ///
    /// Returns a vector of tuples, each tuple contains:
    /// - The vCpu id.
    /// - The pCpu affinity mask, `None` if not set.
    /// - The physical id of the vCpu, equal to vCpu id if not provided.
    pub fn get_vcpu_affinities_pcpu_ids(&self) -> Vec<(usize, Option<usize>, usize)> {
        self.inner_const()
            .phys_cpu_ls
            .get_vcpu_affinities_pcpu_ids()
    }

    // /// Returns a reference to the VM's configuration.
    // pub fn config(&self) -> &AxVMConfig {
    //     &self.inner_const.config
    // }

    /// Maps a region of host physical memory to guest physical memory.
    pub fn map_region(
        &self,
        gpa: GuestPhysAddr,
        hpa: HostPhysAddr,
        size: usize,
        flags: MappingFlags,
    ) -> AxResult {
        self.inner_mut
            .lock()
            .address_space
            .map_linear(gpa, hpa, size, flags)?;
        Ok(())
    }

    /// Unmaps a region of guest physical memory.
    pub fn unmap_region(&self, gpa: GuestPhysAddr, size: usize) -> AxResult {
        self.inner_mut.lock().address_space.unmap(gpa, size)?;
        Ok(())
    }

    /// Reads an object of type `T` from the guest physical address.
    pub fn read_from_guest_of<T>(&self, gpa_ptr: GuestPhysAddr) -> AxResult<T> {
        let size = core::mem::size_of::<T>();

        // Ensure the address is properly aligned for the type.
        if !gpa_ptr
            .as_usize()
            .is_multiple_of(core::mem::align_of::<T>())
        {
            return ax_err!(InvalidInput, "Unaligned guest physical address");
        }

        let g = self.inner_mut.lock();
        match g.address_space.translated_byte_buffer(gpa_ptr, size) {
            Some(buffers) => {
                let mut data_bytes = Vec::with_capacity(size);
                for chunk in buffers {
                    let remaining = size - data_bytes.len();
                    let chunk_size = remaining.min(chunk.len());
                    data_bytes.extend_from_slice(&chunk[..chunk_size]);
                    if data_bytes.len() >= size {
                        break;
                    }
                }
                if data_bytes.len() < size {
                    return ax_err!(
                        InvalidInput,
                        "Insufficient data in guest memory to read the requested object"
                    );
                }
                let data: T = unsafe {
                    // Use `ptr::read_unaligned` for safety in case of unaligned memory.
                    core::ptr::read_unaligned(data_bytes.as_ptr() as *const T)
                };
                Ok(data)
            }
            None => ax_err!(
                InvalidInput,
                "Failed to translate guest physical address or insufficient buffer size"
            ),
        }
    }

    /// Writes an object of type `T` to the guest physical address.
    pub fn write_to_guest_of<T>(&self, gpa_ptr: GuestPhysAddr, data: &T) -> AxResult {
        match self
            .inner_mut
            .lock()
            .address_space
            .translated_byte_buffer(gpa_ptr, core::mem::size_of::<T>())
        {
            Some(mut buffer) => {
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        data as *const T as *const u8,
                        core::mem::size_of::<T>(),
                    )
                };
                let mut copied_bytes = 0;
                for chunk in buffer.iter_mut() {
                    let end = copied_bytes + chunk.len();
                    chunk.copy_from_slice(&bytes[copied_bytes..end]);
                    copied_bytes += chunk.len();
                }
                Ok(())
            }
            None => ax_err!(InvalidInput, "Failed to translate guest physical address"),
        }
    }

    /// Reads raw bytes from guest physical memory.
    fn read_guest_bytes(&self, gpa: GuestPhysAddr, len: usize) -> AxResult<Vec<u8>> {
        let g = self.inner_mut.lock();
        match g.address_space.translated_byte_buffer(gpa, len) {
            Some(buffers) => {
                let mut data = Vec::with_capacity(len);
                for chunk in buffers {
                    let remaining = len - data.len();
                    let chunk_size = remaining.min(chunk.len());
                    data.extend_from_slice(&chunk[..chunk_size]);
                    if data.len() >= len {
                        break;
                    }
                }
                Ok(data)
            }
            None => ax_err!(InvalidInput, "Failed to translate guest physical address"),
        }
    }

    /// Writes raw bytes to guest physical memory.
    fn write_guest_bytes(&self, gpa: GuestPhysAddr, data: &[u8]) -> AxResult {
        let g = self.inner_mut.lock();
        match g.address_space.translated_byte_buffer(gpa, data.len()) {
            Some(mut buffers) => {
                let mut offset = 0;
                for chunk in buffers.iter_mut() {
                    let end = (offset + chunk.len()).min(data.len());
                    let copy_len = end - offset;
                    chunk[..copy_len].copy_from_slice(&data[offset..end]);
                    offset = end;
                }
                Ok(())
            }
            None => ax_err!(InvalidInput, "Failed to translate guest physical address"),
        }
    }

    /// Appends bytes to the debugcon buffer, flushing complete lines.
    fn debugcon_write_bytes(&self, data: &[u8]) {
        let mut buf = self.debugcon_buf.lock();
        for &byte in data {
            if byte == b'\n' {
                if let Ok(line) = core::str::from_utf8(&buf) {
                    info!("OVMF debugcon: {}", line);
                } else {
                    info!("OVMF debugcon: {:?}", buf);
                }
                buf.clear();
            } else {
                buf.push(byte);
            }
        }
    }

    /// Allocates an IVC channel for inter-VM communication region.
    ///
    /// ## Arguments
    /// * `expected_size` - The expected size of the IVC channel in bytes.
    /// ## Returns
    /// * `AxResult<(GuestPhysAddr, usize)>` - A tuple containing the guest physical address of the allocated IVC channel and its actual size.
    pub fn alloc_ivc_channel(&self, expected_size: usize) -> AxResult<(GuestPhysAddr, usize)> {
        // Ensure the expected size is aligned to 4K.
        let size = align_up_4k(expected_size);
        let gpa = self.inner_const().devices.alloc_ivc_channel(size)?;
        Ok((gpa, size))
    }

    /// Releases an IVC channel for inter-VM communication region.
    /// ## Arguments
    /// * `gpa` - The guest physical address of the IVC channel to release.
    /// * `size` - The size of the IVC channel in bytes.
    /// ## Returns
    /// * `AxResult<()>` - An empty result indicating success or failure.
    pub fn release_ivc_channel(&self, gpa: GuestPhysAddr, size: usize) -> AxResult {
        self.inner_const().devices.release_ivc_channel(gpa, size)?;
        Ok(())
    }

    /// Allocates a new memory region for the VM.
    pub fn alloc_memory_region(
        &self,
        layout: Layout,
        gpa: Option<GuestPhysAddr>,
    ) -> AxResult<&[u8]> {
        assert!(
            layout.size() > 0,
            "Cannot allocate zero-sized memory region"
        );

        let hva = unsafe { alloc::alloc::alloc_zeroed(layout) };
        if hva.is_null() {
            return Err(AxError::NoMemory);
        }
        let s = unsafe { core::slice::from_raw_parts_mut(hva, layout.size()) };
        let hva = HostVirtAddr::from_mut_ptr_of(hva);

        let hpa = virt_to_phys(hva);

        let gpa = gpa.unwrap_or_else(|| hpa.as_usize().into());

        let mut g = self.inner_mut.lock();
        g.address_space.map_linear(
            gpa,
            hpa,
            layout.size(),
            MappingFlags::READ | MappingFlags::WRITE | MappingFlags::EXECUTE | MappingFlags::USER,
        )?;
        g.memory_regions.push(VMMemoryRegion {
            gpa,
            hva,
            layout,
            needs_dealloc: true, // This region was allocated and needs to be freed
        });

        Ok(s)
    }

    /// Returns a list of all memory regions in the VM.
    pub fn memory_regions(&self) -> Vec<VMMemoryRegion> {
        self.inner_mut.lock().memory_regions.clone()
    }

    /// Maps a reserved memory region for the VM.
    pub fn map_reserved_memory_region(
        &self,
        layout: Layout,
        gpa: Option<GuestPhysAddr>,
    ) -> AxResult<&[u8]> {
        assert!(
            layout.size() > 0,
            "Cannot allocate zero-sized memory region"
        );
        let mut g = self.inner_mut.lock();
        g.address_space.map_linear(
            gpa.unwrap(),
            gpa.unwrap().as_usize().into(),
            layout.size(),
            MappingFlags::READ | MappingFlags::WRITE | MappingFlags::EXECUTE | MappingFlags::USER,
        )?;
        let hva = gpa.unwrap().as_usize().into();
        let tem_hva = gpa.unwrap().as_usize() as *mut u8;
        let s = unsafe { core::slice::from_raw_parts_mut(tem_hva, layout.size()) };
        let gpa = gpa.unwrap();
        g.memory_regions.push(VMMemoryRegion {
            gpa,
            hva,
            layout,
            needs_dealloc: false, // This is a reserved region, not allocated
        });
        Ok(s)
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_ovmf_virtio_blk_io_read(
        &self,
        port: Port,
        width: AccessWidth,
    ) -> AxResult<Option<usize>> {
        if !(OVMF_VIRTIO_BLK_IO_BASE..OVMF_VIRTIO_BLK_IO_BASE + OVMF_VIRTIO_BLK_IO_SIZE)
            .contains(&port.number())
        {
            return Ok(None);
        }

        let value = unsafe {
            match width {
                AccessWidth::Byte => X86Port::<u8>::new(port.number()).read() as usize,
                AccessWidth::Word => X86Port::<u16>::new(port.number()).read() as usize,
                AccessWidth::Dword => X86Port::<u32>::new(port.number()).read() as usize,
                AccessWidth::Qword => {
                    return ax_err!(InvalidInput, "unsupported qword virtio-blk I/O read");
                }
            }
        };

        if port.number() == OVMF_VIRTIO_BLK_IO_BASE + 0x0c && width == AccessWidth::Word {
            self.inner_mut.lock().ovmf_virtio_blk.queue_size = value as u16;
        }

        info!(
            "[OVMF-VIRTIO-BLK-IO] in port={:#x} width={width:?} value={value:#x}",
            port.number()
        );
        Ok(Some(value))
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_ovmf_virtio_blk_io_write(
        &self,
        port: Port,
        width: AccessWidth,
        val: usize,
    ) -> AxResult<bool> {
        if !(OVMF_VIRTIO_BLK_IO_BASE..OVMF_VIRTIO_BLK_IO_BASE + OVMF_VIRTIO_BLK_IO_SIZE)
            .contains(&port.number())
        {
            return Ok(false);
        }

        let mut forwarded = val;
        if port.number() == OVMF_VIRTIO_BLK_QUEUE_PFN && width == AccessWidth::Dword {
            if let Some(translated_pfn) = self.translate_ovmf_virtio_blk_queue_pfn(val as u32) {
                forwarded = translated_pfn as usize;
            }
        }

        if port.number() == OVMF_VIRTIO_BLK_QUEUE_NOTIFY && width == AccessWidth::Word {
            self.rewrite_ovmf_virtio_blk_descriptors()?;
            self.dump_ovmf_virtio_blk_queue("before-notify")?;
        }

        unsafe {
            match width {
                AccessWidth::Byte => X86Port::<u8>::new(port.number()).write(forwarded as u8),
                AccessWidth::Word => X86Port::<u16>::new(port.number()).write(forwarded as u16),
                AccessWidth::Dword => X86Port::<u32>::new(port.number()).write(forwarded as u32),
                AccessWidth::Qword => {
                    return ax_err!(InvalidInput, "unsupported qword virtio-blk I/O write");
                }
            }
        }

        info!(
            "[OVMF-VIRTIO-BLK-IO] out port={:#x} width={width:?} value={val:#x} \
             forwarded={forwarded:#x}",
            port.number()
        );
        Ok(true)
    }

    #[cfg(target_arch = "x86_64")]
    fn translate_ovmf_virtio_blk_queue_pfn(&self, queue_pfn: u32) -> Option<u32> {
        let queue_gpa = GuestPhysAddr::from((queue_pfn as usize) << 12);
        let mut g = self.inner_mut.lock();
        let (queue_hpa, limit) = g.address_space.translate_and_get_limit(queue_gpa)?;
        let translated_pfn = (queue_hpa.as_usize() >> 12) as u32;
        g.ovmf_virtio_blk.queue_pfn = queue_pfn;
        g.ovmf_virtio_blk.translated_queue_pfn = translated_pfn;
        info!(
            "[OVMF-VIRTIO-BLK] queue_pfn={queue_pfn:#x} queue_gpa={queue_gpa:?} \
             queue_hpa={queue_hpa:?} limit={limit:#x} translated_pfn={translated_pfn:#x}"
        );
        Some(translated_pfn)
    }

    #[cfg(target_arch = "x86_64")]
    fn rewrite_ovmf_virtio_blk_descriptors(&self) -> AxResult {
        let (queue_pfn, queue_size) = {
            let g = self.inner_mut.lock();
            (g.ovmf_virtio_blk.queue_pfn, g.ovmf_virtio_blk.queue_size)
        };
        if queue_pfn == 0 || queue_size == 0 {
            return Ok(());
        }

        let queue_gpa = GuestPhysAddr::from((queue_pfn as usize) << 12);
        let avail_gpa = GuestPhysAddr::from(queue_gpa.as_usize() + queue_size as usize * 16);
        let mut avail = [0u8; 6];
        {
            let g = self.inner_mut.lock();
            Self::read_guest_bytes_locked(&g.address_space, avail_gpa, &mut avail)?;
        }
        let avail_idx = u16::from_le_bytes(avail[2..4].try_into().unwrap());
        let ring_slot = avail_idx.wrapping_sub(1) as usize % queue_size as usize;
        let ring_gpa = GuestPhysAddr::from(avail_gpa.as_usize() + 4 + ring_slot * 2);
        let mut head_bytes = [0u8; 2];
        {
            let g = self.inner_mut.lock();
            Self::read_guest_bytes_locked(&g.address_space, ring_gpa, &mut head_bytes)?;
        }
        let head = u16::from_le_bytes(head_bytes) as usize;

        let mut next = head;
        for _ in 0..queue_size.min(8) {
            if next >= queue_size as usize {
                warn!("[OVMF-VIRTIO-BLK] descriptor index {next} out of queue size {queue_size}");
                break;
            }

            let desc_gpa = GuestPhysAddr::from(queue_gpa.as_usize() + next * 16);
            let mut desc = [0u8; 16];
            {
                let g = self.inner_mut.lock();
                Self::read_guest_bytes_locked(&g.address_space, desc_gpa, &mut desc)?;
            }

            let addr = u64::from_le_bytes(desc[0..8].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(desc[8..12].try_into().unwrap());
            let flags = u16::from_le_bytes(desc[12..14].try_into().unwrap());
            let desc_next = u16::from_le_bytes(desc[14..16].try_into().unwrap()) as usize;
            let translated = {
                let g = self.inner_mut.lock();
                g.address_space
                    .translate_and_get_limit(GuestPhysAddr::from(addr))
                    .map(|(hpa, limit)| (hpa.as_usize() as u64, limit))
            };

            if let Some((translated_addr, limit)) = translated {
                desc[0..8].copy_from_slice(&translated_addr.to_le_bytes());
                {
                    let g = self.inner_mut.lock();
                    Self::write_guest_bytes_locked(&g.address_space, desc_gpa, &desc)?;
                }
                info!(
                    "[OVMF-VIRTIO-BLK] rewrite desc[{next}] addr={addr:#x}->{translated_addr:#x} \
                     len={len:#x} flags={flags:#x} next={desc_next} limit={limit:#x}"
                );
            } else {
                warn!(
                    "[OVMF-VIRTIO-BLK] failed to translate desc[{next}] addr={addr:#x} \
                     len={len:#x} flags={flags:#x} next={desc_next}"
                );
            }

            if flags & 0x1 == 0 {
                break;
            }
            next = desc_next;
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn dump_ovmf_virtio_blk_queue(&self, tag: &str) -> AxResult {
        let (queue_pfn, queue_size, translated_queue_pfn) = {
            let g = self.inner_mut.lock();
            (
                g.ovmf_virtio_blk.queue_pfn,
                g.ovmf_virtio_blk.queue_size,
                g.ovmf_virtio_blk.translated_queue_pfn,
            )
        };
        if queue_pfn == 0 || queue_size == 0 {
            info!(
                "[OVMF-VIRTIO-BLK] {tag}: queue not configured pfn={queue_pfn:#x} \
                 size={queue_size}"
            );
            return Ok(());
        }

        let queue_gpa = GuestPhysAddr::from((queue_pfn as usize) << 12);
        let avail_gpa = GuestPhysAddr::from(queue_gpa.as_usize() + queue_size as usize * 16);
        let used_gpa = GuestPhysAddr::from(
            (avail_gpa.as_usize() + 4 + queue_size as usize * 2 + 0xfff) & !0xfff,
        );
        let mut avail = [0u8; 6];
        let mut used = [0u8; 4];
        {
            let g = self.inner_mut.lock();
            Self::read_guest_bytes_locked(&g.address_space, avail_gpa, &mut avail)?;
            Self::read_guest_bytes_locked(&g.address_space, used_gpa, &mut used)?;
        }
        let avail_flags = u16::from_le_bytes(avail[0..2].try_into().unwrap());
        let avail_idx = u16::from_le_bytes(avail[2..4].try_into().unwrap());
        let avail_head = u16::from_le_bytes(avail[4..6].try_into().unwrap());
        let used_flags = u16::from_le_bytes(used[0..2].try_into().unwrap());
        let used_idx = u16::from_le_bytes(used[2..4].try_into().unwrap());
        info!(
            "[OVMF-VIRTIO-BLK] {tag}: queue_pfn={queue_pfn:#x} \
             translated_queue_pfn={translated_queue_pfn:#x} size={queue_size} \
             queue_gpa={queue_gpa:?} avail_idx={avail_idx} avail_head={avail_head} \
             avail_flags={avail_flags:#x} used_idx={used_idx} used_flags={used_flags:#x}"
        );
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_acpi_pm_io_read(&self, port: Port, width: AccessWidth) -> AxResult<Option<usize>> {
        if !(ACPI_PM_IO_BASE..ACPI_PM_IO_BASE + ACPI_PM_IO_SIZE).contains(&port.number()) {
            return Ok(None);
        }
        let value = unsafe {
            match width {
                AccessWidth::Byte => X86Port::<u8>::new(port.number()).read() as usize,
                AccessWidth::Word => X86Port::<u16>::new(port.number()).read() as usize,
                AccessWidth::Dword => X86Port::<u32>::new(port.number()).read() as usize,
                AccessWidth::Qword => {
                    return ax_err!(InvalidInput, "unsupported qword ACPI PM I/O read");
                }
            }
        };
        info!(
            "[ACPI-PM-IO] in port={:#x} width={width:?} value={value:#x}",
            port.number()
        );
        Ok(Some(value))
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_acpi_pm_io_write(
        &self,
        port: Port,
        width: AccessWidth,
        val: usize,
    ) -> AxResult<bool> {
        if !(ACPI_PM_IO_BASE..ACPI_PM_IO_BASE + ACPI_PM_IO_SIZE).contains(&port.number()) {
            return Ok(false);
        }
        unsafe {
            match width {
                AccessWidth::Byte => X86Port::<u8>::new(port.number()).write(val as u8),
                AccessWidth::Word => X86Port::<u16>::new(port.number()).write(val as u16),
                AccessWidth::Dword => X86Port::<u32>::new(port.number()).write(val as u32),
                AccessWidth::Qword => {
                    return ax_err!(InvalidInput, "unsupported qword ACPI PM I/O write");
                }
            }
        }
        info!(
            "[ACPI-PM-IO] out port={:#x} width={width:?} value={val:#x}",
            port.number()
        );
        Ok(true)
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_fw_cfg_io_read(&self, port: Port, width: AccessWidth) -> AxResult<Option<usize>> {
        let mut g = self.inner_mut.lock();
        let value = match port.number() {
            FW_CFG_IO_DATA => Some(g.fw_cfg.read_port(width)),
            _ => None,
        };
        Ok(value)
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_fw_cfg_io_write(&self, port: Port, width: AccessWidth, val: usize) -> AxResult<bool> {
        match port.number() {
            FW_CFG_IO_SELECTOR if width == AccessWidth::Word => {
                self.inner_mut.lock().fw_cfg.select(val as u16);
                Ok(true)
            }
            FW_CFG_IO_DMA_ADDRESS | 0x518 if width == AccessWidth::Dword => {
                let mut descriptor = None;
                {
                    let mut g = self.inner_mut.lock();
                    let fw_cfg = &mut g.fw_cfg;
                    let part = u32::from_be(val as u32);
                    if port.number() == FW_CFG_IO_DMA_ADDRESS {
                        fw_cfg.dma_bytes[0..4].copy_from_slice(&part.to_be_bytes());
                        fw_cfg.dma_bytes_written = 4;
                    } else {
                        fw_cfg.dma_bytes[4..8].copy_from_slice(&part.to_be_bytes());
                        if fw_cfg.dma_bytes_written == 4 {
                            let high =
                                u32::from_be_bytes(fw_cfg.dma_bytes[0..4].try_into().unwrap())
                                    as u64;
                            let low = u32::from_be_bytes(fw_cfg.dma_bytes[4..8].try_into().unwrap())
                                as u64;
                            fw_cfg.dma_address = (high << 32) | low;
                            descriptor = Some(fw_cfg.dma_address);
                        }
                        fw_cfg.dma_bytes_written = 0;
                    }
                }
                if let Some(descriptor) = descriptor {
                    self.handle_fw_cfg_dma(GuestPhysAddr::from(descriptor as usize))?;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn handle_fw_cfg_dma(&self, descriptor_gpa: GuestPhysAddr) -> AxResult {
        let mut descriptor = [0u8; 16];
        let mut g = self.inner_mut.lock();
        Self::read_guest_bytes_locked(&g.address_space, descriptor_gpa, &mut descriptor)?;

        let control = u32::from_be_bytes(descriptor[0..4].try_into().unwrap());
        let length = u32::from_be_bytes(descriptor[4..8].try_into().unwrap()) as usize;
        let address = u64::from_be_bytes(descriptor[8..16].try_into().unwrap()) as usize;
        let data_gpa = GuestPhysAddr::from(address);
        let mut status = 0u32;

        if control & FW_CFG_DMA_CTL_SELECT != 0 {
            g.fw_cfg.select((control >> 16) as u16);
        }

        if control & FW_CFG_DMA_CTL_READ != 0 {
            let bytes = g.fw_cfg.read_bytes(length);
            Self::write_guest_bytes_locked(&g.address_space, data_gpa, &bytes)?;
        } else if control & FW_CFG_DMA_CTL_WRITE != 0 {
            let mut bytes = vec![0u8; length];
            Self::read_guest_bytes_locked(&g.address_space, data_gpa, &mut bytes)?;
        } else if control & FW_CFG_DMA_CTL_SKIP != 0 {
            g.fw_cfg.skip_bytes(length);
        } else if control & FW_CFG_DMA_CTL_ERROR != 0 {
            status = FW_CFG_DMA_CTL_ERROR;
        }

        Self::write_guest_bytes_locked(&g.address_space, descriptor_gpa, &status.to_be_bytes())?;
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn read_guest_bytes_locked(
        address_space: &AddrSpace<HostPagingHandler>,
        gpa: GuestPhysAddr,
        buffer: &mut [u8],
    ) -> AxResult {
        match address_space.translated_byte_buffer(gpa, buffer.len()) {
            Some(mut slices) => {
                let mut copied = 0;
                for slice in &mut slices {
                    let take = (buffer.len() - copied).min(slice.len());
                    buffer[copied..copied + take].copy_from_slice(&slice[..take]);
                    copied += take;
                }
                Ok(())
            }
            None => ax_err!(InvalidInput, "failed to translate guest buffer"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn write_guest_bytes_locked(
        address_space: &AddrSpace<HostPagingHandler>,
        gpa: GuestPhysAddr,
        buffer: &[u8],
    ) -> AxResult {
        match address_space.translated_byte_buffer(gpa, buffer.len()) {
            Some(mut slices) => {
                let mut copied = 0;
                for slice in &mut slices {
                    let take = (buffer.len() - copied).min(slice.len());
                    slice[..take].copy_from_slice(&buffer[copied..copied + take]);
                    copied += take;
                }
                Ok(())
            }
            None => ax_err!(InvalidInput, "failed to translate guest buffer"),
        }
    }

    /// Cleanup resources for the VM before drop.
    /// This is called internally by the Drop implementation.
    fn cleanup_resources(&self) {
        info!("Cleaning up VM[{}] resources...", self.id());

        // 1. Ensure the VM is in Stopping or Stopped state
        let current_status = self.vm_status();
        if !matches!(current_status, VMStatus::Stopping | VMStatus::Stopped) {
            warn!(
                "VM[{}] is being dropped without explicit shutdown (status: {:?}), marking as \
                 stopping",
                self.id(),
                current_status
            );
            self.set_vm_status(VMStatus::Stopping);
        }

        let mut inner_mut = self.inner_mut.lock();

        // First, collect all memory regions to clean up
        // We need to clone the regions to avoid borrowing issues
        let regions_to_cleanup: Vec<VMMemoryRegion> = inner_mut.memory_regions.clone();

        // Unmap all memory regions from the address space
        // This must be done BEFORE deallocating memory to avoid use-after-free
        for region in &regions_to_cleanup {
            debug!(
                "VM[{}] unmapping memory region: GPA={:#x}, size={:#x}",
                self.id(),
                region.gpa.as_usize(),
                region.size()
            );
            // Unmap the region from guest physical address space
            if let Err(e) = inner_mut.address_space.unmap(region.gpa, region.size()) {
                warn!(
                    "VM[{}] failed to unmap region at GPA={:#x}: {:?}",
                    self.id(),
                    region.gpa.as_usize(),
                    e
                );
            }
        }

        // Now it's safe to deallocate the memory
        for region in &regions_to_cleanup {
            // Only deallocate memory regions that were allocated by the allocator
            if region.needs_dealloc {
                debug!(
                    "VM[{}] deallocating memory region: HVA={:#x}, size={:#x}",
                    self.id(),
                    region.hva.as_usize(),
                    region.size()
                );
                unsafe {
                    alloc::alloc::dealloc(region.hva.as_mut_ptr(), region.layout);
                }
            } else {
                debug!(
                    "VM[{}] skipping dealloc for reserved memory region: GPA={:#x}, HVA={:#x}, \
                     size={:#x}",
                    self.id(),
                    region.gpa.as_usize(),
                    region.hva.as_usize(),
                    region.size()
                );
            }
        }
        inner_mut.memory_regions.clear();

        // Clear remaining address space mappings
        // This includes:
        // - Passthrough device MMIO mappings
        // - Emulated device MMIO mappings
        // - Reserved memory mappings
        // - All other page table entries
        debug!(
            "VM[{}] clearing remaining address space mappings",
            self.id()
        );
        inner_mut.address_space.clear();

        // Release the lock before accessing inner_const
        drop(inner_mut);

        // Device cleanup
        // Although devices will be automatically dropped when inner_const is dropped,
        // we should perform explicit cleanup if devices hold resources like:
        // - Hardware interrupt registrations
        // - DMA mappings
        // - Background threads or timers
        if let Some(inner_const) = self.inner_const.get() {
            debug!(
                "VM[{}] devices cleanup: {} MMIO devices, {} SysReg devices",
                self.id(),
                inner_const.devices.iter_mmio_dev().count(),
                inner_const.devices.iter_sys_reg_dev().count()
            );

            // TODO: Add device-specific cleanup if needed
            // For example:
            // - Stop device background tasks
            // - Unregister interrupts
            // - Release device-specific resources

            // Note: Device Arc references will be dropped automatically when
            // inner_const is dropped at the end of AxVM's drop
        }

        info!("VM[{}] resources cleanup completed", self.id());
    }
}

impl Drop for AxVM {
    fn drop(&mut self) {
        info!("Dropping VM[{}]", self.id());

        // Clean up all allocated resources
        self.cleanup_resources();

        info!("VM[{}] dropped", self.id());
    }
}
