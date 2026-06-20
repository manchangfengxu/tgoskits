use alloc::{vec, vec::Vec};

use ax_errno::{AxError, AxResult, ax_err};
use axaddrspace::GuestMemoryAccessor;
use axdevice_base::{AccessWidth, Port};
use axvm_types::GuestPhysAddr;

pub const LEGACY_BLK_IO_BASE: u16 = 0x6000;
pub const LEGACY_BLK_IO_SIZE: u16 = 0x80;

const REG_DEVICE_FEATURES: u16 = 0x00;
const REG_QUEUE_PFN: u16 = 0x08;
const REG_QUEUE_SIZE: u16 = 0x0c;
const REG_QUEUE_NOTIFY: u16 = 0x10;
const REG_CAPACITY_LOW: u16 = 0x14;
const REG_CAPACITY_HIGH: u16 = 0x18;
const REG_STATUS: u16 = 0x12;
const REG_ISR_STATUS: u16 = 0x13;

const SECTOR_SIZE: usize = 512;
const DEFAULT_QUEUE_SIZE: u16 = 8;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;
const VIRTIO_BLK_T_GET_ID: u32 = 8;

const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;
const VIRTIO_BLK_S_UNSUPP: u8 = 2;

const VIRTIO_ISR_QUEUE: u8 = 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LegacyNotifyResult {
    pub used_any: bool,
    pub published_used_count: u16,
    pub should_raise_irq: bool,
}

impl LegacyNotifyResult {
    const fn idle() -> Self {
        Self {
            used_any: false,
            published_used_count: 0,
            should_raise_irq: false,
        }
    }

    const fn completed_one() -> Self {
        Self {
            used_any: true,
            published_used_count: 1,
            should_raise_irq: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.used_any |= other.used_any;
        self.published_used_count = self
            .published_used_count
            .saturating_add(other.published_used_count);
        self.should_raise_irq |= other.should_raise_irq;
    }
}

#[derive(Clone, Copy, Debug)]
struct Descriptor {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

impl Descriptor {
    fn writable(self) -> bool {
        self.flags & VIRTQ_DESC_F_WRITE != 0
    }
}

#[derive(Debug)]
struct VirtioBlkRequest {
    kind: u32,
    sector: u64,
    data: Vec<Descriptor>,
    status: Descriptor,
}

impl VirtioBlkRequest {
    fn from_chain<M: GuestMemoryAccessor>(descs: &[Descriptor], mem: &M) -> AxResult<Self> {
        if descs.len() < 2 {
            return ax_err!(InvalidInput, "virtio-blk descriptor chain is too short");
        }
        let header = descs[0];
        if header.writable() || header.len < 16 {
            return ax_err!(InvalidInput, "invalid virtio-blk request header descriptor");
        }
        let status = *descs.last().ok_or(AxError::InvalidInput)?;
        if !status.writable() || status.len == 0 {
            return ax_err!(InvalidInput, "invalid virtio-blk status descriptor");
        }

        let mut raw = [0u8; 16];
        mem.read_buffer(GuestPhysAddr::from(header.addr as usize), &mut raw)?;
        let kind = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let sector = u64::from_le_bytes(raw[8..16].try_into().unwrap());

        Ok(Self {
            kind,
            sector,
            data: descs[1..descs.len() - 1].to_vec(),
            status,
        })
    }

    fn validate_data_descriptors(&self) -> AxResult {
        match self.kind {
            VIRTIO_BLK_T_IN | VIRTIO_BLK_T_GET_ID => {
                if self.data.is_empty() {
                    return ax_err!(
                        InvalidInput,
                        "virtio-blk read request has no data descriptors"
                    );
                }
                if self.data.iter().any(|desc| !desc.writable()) {
                    return ax_err!(
                        InvalidInput,
                        "virtio-blk read data descriptors must be writable"
                    );
                }
            }
            VIRTIO_BLK_T_OUT => {
                if self.data.is_empty() {
                    return ax_err!(
                        InvalidInput,
                        "virtio-blk write request has no data descriptors"
                    );
                }
                if self.data.iter().any(|desc| desc.writable()) {
                    return ax_err!(
                        InvalidInput,
                        "virtio-blk write data descriptors must be readable"
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequestCompletion {
    status: u8,
    used_len: u32,
}

impl RequestCompletion {
    const fn ok(used_len: u32) -> Self {
        Self {
            status: VIRTIO_BLK_S_OK,
            used_len,
        }
    }

    const fn ioerr() -> Self {
        Self {
            status: VIRTIO_BLK_S_IOERR,
            used_len: 1,
        }
    }

    const fn unsupp() -> Self {
        Self {
            status: VIRTIO_BLK_S_UNSUPP,
            used_len: 1,
        }
    }
}

#[derive(Debug)]
struct LegacyQueue {
    size: u16,
    pfn: u32,
    last_avail_idx: u16,
}

impl LegacyQueue {
    const fn new() -> Self {
        Self {
            size: DEFAULT_QUEUE_SIZE,
            pfn: 0,
            last_avail_idx: 0,
        }
    }

    fn set_size(&mut self, size: u16) {
        self.size = size.max(1);
    }

    fn set_pfn(&mut self, pfn: u32) {
        if self.pfn != pfn {
            self.last_avail_idx = 0;
        }
        self.pfn = pfn;
    }

    fn base(&self) -> AxResult<usize> {
        if self.pfn == 0 {
            return ax_err!(InvalidInput, "virtio-blk queue is not configured");
        }
        Ok((self.pfn as usize) << 12)
    }

    fn desc_table_gpa(&self) -> AxResult<usize> {
        self.base()
    }

    fn avail_ring_gpa(&self) -> AxResult<usize> {
        Ok(self.base()? + self.size as usize * 16)
    }

    fn used_ring_gpa(&self) -> AxResult<usize> {
        let avail_end = self.avail_ring_gpa()? + 4 + self.size as usize * 2;
        Ok((avail_end + 0xfff) & !0xfff)
    }

    fn read_desc<M: GuestMemoryAccessor>(&self, index: u16, mem: &M) -> AxResult<Descriptor> {
        if index >= self.size {
            return ax_err!(InvalidInput, "virtio-blk descriptor index out of range");
        }
        let gpa = self.desc_table_gpa()? + index as usize * 16;
        let mut raw = [0u8; 16];
        mem.read_buffer(GuestPhysAddr::from(gpa), &mut raw)?;
        Ok(Descriptor {
            addr: u64::from_le_bytes(raw[0..8].try_into().unwrap()),
            len: u32::from_le_bytes(raw[8..12].try_into().unwrap()),
            flags: u16::from_le_bytes(raw[12..14].try_into().unwrap()),
            next: u16::from_le_bytes(raw[14..16].try_into().unwrap()),
        })
    }

    fn collect_chain<M: GuestMemoryAccessor>(
        &self,
        head: u16,
        mem: &M,
    ) -> AxResult<Vec<Descriptor>> {
        let mut descs = Vec::new();
        let mut index = head;
        for _ in 0..self.size {
            let desc = self.read_desc(index, mem)?;
            descs.push(desc);
            if desc.flags & VIRTQ_DESC_F_NEXT == 0 {
                return Ok(descs);
            }
            index = desc.next;
        }
        ax_err!(InvalidInput, "virtio-blk descriptor chain loop")
    }

    fn pop_available<M: GuestMemoryAccessor>(&mut self, mem: &M) -> AxResult<Option<u16>> {
        let avail = self.avail_ring_gpa()?;
        let avail_idx: u16 = mem.read_obj(GuestPhysAddr::from(avail + 2))?;
        if self.last_avail_idx == avail_idx {
            return Ok(None);
        }
        let slot = self.last_avail_idx % self.size;
        let head: u16 = mem.read_obj(GuestPhysAddr::from(avail + 4 + slot as usize * 2))?;
        self.last_avail_idx = self.last_avail_idx.wrapping_add(1);
        Ok(Some(head))
    }

    fn publish_used<M: GuestMemoryAccessor>(&self, head: u16, len: u32, mem: &M) -> AxResult {
        let used = self.used_ring_gpa()?;
        let idx: u16 = mem.read_obj(GuestPhysAddr::from(used + 2))?;
        let slot = idx % self.size;
        mem.write_obj(
            GuestPhysAddr::from(used + 4 + slot as usize * 8),
            head as u32,
        )?;
        mem.write_obj(GuestPhysAddr::from(used + 8 + slot as usize * 8), len)?;
        mem.write_obj(GuestPhysAddr::from(used + 2), idx.wrapping_add(1))?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct LegacyVirtioBlk {
    queue: LegacyQueue,
    disk: MemoryDisk,
    status: u8,
    isr_status: u8,
}

#[derive(Debug, Default)]
struct MemoryDisk {
    bytes: Vec<u8>,
}

impl MemoryDisk {
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    fn capacity_sectors(&self) -> u64 {
        (self.bytes.len() / SECTOR_SIZE) as u64
    }

    fn read_at(&self, offset: usize, dst: &mut [u8]) -> AxResult {
        let end = checked_disk_range(offset, dst.len(), self.bytes.len())?;
        dst.copy_from_slice(&self.bytes[offset..end]);
        Ok(())
    }

    fn write_at(&mut self, offset: usize, src: &[u8]) -> AxResult {
        let end = checked_disk_range(offset, src.len(), self.bytes.len())?;
        self.bytes[offset..end].copy_from_slice(src);
        Ok(())
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

fn checked_disk_range(offset: usize, len: usize, disk_len: usize) -> AxResult<usize> {
    offset
        .checked_add(len)
        .filter(|end| *end <= disk_len)
        .ok_or(AxError::InvalidInput)
}

impl LegacyVirtioBlk {
    pub const fn new() -> Self {
        Self {
            queue: LegacyQueue::new(),
            disk: MemoryDisk { bytes: Vec::new() },
            status: 0,
            isr_status: 0,
        }
    }

    pub fn owns_port(port: Port) -> bool {
        (LEGACY_BLK_IO_BASE..LEGACY_BLK_IO_BASE + LEGACY_BLK_IO_SIZE).contains(&port.number())
    }

    pub fn install_disk_image(&mut self, disk: Vec<u8>) {
        self.disk = MemoryDisk::new(disk);
    }

    pub fn disk_image(&self) -> &[u8] {
        self.disk.as_slice()
    }

    pub fn handle_read(&mut self, port: Port, width: AccessWidth) -> AxResult<usize> {
        if !Self::owns_port(port) {
            return ax_err!(InvalidInput, "virtio-blk port read outside device range");
        }
        let offset = port.number() - LEGACY_BLK_IO_BASE;
        match (offset, width) {
            (REG_DEVICE_FEATURES, AccessWidth::Dword) => Ok(0),
            (REG_QUEUE_PFN, AccessWidth::Dword) => Ok(self.queue.pfn as usize),
            (REG_QUEUE_SIZE, AccessWidth::Word) => Ok(self.queue.size as usize),
            (REG_CAPACITY_LOW, AccessWidth::Dword) => Ok(self.capacity_sectors() as u32 as usize),
            (REG_CAPACITY_HIGH, AccessWidth::Dword) => Ok((self.capacity_sectors() >> 32) as usize),
            (REG_STATUS, AccessWidth::Byte) => Ok(self.status as usize),
            (REG_ISR_STATUS, AccessWidth::Byte) => {
                let value = self.isr_status;
                self.isr_status = 0;
                Ok(value as usize)
            }
            _ => Ok(0),
        }
    }

    pub fn handle_write<M: GuestMemoryAccessor>(
        &mut self,
        port: Port,
        width: AccessWidth,
        value: usize,
        mem: &M,
    ) -> AxResult<LegacyNotifyResult> {
        if !Self::owns_port(port) {
            return ax_err!(InvalidInput, "virtio-blk port write outside device range");
        }
        let offset = port.number() - LEGACY_BLK_IO_BASE;
        let notify = match (offset, width) {
            (REG_QUEUE_SIZE, AccessWidth::Word) => {
                self.queue.set_size(value as u16);
                LegacyNotifyResult::idle()
            }
            (REG_QUEUE_PFN, AccessWidth::Dword) => {
                self.queue.set_pfn(value as u32);
                LegacyNotifyResult::idle()
            }
            (REG_QUEUE_NOTIFY, AccessWidth::Word) => self.process_queue(mem)?,
            (REG_STATUS, AccessWidth::Byte) => {
                self.status = value as u8;
                LegacyNotifyResult::idle()
            }
            _ => LegacyNotifyResult::idle(),
        };
        if notify.should_raise_irq {
            self.isr_status |= VIRTIO_ISR_QUEUE;
        }
        Ok(notify)
    }

    fn capacity_sectors(&self) -> u64 {
        self.disk.capacity_sectors()
    }

    fn process_queue<M: GuestMemoryAccessor>(&mut self, mem: &M) -> AxResult<LegacyNotifyResult> {
        let mut notify = LegacyNotifyResult::idle();
        while let Some(head) = self.queue.pop_available(mem)? {
            let descs = self.queue.collect_chain(head, mem)?;
            let request = VirtioBlkRequest::from_chain(&descs, mem)?;
            let completion = self.execute_request(&request, mem)?;
            self.write_status(&request, mem, completion.status)?;
            self.queue.publish_used(head, completion.used_len, mem)?;
            notify.merge(LegacyNotifyResult::completed_one());
        }
        Ok(notify)
    }

    fn execute_request<M: GuestMemoryAccessor>(
        &mut self,
        request: &VirtioBlkRequest,
        mem: &M,
    ) -> AxResult<RequestCompletion> {
        match request.kind {
            VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_GET_ID => {
                if let Err(err) = request.validate_data_descriptors() {
                    debug!("virtio-blk request validation failed: {err:?}");
                    return Ok(RequestCompletion::ioerr());
                }
            }
            _ => return Ok(RequestCompletion::unsupp()),
        }

        match request.kind {
            VIRTIO_BLK_T_IN => self.execute_read(request, mem),
            VIRTIO_BLK_T_OUT => self.execute_write(request, mem),
            VIRTIO_BLK_T_GET_ID => self.execute_get_id(request, mem),
            _ => Ok(RequestCompletion::unsupp()),
        }
    }

    fn execute_read<M: GuestMemoryAccessor>(
        &mut self,
        request: &VirtioBlkRequest,
        mem: &M,
    ) -> AxResult<RequestCompletion> {
        let mut offset = self.disk_offset(request.sector)?;
        let mut used_len = 0u32;
        for desc in &request.data {
            let len = desc.len as usize;
            let mut bytes = vec![0; len];
            if self.disk.read_at(offset, &mut bytes).is_err() {
                return Ok(RequestCompletion::ioerr());
            }
            mem.write_buffer(GuestPhysAddr::from(desc.addr as usize), &bytes)?;
            offset += len;
            used_len = used_len.saturating_add(desc.len);
        }
        Ok(RequestCompletion::ok(used_len + 1))
    }

    fn execute_write<M: GuestMemoryAccessor>(
        &mut self,
        request: &VirtioBlkRequest,
        mem: &M,
    ) -> AxResult<RequestCompletion> {
        let mut offset = self.disk_offset(request.sector)?;
        for desc in &request.data {
            let len = desc.len as usize;
            let mut bytes = vec![0; len];
            mem.read_buffer(GuestPhysAddr::from(desc.addr as usize), &mut bytes)?;
            if self.disk.write_at(offset, &bytes).is_err() {
                return Ok(RequestCompletion::ioerr());
            }
            offset += len;
        }
        Ok(RequestCompletion::ok(1))
    }

    fn execute_get_id<M: GuestMemoryAccessor>(
        &mut self,
        request: &VirtioBlkRequest,
        mem: &M,
    ) -> AxResult<RequestCompletion> {
        const SERIAL: &[u8] = b"AXVISOR-UEFI-BLK\0";
        let mut written = 0u32;
        for desc in &request.data {
            let remaining = &SERIAL[written as usize..];
            if remaining.is_empty() {
                break;
            }
            let take = remaining.len().min(desc.len as usize);
            mem.write_buffer(GuestPhysAddr::from(desc.addr as usize), &remaining[..take])?;
            written += take as u32;
        }
        Ok(RequestCompletion::ok(written))
    }

    fn write_status<M: GuestMemoryAccessor>(
        &self,
        request: &VirtioBlkRequest,
        mem: &M,
        status: u8,
    ) -> AxResult {
        mem.write_obj(GuestPhysAddr::from(request.status.addr as usize), status)
    }

    fn disk_offset(&self, sector: u64) -> AxResult<usize> {
        let offset = sector
            .checked_mul(SECTOR_SIZE as u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(AxError::InvalidInput)?;
        if offset > self.disk.as_slice().len() {
            return ax_err!(InvalidInput, "virtio-blk sector is outside disk image");
        }
        Ok(offset)
    }
}

impl Default for LegacyVirtioBlk {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use ax_memory_addr::PhysAddr;
    use axaddrspace::GuestMemoryAccessor;
    use axdevice_base::{AccessWidth, Port};
    use axvm_types::GuestPhysAddr;

    use super::{
        LEGACY_BLK_IO_BASE, LegacyVirtioBlk, MemoryDisk, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_T_GET_ID,
    };

    struct TestGuestMemory {
        bytes: vec::Vec<u8>,
    }

    impl TestGuestMemory {
        fn new(size: usize) -> Self {
            Self {
                bytes: vec![0; size],
            }
        }

        fn write(&mut self, gpa: usize, bytes: &[u8]) {
            self.bytes[gpa..gpa + bytes.len()].copy_from_slice(bytes);
        }

        fn read(&self, gpa: usize, len: usize) -> &[u8] {
            &self.bytes[gpa..gpa + len]
        }

        fn write_u16(&mut self, gpa: usize, value: u16) {
            self.write(gpa, &value.to_le_bytes());
        }

        fn write_u32(&mut self, gpa: usize, value: u32) {
            self.write(gpa, &value.to_le_bytes());
        }

        fn write_u64(&mut self, gpa: usize, value: u64) {
            self.write(gpa, &value.to_le_bytes());
        }

        fn read_u16(&self, gpa: usize) -> u16 {
            u16::from_le_bytes(self.read(gpa, 2).try_into().unwrap())
        }

        fn read_u32(&self, gpa: usize) -> u32 {
            u32::from_le_bytes(self.read(gpa, 4).try_into().unwrap())
        }

        fn write_desc(
            &mut self,
            table: usize,
            index: usize,
            addr: u64,
            len: u32,
            flags: u16,
            next: u16,
        ) {
            let base = table + index * 16;
            self.write_u64(base, addr);
            self.write_u32(base + 8, len);
            self.write_u16(base + 12, flags);
            self.write_u16(base + 14, next);
        }
    }

    impl GuestMemoryAccessor for TestGuestMemory {
        fn translate_and_get_limit(&self, guest_addr: GuestPhysAddr) -> Option<(PhysAddr, usize)> {
            let start = guest_addr.as_usize();
            if start >= self.bytes.len() {
                return None;
            }
            let ptr = unsafe { self.bytes.as_ptr().add(start) as usize };
            Some((PhysAddr::from_usize(ptr), self.bytes.len() - start))
        }
    }

    #[test]
    fn memory_disk_reports_capacity_and_updates_sectors() {
        let mut disk = MemoryDisk::new(vec![0; 1024]);
        let mut sector = [0u8; 4];

        disk.read_at(512, &mut sector).unwrap();
        assert_eq!(sector, [0; 4]);

        disk.write_at(512, &[1, 2, 3, 4]).unwrap();
        disk.read_at(512, &mut sector).unwrap();

        assert_eq!(disk.capacity_sectors(), 2);
        assert_eq!(sector, [1, 2, 3, 4]);
    }

    #[test]
    fn reports_capacity_from_installed_disk_image() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let low = blk
            .handle_read(Port::new(LEGACY_BLK_IO_BASE + 0x14), AccessWidth::Dword)
            .unwrap();
        let high = blk
            .handle_read(Port::new(LEGACY_BLK_IO_BASE + 0x18), AccessWidth::Dword)
            .unwrap();

        assert_eq!(low, 2);
        assert_eq!(high, 0);
    }

    #[test]
    fn read_request_copies_sector_into_guest_buffer_and_publishes_used_ring() {
        let mut disk = vec![0; 1024];
        disk[512..516].copy_from_slice(&[1, 2, 3, 4]);
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(disk);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let used = (avail + 4 + 8 * 2 + 0xfff) & !0xfff;
        let header = 0x3000;
        let data = 0x3100;
        let status = 0x3200;

        mem.write_u32(header, 0);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 1);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, data as u64, 4, 1 | 2, 2);
        mem.write_desc(desc, 2, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x10),
            AccessWidth::Word,
            0,
            &mem,
        )
        .unwrap();

        assert_eq!(mem.read(data, 4), &[1, 2, 3, 4]);
        assert_eq!(mem.read(status, 1), &[0]);
        assert_eq!(mem.read_u16(used + 2), 1);
        assert_eq!(mem.read_u32(used + 4), 0);
        assert_eq!(mem.read_u32(used + 8), 5);
    }

    #[test]
    fn write_request_updates_backend_and_reports_only_status_byte_used() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let used = (avail + 4 + 8 * 2 + 0xfff) & !0xfff;
        let header = 0x3000;
        let data = 0x3100;
        let status = 0x3200;

        mem.write_u32(header, 1);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 1);
        mem.write(data, &[9, 8, 7, 6]);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, data as u64, 4, 1, 2);
        mem.write_desc(desc, 2, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x10),
            AccessWidth::Word,
            0,
            &mem,
        )
        .unwrap();

        assert_eq!(&blk.disk_image()[512..516], &[9, 8, 7, 6]);
        assert_eq!(mem.read(status, 1), &[0]);
        assert_eq!(mem.read_u16(used + 2), 1);
        assert_eq!(mem.read_u32(used + 8), 1);
    }

    #[test]
    fn notify_without_available_request_returns_idle_notify_result() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let avail = queue + 16 * 8;
        mem.write_u16(avail + 2, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();

        let notify = blk
            .handle_write(
                Port::new(LEGACY_BLK_IO_BASE + 0x10),
                AccessWidth::Word,
                0,
                &mem,
            )
            .unwrap();
        let isr = blk
            .handle_read(Port::new(LEGACY_BLK_IO_BASE + 0x13), AccessWidth::Byte)
            .unwrap();

        assert!(!notify.used_any);
        assert_eq!(notify.published_used_count, 0);
        assert!(!notify.should_raise_irq);
        assert_eq!(isr, 0);
    }

    #[test]
    fn completed_request_reports_notify_result_and_sets_queue_isr() {
        let mut disk = vec![0; 1024];
        disk[512..516].copy_from_slice(&[1, 2, 3, 4]);
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(disk);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let header = 0x3000;
        let data = 0x3100;
        let status = 0x3200;

        mem.write_u32(header, 0);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 1);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, data as u64, 4, 1 | 2, 2);
        mem.write_desc(desc, 2, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();

        let notify = blk
            .handle_write(
                Port::new(LEGACY_BLK_IO_BASE + 0x10),
                AccessWidth::Word,
                0,
                &mem,
            )
            .unwrap();
        let first_isr = blk
            .handle_read(Port::new(LEGACY_BLK_IO_BASE + 0x13), AccessWidth::Byte)
            .unwrap();
        let second_isr = blk
            .handle_read(Port::new(LEGACY_BLK_IO_BASE + 0x13), AccessWidth::Byte)
            .unwrap();

        assert!(notify.used_any);
        assert_eq!(notify.published_used_count, 1);
        assert!(notify.should_raise_irq);
        assert_eq!(first_isr, 1);
        assert_eq!(second_isr, 0);
    }

    #[test]
    fn read_request_without_data_descriptors_reports_ioerr() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let used = (avail + 4 + 8 * 2 + 0xfff) & !0xfff;
        let header = 0x3000;
        let status = 0x3200;

        mem.write_u32(header, 0);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 0);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();

        let notify = blk
            .handle_write(
                Port::new(LEGACY_BLK_IO_BASE + 0x10),
                AccessWidth::Word,
                0,
                &mem,
            )
            .unwrap();

        assert!(notify.used_any);
        assert_eq!(mem.read(status, 1), &[VIRTIO_BLK_S_IOERR]);
        assert_eq!(mem.read_u16(used + 2), 1);
        assert_eq!(mem.read_u32(used + 8), 1);
    }

    #[test]
    fn get_id_with_readonly_data_descriptor_reports_ioerr() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let used = (avail + 4 + 8 * 2 + 0xfff) & !0xfff;
        let header = 0x3000;
        let data = 0x3100;
        let status = 0x3200;

        mem.write_u32(header, VIRTIO_BLK_T_GET_ID);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 0);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, data as u64, 20, 1, 2);
        mem.write_desc(desc, 2, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();

        let notify = blk
            .handle_write(
                Port::new(LEGACY_BLK_IO_BASE + 0x10),
                AccessWidth::Word,
                0,
                &mem,
            )
            .unwrap();

        assert!(notify.used_any);
        assert_eq!(mem.read(status, 1), &[VIRTIO_BLK_S_IOERR]);
        assert_eq!(mem.read_u16(used + 2), 1);
        assert_eq!(mem.read_u32(used + 8), 1);
    }

    #[test]
    fn get_id_writes_a_null_terminated_serial() {
        let mut blk = LegacyVirtioBlk::new();
        blk.install_disk_image(vec![0; 1024]);

        let mut mem = TestGuestMemory::new(0x5000);
        let queue = 0x1000;
        let desc = queue;
        let avail = queue + 16 * 8;
        let used = (avail + 4 + 8 * 2 + 0xfff) & !0xfff;
        let header = 0x3000;
        let data = 0x3100;
        let status = 0x3200;

        mem.write_u32(header, VIRTIO_BLK_T_GET_ID);
        mem.write_u32(header + 4, 0);
        mem.write_u64(header + 8, 0);
        mem.write(data, &[0xff; 20]);
        mem.write_desc(desc, 0, header as u64, 16, 1, 1);
        mem.write_desc(desc, 1, data as u64, 20, 1 | 2, 2);
        mem.write_desc(desc, 2, status as u64, 1, 2, 0);
        mem.write_u16(avail + 2, 1);
        mem.write_u16(avail + 4, 0);

        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x0c),
            AccessWidth::Word,
            8,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x08),
            AccessWidth::Dword,
            queue >> 12,
            &mem,
        )
        .unwrap();
        blk.handle_write(
            Port::new(LEGACY_BLK_IO_BASE + 0x10),
            AccessWidth::Word,
            0,
            &mem,
        )
        .unwrap();

        assert_eq!(mem.read(data, 17), b"AXVISOR-UEFI-BLK\0");
        assert_eq!(mem.read(status, 1), &[0]);
        assert_eq!(mem.read_u16(used + 2), 1);
        assert_eq!(mem.read_u32(used + 8), 17);
    }
}
