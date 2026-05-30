#![no_std]
#![no_main]
#![allow(dead_code)]

use core::{
    fmt::{self, Write},
    panic::PanicInfo,
    sync::atomic::{AtomicPtr, Ordering},
};

type EfiHandle = *mut core::ffi::c_void;
type EfiStatus = usize;

const EFI_SUCCESS: EfiStatus = 0;
const EFI_BUFFER_TOO_SMALL: EfiStatus = 0x8000_0000_0000_0005;
const EFI_CONVENTIONAL_MEMORY: u32 = 7;

#[repr(C)]
struct SimpleTextOutputProtocol {
    reset: usize,
    output_string:
        unsafe extern "C" fn(this: *mut SimpleTextOutputProtocol, s: *const u16) -> EfiStatus,
    test_string: usize,
    query_mode: usize,
    set_mode: usize,
    set_attribute: usize,
    clear_screen: unsafe extern "C" fn(this: *mut SimpleTextOutputProtocol) -> EfiStatus,
    set_cursor_position: usize,
    enable_cursor: usize,
    mode: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EfiMemoryDescriptor {
    typ: u32,
    pad: u32,
    phys_start: u64,
    virt_start: u64,
    number_of_pages: u64,
    attribute: u64,
}

#[repr(C)]
struct BootServices {
    header: [u8; 24],
    raise_tpl: usize,
    restore_tpl: usize,
    allocate_pages: usize,
    free_pages: usize,
    get_memory_map: unsafe extern "C" fn(
        memory_map_size: *mut usize,
        memory_map: *mut EfiMemoryDescriptor,
        map_key: *mut usize,
        descriptor_size: *mut usize,
        descriptor_version: *mut u32,
    ) -> EfiStatus,
}

#[repr(C)]
struct ConfigurationTable {
    vendor_guid: [u8; 16],
    vendor_table: usize,
}

#[repr(C)]
struct SystemTable {
    header: [u8; 24],
    firmware_vendor: *const u16,
    firmware_revision: u32,
    console_in_handle: usize,
    con_in: usize,
    console_out_handle: usize,
    con_out: *mut SimpleTextOutputProtocol,
    standard_error_handle: usize,
    std_err: usize,
    runtime_services: usize,
    boot_services: *mut BootServices,
    number_of_table_entries: usize,
    configuration_table: *mut ConfigurationTable,
}

struct UefiConsole {
    con_out: *mut SimpleTextOutputProtocol,
}

impl UefiConsole {
    fn new(system_table: *mut SystemTable) -> Option<Self> {
        let con_out = unsafe { (*system_table).con_out };
        (!con_out.is_null()).then_some(Self { con_out })
    }

    fn clear(&mut self) {
        unsafe {
            ((*self.con_out).clear_screen)(self.con_out);
        }
    }
}

impl Write for UefiConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for ch in s.encode_utf16() {
            let buf = [ch, 0];
            unsafe {
                ((*self.con_out).output_string)(self.con_out, buf.as_ptr());
            }
        }
        Ok(())
    }
}

static SYSTEM_TABLE: AtomicPtr<SystemTable> = AtomicPtr::new(core::ptr::null_mut());

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    let system_table = SYSTEM_TABLE.load(Ordering::Relaxed);
    if let Some(mut console) = (!system_table.is_null())
        .then(|| UefiConsole::new(system_table))
        .flatten()
    {
        let _ = writeln!(console, "\r\nPANIC: {info}\r\n");
    }
    halt()
}

#[unsafe(no_mangle)]
extern "C" fn efi_main(_image: EfiHandle, system_table: *mut SystemTable) -> EfiStatus {
    SYSTEM_TABLE.store(system_table, Ordering::Relaxed);

    let Some(mut console) = UefiConsole::new(system_table) else {
        return EFI_SUCCESS;
    };
    console.clear();

    let _ = writeln!(console, "\r\n======================================");
    let _ = writeln!(console, "  ArceOS UEFI helloworld");
    let _ = writeln!(console, "======================================\r\n");
    let _ = writeln!(console, "OVMF loaded this image as PE32+ EFI application.");
    let _ = writeln!(
        console,
        "This keeps the legacy multiboot entry untouched.\r\n"
    );

    print_memory_map(system_table, &mut console);

    let _ = writeln!(console, "\r\n======================================");
    let _ = writeln!(console, "  ArceOS UEFI shell-stage boot OK");
    let _ = writeln!(console, "======================================\r\n");

    halt()
}

fn print_memory_map(system_table: *mut SystemTable, console: &mut UefiConsole) {
    let boot_services = unsafe { (*system_table).boot_services };
    if boot_services.is_null() {
        let _ = writeln!(console, "ERROR: UEFI BootServices is null.");
        return;
    }

    let mut map_size = 0usize;
    let mut map_key = 0usize;
    let mut desc_size = 0usize;
    let mut desc_version = 0u32;

    let status = unsafe {
        ((*boot_services).get_memory_map)(
            &mut map_size,
            core::ptr::null_mut(),
            &mut map_key,
            &mut desc_size,
            &mut desc_version,
        )
    };

    if status != EFI_BUFFER_TOO_SMALL || desc_size == 0 {
        let _ = writeln!(
            console,
            "ERROR: GetMemoryMap probe status=0x{status:x} desc_size={desc_size}"
        );
        return;
    }

    let mut map_buf = [0u8; 64 * 1024];
    map_size = map_size.saturating_add(desc_size * 8);
    if map_size > map_buf.len() {
        let _ = writeln!(console, "ERROR: memory map too large: {map_size} bytes");
        return;
    }

    let status = unsafe {
        ((*boot_services).get_memory_map)(
            &mut map_size,
            map_buf.as_mut_ptr() as *mut EfiMemoryDescriptor,
            &mut map_key,
            &mut desc_size,
            &mut desc_version,
        )
    };

    if status != EFI_SUCCESS {
        let _ = writeln!(console, "ERROR: GetMemoryMap status=0x{status:x}");
        return;
    }

    let entries = map_size / desc_size;
    let mut total_pages = 0u64;
    let mut conventional_pages = 0u64;

    let _ = writeln!(console, "UEFI memory map:");
    let _ = writeln!(console, "  entries:            {entries}");
    let _ = writeln!(console, "  descriptor size:    {desc_size}");
    let _ = writeln!(console, "  descriptor version: {desc_version}");

    for i in 0..entries {
        let offset = i * desc_size;
        let desc = unsafe { &*(map_buf.as_ptr().add(offset) as *const EfiMemoryDescriptor) };
        total_pages += desc.number_of_pages;
        if desc.typ == EFI_CONVENTIONAL_MEMORY {
            conventional_pages += desc.number_of_pages;
        }

        if i < 16 {
            let start = desc.phys_start;
            let end = start + desc.number_of_pages * 4096;
            let typ = memory_type_name(desc.typ);
            let _ = writeln!(
                console,
                "  {typ:<13} [0x{start:016x}, 0x{end:016x}) {pages:>5} pages",
                pages = desc.number_of_pages,
            );
        }
    }

    if entries > 16 {
        let _ = writeln!(console, "  ... {} more entries", entries - 16);
    }

    let _ = writeln!(console, "  total memory:       {} MiB", total_pages / 256);
    let _ = writeln!(
        console,
        "  conventional:       {} MiB",
        conventional_pages / 256
    );
}

fn memory_type_name(typ: u32) -> &'static str {
    match typ {
        0 => "Reserved",
        1 => "LoaderCode",
        2 => "LoaderData",
        3 => "BootCode",
        4 => "BootData",
        5 => "RuntimeCode",
        6 => "RuntimeData",
        7 => "Conventional",
        9 => "ACPIReclaim",
        10 => "ACPINVS",
        11 => "MMIO",
        12 => "MMIOPort",
        _ => "Unknown",
    }
}

fn halt() -> ! {
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}
