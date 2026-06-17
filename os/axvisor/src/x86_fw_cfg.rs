use alloc::{format, vec::Vec};

use ax_errno::{AxResult, ax_err};
use axdevice::fw_cfg::{
    FW_CFG_IO_DATA, FW_CFG_IO_SELECTOR, FwCfgFileDirEntry, QEMU_FW_CFG_ITEM_FILE_DIR,
    parse_file_directory,
};
use spin::Once;

const ACPI_TABLES_FILE: &str = "etc/acpi/tables";
const ACPI_RSDP_FILE: &str = "etc/acpi/rsdp";
const ACPI_TABLE_LOADER_FILE: &str = "etc/table-loader";
const FW_CFG_FILE_DIR_ENTRY_SIZE: usize = 4 + 2 + 2 + 56;
const MAX_FW_CFG_FILE_DIR_ENTRIES: usize = 4096;

static OUTER_QEMU_ACPI_FW_CFG_BLOBS: Once<OuterQemuAcpiFwCfgBlobs> = Once::new();

#[derive(Clone, Debug)]
pub struct OuterQemuAcpiFwCfgBlobs {
    tables: Vec<u8>,
    rsdp: Vec<u8>,
    table_loader: Vec<u8>,
}

impl OuterQemuAcpiFwCfgBlobs {
    pub fn entries(&self) -> [(&'static str, &[u8]); 3] {
        [
            (ACPI_TABLES_FILE, self.tables.as_slice()),
            (ACPI_RSDP_FILE, self.rsdp.as_slice()),
            (ACPI_TABLE_LOADER_FILE, self.table_loader.as_slice()),
        ]
    }
}

pub fn cached_outer_qemu_acpi_fw_cfg_blobs() -> AxResult<&'static OuterQemuAcpiFwCfgBlobs> {
    OUTER_QEMU_ACPI_FW_CFG_BLOBS.try_call_once(read_outer_qemu_acpi_fw_cfg_blobs)
}

fn read_outer_qemu_acpi_fw_cfg_blobs() -> AxResult<OuterQemuAcpiFwCfgBlobs> {
    let file_dir = read_outer_fw_cfg_file_directory()?;
    let entries = parse_file_directory(&file_dir)?;
    let blobs = collect_required_acpi_blobs(&entries, |entry| {
        select_fw_cfg_item(entry.selector);
        Ok(read_fw_cfg_bytes(entry.size))
    })?;
    info!(
        "Loaded outer QEMU ACPI fw_cfg blobs: tables={} bytes, rsdp={} bytes, table-loader={} bytes",
        blobs.tables.len(),
        blobs.rsdp.len(),
        blobs.table_loader.len()
    );
    Ok(blobs)
}

fn read_outer_fw_cfg_file_directory() -> AxResult<Vec<u8>> {
    select_fw_cfg_item(QEMU_FW_CFG_ITEM_FILE_DIR);
    let header = read_fw_cfg_bytes(4);
    let count = u32::from_be_bytes(header.as_slice().try_into().unwrap()) as usize;
    if count > MAX_FW_CFG_FILE_DIR_ENTRIES {
        return ax_err!(
            InvalidData,
            format!("outer fw_cfg file directory entry count is too large: {count}")
        );
    }
    let Some(entry_bytes) = count.checked_mul(FW_CFG_FILE_DIR_ENTRY_SIZE) else {
        return ax_err!(
            InvalidInput,
            "outer fw_cfg file directory entry count overflow"
        );
    };
    let mut dir = header;
    dir.extend_from_slice(&read_fw_cfg_bytes(entry_bytes));
    Ok(dir)
}

fn collect_required_acpi_blobs<F>(
    entries: &[FwCfgFileDirEntry],
    mut read_entry: F,
) -> AxResult<OuterQemuAcpiFwCfgBlobs>
where
    F: FnMut(&FwCfgFileDirEntry) -> AxResult<Vec<u8>>,
{
    Ok(OuterQemuAcpiFwCfgBlobs {
        tables: read_named_fw_cfg_file(entries, ACPI_TABLES_FILE, &mut read_entry)?,
        rsdp: read_named_fw_cfg_file(entries, ACPI_RSDP_FILE, &mut read_entry)?,
        table_loader: read_named_fw_cfg_file(entries, ACPI_TABLE_LOADER_FILE, &mut read_entry)?,
    })
}

fn read_named_fw_cfg_file<F>(
    entries: &[FwCfgFileDirEntry],
    name: &str,
    read_entry: &mut F,
) -> AxResult<Vec<u8>>
where
    F: FnMut(&FwCfgFileDirEntry) -> AxResult<Vec<u8>>,
{
    let Some(entry) = entries.iter().find(|entry| entry.name == name) else {
        return ax_err!(NotFound, format!("outer fw_cfg file {name} is missing"));
    };
    read_entry(entry)
}

fn select_fw_cfg_item(selector: u16) {
    unsafe {
        x86::io::outw(FW_CFG_IO_SELECTOR, selector);
    }
}

fn read_fw_cfg_bytes(len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        bytes.push(unsafe { x86::io::inb(FW_CFG_IO_DATA) });
    }
    bytes
}

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, string::String, vec};

    use super::{
        ACPI_RSDP_FILE, ACPI_TABLE_LOADER_FILE, ACPI_TABLES_FILE, FwCfgFileDirEntry,
        collect_required_acpi_blobs,
    };

    #[test]
    fn collect_required_acpi_blobs_reads_all_three_files() {
        let entries = vec![
            FwCfgFileDirEntry {
                size: 3,
                selector: 0x20,
                name: String::from(ACPI_TABLES_FILE),
            },
            FwCfgFileDirEntry {
                size: 2,
                selector: 0x21,
                name: String::from(ACPI_RSDP_FILE),
            },
            FwCfgFileDirEntry {
                size: 4,
                selector: 0x22,
                name: String::from(ACPI_TABLE_LOADER_FILE),
            },
        ];
        let payloads = BTreeMap::from([
            (0x20, b"tab".to_vec()),
            (0x21, b"rs".to_vec()),
            (0x22, b"load".to_vec()),
        ]);

        let blobs = collect_required_acpi_blobs(&entries, |entry| {
            Ok(payloads.get(&entry.selector).cloned().unwrap())
        })
        .unwrap();

        assert_eq!(
            blobs.entries(),
            [
                (ACPI_TABLES_FILE, b"tab".as_slice()),
                (ACPI_RSDP_FILE, b"rs".as_slice()),
                (ACPI_TABLE_LOADER_FILE, b"load".as_slice()),
            ]
        );
    }

    #[test]
    fn collect_required_acpi_blobs_rejects_missing_required_file() {
        let entries = vec![
            FwCfgFileDirEntry {
                size: 3,
                selector: 0x20,
                name: String::from(ACPI_TABLES_FILE),
            },
            FwCfgFileDirEntry {
                size: 2,
                selector: 0x21,
                name: String::from(ACPI_RSDP_FILE),
            },
        ];

        let err = collect_required_acpi_blobs(&entries, |_entry| Ok(Vec::new())).unwrap_err();

        assert_eq!(err, ax_errno::AxError::NotFound);
        assert!(format!("{err}").contains(ACPI_TABLE_LOADER_FILE));
    }
}
