use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

use ax_errno::{AxResult, ax_err};
use ax_kspin::SpinNoIrq as Mutex;
use axdevice_base::{AccessWidth, BaseDeviceOps, EmuDeviceType, Port, PortRange};

const PIC_MASTER_COMMAND_PORT: u16 = 0x20;
const PIC_MASTER_DATA_PORT: u16 = 0x21;
const PIC_SLAVE_COMMAND_PORT: u16 = 0xa0;
const PIC_SLAVE_DATA_PORT: u16 = 0xa1;
const PIC_MASTER_ELCR_PORT: u16 = 0x4d0;
const PIC_SLAVE_ELCR_PORT: u16 = 0x4d1;

const MASTER_ELCR_MASK: u8 = 0xf8;
const SLAVE_ELCR_MASK: u8 = 0xde;
static PIC_INIT_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
static PIC_EOI_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
struct PicChipState {
    master: bool,
    irr: u8,
    imr: u8,
    isr: u8,
    last_irr: u8,
    priority_add: u8,
    irq_base: u8,
    read_reg_select: bool,
    poll: bool,
    special_mask: bool,
    init_state: u8,
    auto_eoi: bool,
    rotate_on_auto_eoi: bool,
    special_fully_nested_mode: bool,
    init4: bool,
    single_mode: bool,
    ltim: bool,
    elcr: u8,
    elcr_mask: u8,
}

impl PicChipState {
    const fn new(master: bool, elcr_mask: u8) -> Self {
        Self {
            master,
            irr: 0,
            imr: 0,
            isr: 0,
            last_irr: 0,
            priority_add: 0,
            irq_base: 0,
            read_reg_select: false,
            poll: false,
            special_mask: false,
            init_state: 0,
            auto_eoi: false,
            rotate_on_auto_eoi: false,
            special_fully_nested_mode: false,
            init4: false,
            single_mode: false,
            ltim: false,
            elcr: 0,
            elcr_mask,
        }
    }

    fn reset_common(&mut self) {
        self.last_irr = 0;
        self.irr &= self.elcr;
        self.imr = 0;
        self.isr = 0;
        self.priority_add = 0;
        self.irq_base = 0;
        self.read_reg_select = false;
        self.poll = false;
        self.special_mask = false;
        self.init_state = 0;
        self.auto_eoi = false;
        self.rotate_on_auto_eoi = false;
        self.special_fully_nested_mode = false;
        self.init4 = false;
        self.single_mode = false;
    }

    fn get_priority(&self, mask: u8) -> u8 {
        if mask == 0 {
            return 8;
        }

        let mut priority = 0;
        while priority < 8
            && mask & (1 << ((priority as usize + self.priority_add as usize) & 7)) == 0
        {
            priority += 1;
        }
        priority
    }

    fn get_irq(&self) -> Option<u8> {
        let mask = self.irr & !self.imr;
        let priority = self.get_priority(mask);
        if priority == 8 {
            return None;
        }

        let mut in_service = self.isr;
        if self.special_mask {
            in_service &= !self.imr;
        }
        if self.master && self.special_fully_nested_mode {
            in_service &= !(1 << 2);
        }
        let current_priority = self.get_priority(in_service);
        (priority < current_priority)
            .then_some(((priority as usize + self.priority_add as usize) & 7) as u8)
    }

    fn intack(&mut self, irq: u8) {
        if self.auto_eoi {
            if self.rotate_on_auto_eoi {
                self.priority_add = (irq + 1) & 7;
            }
        } else {
            self.isr |= 1 << irq;
        }

        if !self.ltim && self.elcr & (1 << irq) == 0 {
            self.irr &= !(1 << irq);
        }
    }

    fn set_irq_line(&mut self, irq: u8, level: bool) {
        let mask = 1 << irq;
        if self.ltim || self.elcr & mask != 0 {
            if level {
                self.irr |= mask;
                self.last_irr |= mask;
            } else {
                self.irr &= !mask;
                self.last_irr &= !mask;
            }
            return;
        }

        if level {
            if self.last_irr & mask == 0 {
                self.irr |= mask;
            }
            self.last_irr |= mask;
        } else {
            self.last_irr &= !mask;
        }
    }

    fn command_write(&mut self, value: u8) {
        if value & 0x10 != 0 {
            self.reset_common();
            self.init_state = 1;
            self.init4 = value & 0x01 != 0;
            self.single_mode = value & 0x02 != 0;
            self.ltim = value & 0x08 != 0;
            if PIC_INIT_LOG_COUNT.fetch_add(1, Ordering::AcqRel) < 16 {
                info!(
                    "[VPIC] {} ICW1 init: single_mode={} init4={} ltim={}",
                    if self.master { "master" } else { "slave" },
                    self.single_mode,
                    self.init4,
                    self.ltim
                );
            }
            return;
        }

        if value & 0x08 != 0 {
            if value & 0x04 != 0 {
                self.poll = true;
            }
            if value & 0x02 != 0 {
                self.read_reg_select = value & 0x01 != 0;
            }
            if value & 0x40 != 0 {
                self.special_mask = value & 0x20 != 0;
            }
            return;
        }

        let command = value >> 5;
        match command {
            0 | 4 => {
                self.rotate_on_auto_eoi = command >> 2 != 0;
            }
            1 | 5 => {
                let isr_before = self.isr;
                let priority = self.get_priority(self.isr);
                if priority != 8 {
                    let irq = ((priority as usize + self.priority_add as usize) & 7) as u8;
                    self.isr &= !(1 << irq);
                    if command == 5 {
                        self.priority_add = (irq + 1) & 7;
                    }
                    if PIC_EOI_LOG_COUNT.fetch_add(1, Ordering::AcqRel) < 32 {
                        info!(
                            "[VPIC] {} OCW2 EOI cmd={command} irq={irq} isr: {isr_before:#x} -> \
                             {:#x}",
                            if self.master { "master" } else { "slave" },
                            self.isr
                        );
                    }
                }
            }
            3 => {
                let irq = value & 7;
                let isr_before = self.isr;
                self.isr &= !(1 << irq);
                if PIC_EOI_LOG_COUNT.fetch_add(1, Ordering::AcqRel) < 32 {
                    info!(
                        "[VPIC] {} OCW2 specific-EOI irq={irq} isr: {isr_before:#x} -> {:#x}",
                        if self.master { "master" } else { "slave" },
                        self.isr
                    );
                }
            }
            6 => {
                self.priority_add = (value + 1) & 7;
            }
            7 => {
                let irq = value & 7;
                let isr_before = self.isr;
                self.isr &= !(1 << irq);
                self.priority_add = (irq + 1) & 7;
                if PIC_EOI_LOG_COUNT.fetch_add(1, Ordering::AcqRel) < 32 {
                    info!(
                        "[VPIC] {} OCW2 rotate-specific-EOI irq={irq} isr: {isr_before:#x} -> \
                         {:#x}",
                        if self.master { "master" } else { "slave" },
                        self.isr
                    );
                }
            }
            _ => {}
        }
    }

    fn data_write(&mut self, value: u8) {
        match self.init_state {
            0 => {
                self.imr = value;
                info!(
                    "[VPIC] {} OCW1 imr={:#x}",
                    if self.master { "master" } else { "slave" },
                    self.imr
                );
            }
            1 => {
                self.irq_base = value & 0xf8;
                if PIC_INIT_LOG_COUNT.fetch_add(1, Ordering::AcqRel) < 16 {
                    info!(
                        "[VPIC] {} ICW2 irq_base={:#x}",
                        if self.master { "master" } else { "slave" },
                        self.irq_base
                    );
                }
                self.init_state = if self.single_mode {
                    if self.init4 { 3 } else { 0 }
                } else {
                    2
                };
            }
            2 => {
                self.init_state = if self.init4 { 3 } else { 0 };
            }
            3 => {
                self.special_fully_nested_mode = value & 0x10 != 0;
                self.auto_eoi = value & 0x02 != 0;
                self.init_state = 0;
            }
            _ => {}
        }
    }

    fn read(&mut self, command_port: bool) -> u8 {
        if self.poll {
            let value = self.get_irq().map_or(0, |irq| {
                self.intack(irq);
                irq | 0x80
            });
            self.poll = false;
            return value;
        }

        if command_port {
            if self.read_reg_select {
                self.isr
            } else {
                self.irr
            }
        } else {
            self.imr
        }
    }
}

#[derive(Debug)]
struct PicState {
    master: PicChipState,
    slave: PicChipState,
}

impl PicState {
    const fn new() -> Self {
        Self {
            master: PicChipState::new(true, MASTER_ELCR_MASK),
            slave: PicChipState::new(false, SLAVE_ELCR_MASK),
        }
    }

    fn update_cascade(&mut self) {
        let slave_pending = self.slave.get_irq().is_some();
        self.master.set_irq_line(2, slave_pending);
    }

    fn assert_irq(&mut self, irq: usize, level: bool) {
        if irq < 8 {
            self.master.set_irq_line(irq as u8, level);
        } else if irq < 16 {
            self.slave.set_irq_line((irq - 8) as u8, level);
        }
        self.update_cascade();
    }

    fn read_irq_vector(&mut self) -> Option<u8> {
        let master_irq = self.master.get_irq();
        info!(
            "[VPIC] read_irq_vector: master_irq={master_irq:?} master_irr={:#x} master_imr={:#x} \
             master_isr={:#x} master_last_irr={:#x}",
            self.master.irr, self.master.imr, self.master.isr, self.master.last_irr
        );
        let irq = master_irq?;
        if irq == 2 {
            let slave_irq = self.slave.get_irq().unwrap_or(7);
            self.slave.intack(slave_irq);
            self.master.intack(irq);
            self.update_cascade();
            return Some(self.slave.irq_base.wrapping_add(slave_irq));
        }

        self.master.intack(irq);
        self.update_cascade();
        Some(self.master.irq_base.wrapping_add(irq))
    }

    fn read_irq_vector_extint(&mut self) -> Option<u8> {
        let master_irq = self.master.get_irq();
        info!(
            "[VPIC] read_irq_vector_extint: master_irq={master_irq:?} master_irr={:#x} \
             master_imr={:#x} master_isr={:#x}",
            self.master.irr, self.master.imr, self.master.isr
        );
        let irq = master_irq?;
        if irq == 2 {
            let slave_irq = self.slave.get_irq().unwrap_or(7);
            // Clear IRR but do NOT set ISR (ExtINT transparent pass-through)
            self.slave.irr &= !(1 << slave_irq);
            self.master.irr &= !(1 << irq);
            self.update_cascade();
            return Some(self.slave.irq_base.wrapping_add(slave_irq));
        }
        // Clear IRR but do NOT set ISR
        self.master.irr &= !(1 << irq);
        self.update_cascade();
        Some(self.master.irq_base.wrapping_add(irq))
    }

    fn port_read(&mut self, port: u16) -> AxResult<u8> {
        match port {
            PIC_MASTER_COMMAND_PORT => Ok(self.master.read(true)),
            PIC_MASTER_DATA_PORT => Ok(self.master.read(false)),
            PIC_SLAVE_COMMAND_PORT => Ok(self.slave.read(true)),
            PIC_SLAVE_DATA_PORT => Ok(self.slave.read(false)),
            PIC_MASTER_ELCR_PORT => Ok(self.master.elcr),
            PIC_SLAVE_ELCR_PORT => Ok(self.slave.elcr),
            _ => ax_err!(Unsupported, "unsupported x86 PIC read port"),
        }
    }

    fn port_write(&mut self, port: u16, value: u8) -> AxResult {
        match port {
            PIC_MASTER_COMMAND_PORT => self.master.command_write(value),
            PIC_MASTER_DATA_PORT => self.master.data_write(value),
            PIC_SLAVE_COMMAND_PORT => self.slave.command_write(value),
            PIC_SLAVE_DATA_PORT => self.slave.data_write(value),
            PIC_MASTER_ELCR_PORT => self.master.elcr = value & self.master.elcr_mask,
            PIC_SLAVE_ELCR_PORT => self.slave.elcr = value & self.slave.elcr_mask,
            _ => return ax_err!(Unsupported, "unsupported x86 PIC write port"),
        }
        self.update_cascade();
        Ok(())
    }
}

/// Minimal 8259-compatible PIC pair used by the x86 timer bring-up path.
pub struct EmulatedPic8259 {
    state: Mutex<PicState>,
}

impl EmulatedPic8259 {
    /// Create a new master/slave 8259 PIC pair.
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(PicState::new()),
        }
    }

    /// Assert or deassert a legacy ISA IRQ line.
    pub fn assert_irq(&self, irq: usize, level: bool) {
        self.state.lock().assert_irq(irq, level);
    }

    /// Read and acknowledge the current PIC vector for an ExtINT delivery.
    pub fn read_irq_vector(&self) -> Option<u8> {
        self.state.lock().read_irq_vector()
    }

    /// Read the PIC vector for an ExtINT delivery without setting the PIC ISR.
    ///
    /// On real hardware, when the APIC handles the INTA cycle for ExtINT
    /// delivery, the PIC ISR is never set. The APIC transparently reads the
    /// vector from the PIC without issuing an INTA bus cycle to the PIC.
    /// This method matches that behavior: it reads the vector and clears IRR,
    /// but does NOT set ISR. This prevents the ISR-stuck problem where
    /// subsequent PIT ticks are rejected until the guest sends a PIC EOI.
    pub fn read_irq_vector_extint(&self) -> Option<u8> {
        self.state.lock().read_irq_vector_extint()
    }

    /// Diagnostic: return the master PIC ISR value.
    pub fn master_isr(&self) -> u8 {
        self.state.lock().master.isr
    }

    fn handle_port_read(&self, port: Port, width: AccessWidth) -> AxResult<usize> {
        if width != AccessWidth::Byte {
            return ax_err!(Unsupported, "x86 PIC only supports byte port reads");
        }
        Ok(self.state.lock().port_read(port.number())? as usize)
    }

    fn handle_port_write(&self, port: Port, width: AccessWidth, value: usize) -> AxResult {
        if width != AccessWidth::Byte {
            return ax_err!(Unsupported, "x86 PIC only supports byte port writes");
        }
        self.state.lock().port_write(port.number(), value as u8)
    }
}

impl Default for EmulatedPic8259 {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_pic_port_device {
    ($name:ident, $start:expr, $end:expr) => {
        /// PIC port-range wrapper backed by the shared 8259 state.
        pub struct $name {
            controller: Arc<EmulatedPic8259>,
        }

        impl $name {
            /// Create a new PIC port-range wrapper.
            pub fn new(controller: Arc<EmulatedPic8259>) -> Self {
                Self { controller }
            }
        }

        impl BaseDeviceOps<PortRange> for $name {
            fn emu_type(&self) -> EmuDeviceType {
                EmuDeviceType::InterruptController
            }

            fn address_range(&self) -> PortRange {
                PortRange::new(Port::new($start), Port::new($end))
            }

            fn handle_read(&self, port: Port, width: AccessWidth) -> AxResult<usize> {
                self.controller.handle_port_read(port, width)
            }

            fn handle_write(&self, port: Port, width: AccessWidth, value: usize) -> AxResult {
                self.controller.handle_port_write(port, width, value)
            }
        }
    };
}

impl_pic_port_device!(
    EmulatedPicMasterPort,
    PIC_MASTER_COMMAND_PORT,
    PIC_MASTER_DATA_PORT
);
impl_pic_port_device!(
    EmulatedPicSlavePort,
    PIC_SLAVE_COMMAND_PORT,
    PIC_SLAVE_DATA_PORT
);
impl_pic_port_device!(
    EmulatedPicElcrPort,
    PIC_MASTER_ELCR_PORT,
    PIC_SLAVE_ELCR_PORT
);

#[cfg(all(test, any(unix, windows)))]
mod tests {
    use super::EmulatedPic8259;

    fn init_pic(pic: &EmulatedPic8259) {
        pic.handle_port_write(super::Port::new(0x20), super::AccessWidth::Byte, 0x11)
            .unwrap();
        pic.handle_port_write(super::Port::new(0x21), super::AccessWidth::Byte, 0x20)
            .unwrap();
        pic.handle_port_write(super::Port::new(0x21), super::AccessWidth::Byte, 0x04)
            .unwrap();
        pic.handle_port_write(super::Port::new(0x21), super::AccessWidth::Byte, 0x01)
            .unwrap();

        pic.handle_port_write(super::Port::new(0xa0), super::AccessWidth::Byte, 0x11)
            .unwrap();
        pic.handle_port_write(super::Port::new(0xa1), super::AccessWidth::Byte, 0x28)
            .unwrap();
        pic.handle_port_write(super::Port::new(0xa1), super::AccessWidth::Byte, 0x02)
            .unwrap();
        pic.handle_port_write(super::Port::new(0xa1), super::AccessWidth::Byte, 0x01)
            .unwrap();
    }

    #[test]
    fn master_irq0_extint_vector_follows_irq_base() {
        let pic = EmulatedPic8259::new();
        init_pic(&pic);

        pic.assert_irq(0, true);
        pic.assert_irq(0, false);

        assert_eq!(pic.read_irq_vector(), Some(0x20));
    }

    #[test]
    fn slave_irq8_cascades_through_master_irq2() {
        let pic = EmulatedPic8259::new();
        init_pic(&pic);

        pic.assert_irq(8, true);
        pic.assert_irq(8, false);

        assert_eq!(pic.read_irq_vector(), Some(0x28));
    }
}
