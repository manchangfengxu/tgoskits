EDK2_DIR := $(HOME)/edk2
BUILD_DIR := $(EDK2_DIR)/Build/OvmfX64/DEBUG_GCC/FV
ROOTFS := $(HOME)/tgoskits/tmp/axbuild/rootfs/rootfs-x86_64-alpine.img
MNT := /tmp/axvisor-ovmf-mnt

OVMF_CODE := $(BUILD_DIR)/OVMF_CODE.fd
OVMF_VARS := $(BUILD_DIR)/OVMF_VARS.fd

QEMU_CONFIG := $(CURDIR)/tmp/configs/qemu-x86_64.toml
QEMU_RT_CONFIG := $(CURDIR)/tmp/configs/qemu-x86_64-runtime.toml
VM_CONFIG := $(CURDIR)/tmp/configs/ovmf-x86_64-qemu-smp1.toml

.PHONY: ovmf deploy run all clean-mount

all: ovmf deploy run

ovmf:
	cd $(EDK2_DIR) && bash -c 'source edksetup.sh && build -a X64 -t GCC -p OvmfPkg/OvmfPkgX64.dsc -b DEBUG'

deploy: $(OVMF_CODE)
	@mkdir -p $(MNT)
	sudo mount $(ROOTFS) $(MNT)
	sudo mkdir -p $(MNT)/guest/ovmf
	sudo cp $(OVMF_CODE) $(MNT)/guest/ovmf/
	sudo cp $(OVMF_VARS) $(MNT)/guest/ovmf/
	sudo umount $(MNT)
	@echo "OVMF deployed to rootfs."

run:
	cargo xtask qemu --config $(QEMU_CONFIG) --qemu-config $(QEMU_RT_CONFIG) --vmconfigs $(VM_CONFIG)

clean-mount:
	-sudo umount $(MNT) 2>/dev/null || true
