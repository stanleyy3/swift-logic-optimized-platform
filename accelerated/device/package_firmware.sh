#!/usr/bin/env bash
# package_firmware.sh -- build the /lib/firmware/xilinx/<app> bundle that
# dfx-mgrd needs to load an xclbin on Certified Ubuntu for Kria.
#
# Why this exists: the PetaLinux tutorials say to rename the .xclbin to .bin and
# drop it in /lib/firmware/xilinx/<app>. That is not sufficient on Ubuntu.
# UG1630 requires the PL image in *.bit.bin format (produced by bootgen from the
# bitstream inside the xclbin), and requires every file in the directory to be
# named after <app>. A name mismatch is the usual reason `xmutil loadapp` fails.
#
# Usage:
#   ./package_firmware.sh <xclbin> <app_name> [out_dir]
#
# Produces <out_dir>/<app_name>/ containing:
#   <app_name>.bit.bin   PL image, converted from the xclbin's BITSTREAM section
#   <app_name>.dtbo      device tree overlay (carries the zocl node)
#   <app_name>.xclbin    the xclbin itself, for XRT
#   shell.json           tells dfx-mgr this is a flat XRT design
#
# Then, on the board:
#   sudo cp -r <app_name> /lib/firmware/xilinx/
#   sudo systemctl restart dfx-mgrd      # inotify misses dirs created under it
#   sudo xmutil listapps                 # <app_name> should appear as XRT_FLAT
#   sudo xmutil unloadapp
#   sudo xmutil loadapp <app_name>
#
# If xmutil still refuses, fpgautil bypasses dfx-mgrd entirely:
#   sudo fpgautil -b <app_name>.bit.bin -o <app_name>.dtbo

set -euo pipefail

XCLBIN=${1:?usage: package_firmware.sh <xclbin> <app_name> [out_dir]}
APP=${2:?usage: package_firmware.sh <xclbin> <app_name> [out_dir]}
OUT_ROOT=${3:-$(dirname "$0")/build/firmware}

# The base platform ships the overlay that declares the zocl (zyxclmm_drm) node.
# It is generic to the platform, not to the kernel, so it is reused as-is.
PLATFORM_DIR=${PLATFORM_DIR:-/afs/ece.cmu.edu/support/xilinx/xilinx.release/Vivado-2025.2/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1}
PL_DTBO=$PLATFORM_DIR/sw/boot/pl.dtbo

SHELL_JSON=$(dirname "$0")/../host/tests/kv260_platform_test/shell.json

for f in "$XCLBIN" "$PL_DTBO" "$SHELL_JSON"; do
    [ -r "$f" ] || { echo "package_firmware.sh: cannot read $f" >&2; exit 1; }
done

DEST=$OUT_ROOT/$APP
rm -rf "$DEST"
mkdir -p "$DEST"
XCLBIN_ABS=$(readlink -f "$XCLBIN")

cp "$XCLBIN_ABS" "$DEST/$APP.xclbin"
cp "$SHELL_JSON" "$DEST/shell.json"

# The platform's stock pl.dtbo ships `firmware-name = ".bin"` -- an unfilled
# placeholder with an empty basename. The fpga-region driver loads whatever that
# property names, so left alone the overlay applies against a file that does not
# exist and the load fails. Rewrite it to the .bit.bin this bundle contains.
#
# dtc needs -f plus the three -Wno- flags: the platform's fragments carry `reg`
# properties whose cell counts do not match the overlay root, which dtc treats
# as a fatal prerequisite failure rather than a warning. The round trip is
# otherwise lossless -- __symbols__, __fixups__ and __local_fixups__ all survive,
# which matters because the overlay cannot be applied without them.
dtc -I dtb -O dts "$PL_DTBO" -o "$DEST/$APP.dts" 2>/dev/null
sed -i "s|firmware-name = \"[^\"]*\";|firmware-name = \"$APP.bit.bin\";|" "$DEST/$APP.dts"
dtc -I dts -O dtb -f -Wno-reg_format -Wno-unit_address_vs_reg -Wno-pci_device_reg \
    -o "$DEST/$APP.dtbo" "$DEST/$APP.dts" 2>/dev/null
rm -f "$DEST/$APP.dts"

[ -s "$DEST/$APP.dtbo" ] || { echo "package_firmware.sh: failed to build $APP.dtbo" >&2; exit 1; }

# fail loudly rather than shipping an overlay that silently points nowhere
if ! dtc -I dtb -O dts "$DEST/$APP.dtbo" 2>/dev/null | grep -q "firmware-name = \"$APP.bit.bin\";"; then
    echo "package_firmware.sh: $APP.dtbo does not name $APP.bit.bin" >&2
    exit 1
fi
if ! dtc -I dtb -O dts "$DEST/$APP.dtbo" 2>/dev/null | grep -q 'xlnx,zocl'; then
    echo "package_firmware.sh: $APP.dtbo lost its zocl node in the dtc round trip" >&2
    exit 1
fi

# bootgen resolves the .bit path relative to its own working directory, so the
# extraction and the conversion both happen inside $DEST
(
    cd "$DEST"
    xclbinutil --quiet --input "$APP.xclbin" --dump-section "BITSTREAM:RAW:$APP.bit"
    printf 'all: { %s.bit }\n' "$APP" > "$APP.bif"
    bootgen -image "$APP.bif" -arch zynqmp -process_bitstream bin -w
    rm -f "$APP.bit" "$APP.bif"
)

[ -s "$DEST/$APP.bit.bin" ] || { echo "package_firmware.sh: bootgen produced no $APP.bit.bin" >&2; exit 1; }

echo
echo "firmware bundle ready: $DEST"
ls -la "$DEST"
