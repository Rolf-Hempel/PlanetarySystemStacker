# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for building the macOS .app bundle.
# Build with:   pyinstaller PlanetarySystemStacker.spec


a = Analysis(
    ['planetary_system_stacker/planetary_system_stacker.py'],
    pathex=['planetary_system_stacker'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_rthook_cv2.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlanetarySystemStacker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Documentation/Icon/PSS-Icon-128.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlanetarySystemStacker',
)
app = BUNDLE(
    coll,
    name='PlanetarySystemStacker.app',
    icon='Documentation/Icon/PSS-Icon-128.png',
    bundle_identifier='com.timing.PlanetarySystemStacker',
)
