# -*- mode: python -*-

block_cipher = None

from pathlib import Path

p_dir = Path(os.getenv('CONDA_PREFIX', 'C:\Python39'))
s_dir = Path('D:\\SW-Development\\Python\\PlanetarySystemStacker')

a = Analysis(['planetary_system_stacker.py'],
             pathex=[s_dir / 'planetary_system_stacker'],
             binaries=[
              (s_dir / 'planetary_system_stacker\Binaries\Api-ms-win-core-xstate-l2-1-0.dll', '.'),
              (s_dir / 'planetary_system_stacker\Binaries\Api-ms-win-crt-private-l1-1-0.dll', '.'),
              ('C:\Windows\System32\downlevel\API-MS-Win-Eventing-Provider-L1-1-0.dll', '.'),
              (s_dir / 'planetary_system_stacker\Binaries\\api-ms-win-downlevel-shlwapi-l1-1-0.dll', '.')],
             datas=[(s_dir / 'Documentation\\Icon\\PSS-Icon-64.ico', '.' ),
              (s_dir / 'Documentation\\Icon\\PSS-Icon-64.png', '.' ),
             ],
             hiddenimports=['pywt._extensions._cwt', #'scipy._lib.messagestream', 'shelve', 'csv', 'pkg_resources.py2_warn'
             ],
             hookspath=[s_dir / 'planetary_system_stacker\Pyinstaller_hooks'],
             runtime_hooks=[],
             excludes=[
                ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='planetary_system_stacker',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )    # To display a console window, change value to True.
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='PlanetarySystemStacker')
