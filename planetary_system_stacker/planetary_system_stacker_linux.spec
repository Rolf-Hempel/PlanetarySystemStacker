# -*- mode: python -*-

from pathlib import Path

block_cipher = None
s_dir = Path('/home/castor/shares/castor/src/PlanetarySystemStacker')

a = Analysis(['planetary_system_stacker.py'],
             pathex=[s_dir / 'planetary_system_stacker'],
             binaries=[],
             datas=[(s_dir / 'Documentation/Icon/PSS-Icon-64.jpg', '.' ),
             (s_dir / 'Documentation/PlanetarySystemStacker_User-Guide.pdf', '.' ),
             (s_dir / 'planetary_system_stacker/Videos/*', 'Videos' ),
		],
             hiddenimports=['pywt._extensions._cwt' ],
             hookspath=[s_dir / 'planetary_system_stacker/Pyinstaller_hooks'],
             runtime_hooks=[],
	     excludes=[],
             # excludes=['astropy'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='planetary_system_stacker',
          debug=False,
          strip=False,
          upx=True,
          console=False )   # To display a console window, change value to True.
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='PlanetarySystemStacker')
