# -*- mode: python -*-

block_cipher = None

# Integrate astropy as data directory instead of module:
import astropy
astropy_path, = astropy.__path__

a = Analysis(['planetary_system_stacker.py'],
             pathex=['D:\\SW-Development\\Python\\PlanetarySystemStacker\\planetary_system_stacker'],
             binaries=[('C:\Python39\Lib\site-packages\cv2\opencv_videoio_ffmpeg454_64.dll', '.'),
              ('D:\SW-Development\Python\PlanetarySystemStacker\planetary_system_stacker\Binaries\Api-ms-win-core-xstate-l2-1-0.dll', '.'),
              ('D:\SW-Development\Python\PlanetarySystemStacker\planetary_system_stacker\Binaries\Api-ms-win-crt-private-l1-1-0.dll', '.'),
              ('C:\Windows\System32\downlevel\API-MS-Win-Eventing-Provider-L1-1-0.dll', '.'),
              ('D:\SW-Development\Python\PlanetarySystemStacker\planetary_system_stacker\Binaries\\api-ms-win-downlevel-shlwapi-l1-1-0.dll', '.')],
             datas=[( 'D:\\SW-Development\\Python\\PlanetarySystemStacker\\Documentation\\Icon\\PSS-Icon-64.ico', '.' ),
             ( 'D:\\SW-Development\\Python\\PlanetarySystemStacker\\Documentation\\Icon\\PSS-Icon-64.png', '.' ),
             (astropy_path, 'astropy')],
             hiddenimports=['pywt._extensions._cwt', 'scipy._lib.messagestream', 'shelve', 'csv', 'pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['astropy'],
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
