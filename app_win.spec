# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

dir = 'C:\\Users\\Masami\\Documents\\Python\\masamip2_proj_fn'
model = 'model.h5'
model_pad_fill = 'model_pad_fill.h5'
models = [(model, '.'), (model_pad_fill, '.')]


from PyInstaller.utils.hooks import collect_submodules, collect_data_files
tf_hidden_imports = collect_submodules('tensorflow_core')
tf_datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)
a_hidden_imports = collect_submodules('astor')
a_datas = collect_data_files('astor', subdir=None, include_py_files=True)

a = Analysis(['app.py'],
             pathex=[dir],
             binaries=[],
             datas=tf_datas + a_datas + models,
             hiddenimports=tf_hidden_imports + a_hidden_imports + ['pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
