"""
Compiling a project.
In the command line type:

>> python setup.py build

"""
from cx_Freeze import setup, Executable

build_exec_options = {"packages": {"os", "sys", "matplotlib"}, 
    "excludes": ["collections.sys", "collections._weakref"]}

setup(name='Vipper', version='0.2',
      description='Vipper Project',
      options = {"build_exe": build_exec_options},
      executables=[Executable('main.py')])
