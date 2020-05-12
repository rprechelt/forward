import os
import sys
import subprocess

from setuptools import Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """
    This class represents a CMake-built extension.
    """
    def __init__(self, name, sourcedir=''):
        """
        Create a new CMakeExtension.

        Parameters
        ----------
        name: str
            The name of the extension to build.
        sourcedir: str
            The location of the root CMakeLists.txt
        """
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """
    This class is responsible for building extensions with CMake.
    """
    def run(self):
        try:
            # get the CMake version
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            # and report an error if we can't find CMake
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # build each extension individually
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        # get the path to this extension
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # construct our cmake arguments
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DWITH_PYTHON=ON']

        # get a copy of the local environment variables
        env = os.environ.copy()

        # do we want a debug or build release
        cfg = 'Debug' if self.debug else 'Release'

        # set build configuration
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # use multiple threads if we can
        build_args += ['--', f"-j{env.get('NTHREADS', 2)}"]

        # set the C++ version
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        # if the build directory does not exist, create it
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # call cmake to setup projecet
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)

        # and build the project
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
