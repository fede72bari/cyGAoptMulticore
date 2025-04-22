cmake_minimum_required(VERSION 3.12)
project(cyGAoptMultiCore)

# open "Developer Command Prompt for VS 2022"
# mkdir build
# cd build
# cmake ..
# cmake --build . --clean-first --config Release
# in case of troubles localizing pybind11: python -c "import pybind11; print(pybind11.get_cmake_dir())"

cmake_minimum_required(VERSION 3.12)
project(cyGAoptMultiCore)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

# Abilita OpenMP se disponibile
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling OpenMP support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Compila entrambi i file nel modulo
pybind11_add_module(cyGAoptMultiCore cyGAoptMultiCore.cpp)
