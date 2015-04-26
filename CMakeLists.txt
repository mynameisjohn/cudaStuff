cmake_minimum_required(VERSION 2.6)

project (MICROBENCHMARK)

find_package(CUDA QUIET REQUIRED)

add_executable(
	main
	main.cpp
)

cuda_add_library(
	gsRelax
	gsRelax.cu
	gsRelax.h
)

cuda_add_library(
	misc
	misc.cu
	misc.h
)

target_link_libraries(
	main
	gsRelax
	misc
)
