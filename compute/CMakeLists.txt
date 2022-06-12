add_executable(ComputeShader computeShader.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(ComputeShader PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(ComputeShader PRIVATE Vulkan::Vulkan)