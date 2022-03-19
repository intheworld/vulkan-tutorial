cmake_minimum_required(VERSION 3.15)
project(vulkan_tutorial)

set(CMAKE_CXX_STANDARD 20)

add_executable(vulkan_tutorial main.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(vulkan_tutorial PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_tutorial PRIVATE Vulkan::Vulkan)

find_package(glfw3 REQUIRED)
target_link_libraries(vulkan_tutorial PRIVATE glfw)

#find_package(glm REQUIRED)
#target_link_libraries(vulkan_tutorial PRIVATE glm)