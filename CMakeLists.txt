project(ZED_Depth_Refocusing)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)


if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")


IF(WIN32) # Windows
    SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
    SET(GLEW_LIBRARY_DIRS $ENV{GLEW_LIBRARY_DIRS})
    SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
    SET(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIRS})
ENDIF()

## Required package
find_package(ZED REQUIRED)
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)


## Set include directories
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

## Set link directories
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

## Find sources
SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")
FILE(GLOB_RECURSE SRC_CU_FOLDER "${SRC_FOLDER}/*.cu")

## Set definitions
add_definitions(-std=c++14 -O3)

## Set arch/compute flag for CUDA src
if (CMAKE_SYSTEM_NAME2 MATCHES "aarch64")
    IF ((${CUDA_VERSION_MAJOR} VERSION_GREATER 10) OR (${CUDA_VERSION_MAJOR} VERSION_EQUAL 10)) # JP 4
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_62)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_53)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_72)
    ELSE() # JP 3
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_62)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_53)
    ENDIF()

ELSE()
    IF (${CUDA_VERSION_MAJOR} VERSION_LESS 11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_32)
    ENDIF()

        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_52)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_61)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)

    IF ((${CUDA_VERSION_MAJOR} VERSION_GREATER 10) OR (${CUDA_VERSION_MAJOR} VERSION_EQUAL 10))
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_75)
    ENDIF()
    IF ((${CUDA_VERSION_MAJOR} VERSION_GREATER 11) OR (${CUDA_VERSION_MAJOR} VERSION_EQUAL 11))
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80)
    ENDIF()
    IF ((${CUDA_VERSION_MAJOR} VERSION_GREATER 11) OR ((${CUDA_VERSION_MAJOR} VERSION_EQUAL 11) AND (${CUDA_VERSION_MINOR} VERSION_GREATER 0)))
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_86)
    ENDIF()
ENDIF()

## Add sources in executable
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
cuda_add_executable(${PROJECT_NAME} ${SRC_FILES} ${SRC_CU_FOLDER})



# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    ${ZED_LIBRARIES}
    ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
    ${SPECIAL_OS_LIBS}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )

##<<->
