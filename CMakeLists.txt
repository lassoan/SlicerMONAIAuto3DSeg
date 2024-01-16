cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MONAIAuto3DSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/lassoan/SlicerMONAIAuto3DSeg")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Andras Lasso (PerkLab, Queen's University)" "Andres Diaz Pinto (PerkLab, Queen's University)" "Rudolf Bumm (KSGR Switzerland)")
set(EXTENSION_DESCRIPTION "Run MONAI Auto3DSeg fully automatic AI segmentation in 3D Slicer.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/lassoan/SlicerMONAIAuto3DSeg/main/MONAIAuto3DSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/lassoan/SlicerMONAIAuto3DSeg/main/Screenshot01.jpg")
set(EXTENSION_DEPENDS "PyTorch")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MONAIAuto3DSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
