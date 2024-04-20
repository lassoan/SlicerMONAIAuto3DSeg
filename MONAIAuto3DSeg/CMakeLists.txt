#-----------------------------------------------------------------------------
set(MODULE_NAME MONAIAuto3DSeg)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Models.json
  Resources/AnatomicRegionAndModifier-MONAIAuto3DSeg.term.json
  Resources/SegmentationCategoryTypeModifier-MONAIAuto3DSeg.term.json
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/BraTS-GLI-00001-000.jpg
  Resources/Icons/BraTS-MEN-00000-000.jpg
  Resources/Icons/BraTS-MET-00002-000.jpg
  Resources/Icons/BraTS-PED-00030-000.jpg
  Resources/Icons/BraTS-SSA-00002-000.jpg
  Resources/Icons/filter.svg
  Resources/Icons/ICH-ADAPT2.jpg
  Resources/Icons/msd-prostate-01.jpg
  Resources/Icons/ProstateX-0000.jpg
  Resources/Icons/radiology.svg
  Resources/UI/${MODULE_NAME}.ui
  Scripts/auto3dseg_segresnet_inference.py
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
