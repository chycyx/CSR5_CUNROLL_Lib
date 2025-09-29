#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "csr5_cunroll::csr5_cunroll" for configuration ""
set_property(TARGET csr5_cunroll::csr5_cunroll APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(csr5_cunroll::csr5_cunroll PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcsr5_cunroll.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS csr5_cunroll::csr5_cunroll )
list(APPEND _IMPORT_CHECK_FILES_FOR_csr5_cunroll::csr5_cunroll "${_IMPORT_PREFIX}/lib/libcsr5_cunroll.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
