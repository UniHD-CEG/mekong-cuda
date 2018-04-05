
find_path(CUDAH_INCLUDE_DIR cuda.h HINTS /usr/local/cuda PATH_SUFFIXES include)

find_package_handle_standard_args(
  CUDAH
  REQUIRED_VARS CUDAH_INCLUDE_DIR)

if(CUDAH_FOUND)
  set(CUDAH_INCLUDE_DIRS ${CUDAH_INCLUDE_DIR})
endif()

mark_as_advanced(CUDAH_INCLUDE_DIR)
