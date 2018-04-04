# External: Integer Set Library
set(ISL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ISL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Determine version of isl
if (EXISTS "${ISL_SOURCE_DIR}/GIT_HEAD_ID")
  # The source comes from a 'make dist' archive
  file(READ "${ISL_SOURCE_DIR}/GIT_HEAD_ID" GIT_HEAD_ID)
  string(STRIP "${GIT_HEAD_ID}" GIT_HEAD_ID)
elseif (EXISTS "${ISL_SOURCE_DIR}/gitversion.h")
  # The source directory is preconfigured
  file(READ "${ISL_SOURCE_DIR}/gitversion.h" GITVERSION_H)
  string(REGEX REPLACE ".*\\\"([^\\\"]*)\\\".*" "\\1" GIT_HEAD_ID "${GITVERSION_H}")
elseif ()
  # Unknown revision
  # TODO: We could look for a .git and get the revision from HEAD
  set(GIT_HEAD_ID "UNKNOWN")
endif ()

message(STATUS "ISL version: ${GIT_HEAD_ID}")

# Enable small integer optimization and imath
set(USE_GMP_FOR_MP OFF)
set(USE_IMATH_FOR_MP ON)
set(USE_SMALL_INT_OPT ON)

# Determine compiler characteristics
include(CheckCSourceCompiles)

# Like check_c_source_compiles, but sets the result to either
# 0 (error while compiling) or 1 (compiled successfully)
# Required for compatibility with autotool's AC_CHECK_DECLS
function (check_c_source_compiles_numeric _prog _var)
  check_c_source_compiles("${_prog}" "${_var}")
  if ("${${_var}}")
    set("${_var}" 1 PARENT_SCOPE)
  else ()
    set("${_var}" 0 PARENT_SCOPE)
  endif ()
endfunction ()

# Check for the existance of a type
function (check_c_type_exists _type _files _variable)
  set(_includes "")
  foreach (file_name ${_files})
     set(_includes "${_includes}#include<${file_name}>\n")
  endforeach()
  check_c_source_compiles("
    ${_includes}
    ${_type} typeVar;
    int main() {
        return 0;
    }
    " ${_variable})
endfunction ()


check_c_source_compiles("
  int func(void) __attribute__((__warn_unused_result__));
  int main() { return 0; }
  " HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
set(GCC_WARN_UNUSED_RESULT)
if (HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
  set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
endif ()

check_c_source_compiles("
  static void foo(void) __attribute__ ((unused));
  int main() { return 0; }
  " HAVE___ATTRIBUTE__)


check_c_source_compiles_numeric("
  #include <strings.h>
  int main() { (void)ffs(0); return 0; }
  " HAVE_DECL_FFS)

check_c_source_compiles_numeric("
  int main() { __builtin_ffs(0); return 0; }
  " HAVE_DECL___BUILTIN_FFS)

check_c_source_compiles_numeric("
  #include <intrin.h>
  int main() { _BitScanForward(NULL, 0); return 0; }
  " HAVE_DECL__BITSCANFORWARD)

if (NOT HAVE_DECL_FFS AND
    NOT HAVE_DECL___BUILTIN_FFS AND
    NOT HAVE_DECL__BITSCANFORWARD)
  message(FATAL_ERROR "No ffs implementation found")
endif ()


check_c_source_compiles_numeric("
  #include <strings.h>
  int main() { (void)strcasecmp(\"\", \"\"); return 0; }
  " HAVE_DECL_STRCASECMP)

check_c_source_compiles_numeric("
  #include <string.h>
  int main() { _stricmp(\"\", \"\"); return 0; }
  " HAVE_DECL__STRICMP)

if (NOT HAVE_DECL_STRCASECMP AND NOT HAVE_DECL__STRICMP)
  message(FATAL_ERROR "No strcasecmp implementation found")
endif ()


check_c_source_compiles_numeric("
  #include <strings.h>
  int main() { (void)strncasecmp(\"\", \"\", 0); return 0; }
  " HAVE_DECL_STRNCASECMP)

check_c_source_compiles_numeric("
  #include <string.h>
  int main() { _strnicmp(\"\", \"\", 0); return 0; }
  " HAVE_DECL__STRNICMP)

if (NOT HAVE_DECL_STRNCASECMP AND NOT HAVE_DECL__STRNICMP)
  message(FATAL_ERROR "No strncasecmp implementation found")
endif ()


check_c_source_compiles_numeric("
  #include <stdio.h>
  int main() { snprintf((void*)0, 0, \"\"); return 0; }
  " HAVE_DECL_SNPRINTF)

check_c_source_compiles_numeric("
  #include <stdio.h>
  int main() { _snprintf((void*)0, 0, \"\"); return 0; }
  " HAVE_DECL__SNPRINTF)

if (NOT HAVE_DECL_SNPRINTF AND NOT HAVE_DECL__SNPRINTF)
  message(FATAL_ERROR "No snprintf implementation found")
endif ()


# Write configure result
# configure_file(... COPYONLY) avoids that the time stamp changes if the file is identical
file(WRITE "${ISL_BINARY_DIR}/gitversion.h.tmp"
     "#define GIT_HEAD_ID \"${GIT_HEAD_ID}\"")
configure_file("${ISL_BINARY_DIR}/gitversion.h.tmp"
               "${ISL_BINARY_DIR}/gitversion.h" COPYONLY)

configure_file("isl_config.h.cmake" "${ISL_BINARY_DIR}/isl_config.h")

include_directories(BEFORE
  ${ISL_BINARY_DIR}/include
  ${ISL_SOURCE_DIR}/imath
  ${ISL_SOURCE_DIR}
)

set (ISL_INTERNAL_FILES
  basis_reduction_templ.c
  bound.c
  cat.c
  closure.c
  codegen.c
  isl_ast_int.c
  isl_gmp.c
  isl_list_templ.c
  isl_map_lexopt_templ.c
  isl_multi_apply_set.c
  isl_multi_apply_templ.c
  isl_multi_apply_union_set.c
  isl_multi_cmp.c
  isl_multi_coalesce.c
  isl_multi_floor.c
  isl_multi_gist.c
  isl_multi_hash.c
  isl_multi_intersect.c
  isl_multi_templ.c
  isl_power_templ.c
  isl_pw_hash.c
  isl_pw_templ.c
  isl_pw_union_opt.c
  isl_test.c
  isl_test_imath.c
  isl_test_int.c
  isl_union_eval.c
  isl_union_multi.c
  isl_union_neg.c
  isl_union_single.c
  isl_union_templ.c
  isl_val_gmp.c
  isl_val_imath.c
  mp_get_memory_functions.c
  polyhedron_detect_equalities.c
  polyhedron_minimize.c
  polyhedron_sample.c
  polytope_scan.c
  print_templ.c
  set_to_map.c
  schedule.c
  pip.c
  isl_tab_lexopt_templ.c
  flow_cmp.c
  flow.c
  extract_key.c
  read_in_string_templ.c
  print_templ_yaml.c
  schedule_cmp.c
  set_from_map.c
  bset_to_bmap.c
  bset_from_bmap.c
)

set(LLVM_OPTIONAL_SOURCES ISLInternal ${ISL_INTERNAL_FILES})

# ISL files to compile
set (ISL_FILES
    basis_reduction_tab.c
    isl_aff.c
    isl_affine_hull.c
    isl_arg.c
    isl_ast_build.c
    isl_ast_build_expr.c
    isl_ast.c
    isl_ast_codegen.c
    isl_ast_graft.c
    isl_band.c
    isl_bernstein.c
    isl_blk.c
    isl_bound.c
    isl_coalesce.c
    isl_constraint.c
    isl_convex_hull.c
    isl_ctx.c
    isl_deprecated.c
    isl_dim_map.c
    isl_equalities.c
    isl_factorization.c
    isl_farkas.c
    isl_ffs.c
    isl_flow.c
    isl_fold.c
    isl_hash.c
    isl_id.c
    isl_id_to_ast_expr.c
    isl_id_to_id.c
    isl_id_to_pw_aff.c
    isl_ilp.c
    isl_imath.c
    isl_input.c
    isl_int_sioimath.c
    isl_local.c
    isl_local_space.c
    isl_lp.c
    isl_map.c
    isl_map_list.c
    isl_map_simplify.c
    isl_map_subtract.c
    isl_map_to_basic_set.c
    isl_mat.c
    isl_morph.c
    isl_obj.c
    isl_options.c
    isl_output.c
    isl_point.c
    isl_polynomial.c
    isl_printer.c
    isl_range.c
    isl_reordering.c
    isl_sample.c
    isl_scan.c
    isl_schedule.c
    isl_schedule_band.c
    isl_schedule_constraints.c
    isl_schedule_node.c
    isl_schedule_read.c
    isl_schedule_tree.c
    isl_scheduler.c
    isl_seq.c
    isl_set_list.c
    isl_sort.c
    isl_space.c
    isl_stream.c
    isl_tab.c
    isl_tab_pip.c
    isl_tarjan.c
    isl_transitive_closure.c
    isl_union_map.c
    isl_val.c
    isl_val_sioimath.c
    isl_vec.c
    isl_version.c
    isl_vertices.c
    print.c
    imath/gmp_compat.c
    imath/imath.c
    imath/imrat.c
)

add_llvm_library(LLVMisl
  ${ISL_FILES}
)

# ISL requires at least C99 to compile. gcc < 5.0 use -std=gnu89 as default.
#target_enable_c99(ISL)

