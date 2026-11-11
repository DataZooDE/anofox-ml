# This file is included by DuckDB's build system. It specifies which extension to load.

# On a release tag (vX.Y.Z), stamp that as the extension version. DuckDB's own
# detection passes its --match pattern to git with literal quotes, so it never
# matches a tag and always falls back to the commit hash.
set(ANOFOX_ML_EXT_VERSION "")
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match --match "v[0-9]*.[0-9]*.[0-9]*"
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            RESULT_VARIABLE ANOFOX_ML_GIT_RESULT
            OUTPUT_VARIABLE ANOFOX_ML_GIT_TAG
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    if (ANOFOX_ML_GIT_RESULT EQUAL 0 AND ANOFOX_ML_GIT_TAG MATCHES "^v[0-9]+\\.[0-9]+\\.[0-9]+$")
        set(ANOFOX_ML_EXT_VERSION "${ANOFOX_ML_GIT_TAG}")
    endif()
endif()

# On WASM the extension is a static library and the final .wasm comes from a
# post-build emcc step (duckdb/extension/extension_build_tools.cmake) that
# links only the archives in LINKED_LIBS, not target_link_libraries. Without
# the Rust FFI archive every anofox_ml_* function is an unresolved import and
# the module can't load (#29). LINKED_LIBS is unused outside WASM.
set(ANOFOX_ML_LINKED_LIBS "$<TARGET_FILE:anofox_ml_ffi-static>")
# Current Rust objects declare the bulk-memory-opt and
# call-indirect-overlong target features, which emsdk 3.1.71's binaryen
# rejects; their instructions are covered by bulk-memory and reference-types,
# so each wasm platform's feature list is given explicitly, as wasm-opt
# enables them for that platform (the extra arguments ride on
# LINKED_LIBS, which DuckDB splits on spaces). As in anofox-forecast.
set(ANOFOX_ML_WASM_FEATURES "")
if("${DUCKDB_EXPLICIT_PLATFORM}" STREQUAL "wasm_eh")
    set(ANOFOX_ML_WASM_FEATURES "bulk-memory,exception-handling,multivalue,mutable-globals,nontrapping-fptoint,reference-types,sign-ext,simd128")
elseif("${DUCKDB_EXPLICIT_PLATFORM}" STREQUAL "wasm_mvp")
    set(ANOFOX_ML_WASM_FEATURES "bulk-memory,exception-handling,multivalue,mutable-globals,nontrapping-fptoint,reference-types,sign-ext")
elseif("${DUCKDB_EXPLICIT_PLATFORM}" STREQUAL "wasm_threads")
    set(ANOFOX_ML_WASM_FEATURES "atomics,bulk-memory,exception-handling,multivalue,mutable-globals,nontrapping-fptoint,reference-types,sign-ext")
endif()
# Only emsdk 3.x needs this; the DuckDB 2.0 wasm builds use emsdk 6, whose
# binaryen knows those features.
if(DEFINED EMSCRIPTEN_VERSION AND NOT EMSCRIPTEN_VERSION VERSION_LESS "4.0")
    set(ANOFOX_ML_WASM_FEATURES "")
endif()
if(NOT ANOFOX_ML_WASM_FEATURES STREQUAL "")
    string(APPEND ANOFOX_ML_LINKED_LIBS " -Xlinker --no-check-features -Xlinker --features=${ANOFOX_ML_WASM_FEATURES}")
endif()

set(ANOFOX_ML_VERSION_ARG "")
if (NOT ANOFOX_ML_EXT_VERSION STREQUAL "")
    set(ANOFOX_ML_VERSION_ARG EXTENSION_VERSION ${ANOFOX_ML_EXT_VERSION})
endif()
duckdb_extension_load(anofox_ml SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} LOAD_TESTS ${ANOFOX_ML_VERSION_ARG}
    LINKED_LIBS "${ANOFOX_ML_LINKED_LIBS}")
