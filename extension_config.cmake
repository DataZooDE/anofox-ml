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

if (ANOFOX_ML_EXT_VERSION STREQUAL "")
    duckdb_extension_load(anofox_ml SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} LOAD_TESTS)
else()
    duckdb_extension_load(anofox_ml SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} LOAD_TESTS EXTENSION_VERSION ${ANOFOX_ML_EXT_VERSION})
endif()
