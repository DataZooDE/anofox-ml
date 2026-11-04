cmake_minimum_required(VERSION 3.20)

set(CORROSION_VERBOSE_OUTPUT ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get installed Rust targets
execute_process(
    COMMAND rustup target list --installed
    OUTPUT_VARIABLE RUST_TARGETS
    ERROR_QUIET
)

# Propagate arch to rust build for CI
set(Rust_CARGO_TARGET "")
if("${OS_NAME}" STREQUAL "linux")
    if ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    elseif("${CMAKE_CXX_COMPILER}" MATCHES "aarch64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    else()
        string(FIND "${RUST_TARGETS}" "musl" MUSL_TARGET_FOUND)
        if(NOT MUSL_TARGET_FOUND EQUAL -1)
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-musl")
        else()
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
        endif()
    endif()
elseif("${OS_NAME}" STREQUAL "osx")
    if ("${OSX_BUILD_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    elseif ("${OSX_BUILD_ARCH}" STREQUAL "x86_64")
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    elseif ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    endif()
elseif(WIN32)
    if (MINGW AND "${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-pc-windows-gnu")
    elseif (MINGW AND "${OS_ARCH}" STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-pc-windows-gnu")
    elseif (MSVC AND "${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-pc-windows-msvc")
    elseif (MSVC AND "${OS_ARCH}" STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-pc-windows-msvc")
    endif()
endif()

# WASM only when actually building with Emscripten (the target merely being
# installed in rustup must not switch native builds to wasm).
if (EMSCRIPTEN)
    set(Rust_CARGO_TARGET "wasm32-unknown-emscripten")
endif()

message(STATUS "Rust_CARGO_TARGET: ${Rust_CARGO_TARGET}")
message(STATUS "OS_NAME: ${OS_NAME}")
message(STATUS "OS_ARCH: ${OS_ARCH}")

# Fetch Corrosion for Rust integration
# Disable vcpkg for Corrosion's internal CMake to avoid toolchain conflicts
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
# Set any global configuration variables such as `Rust_TOOLCHAIN` before this line!
FetchContent_MakeAvailable(Corrosion)

# Import the Rust FFI crate
corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/crates/anofox-ml-ffi/Cargo.toml"
    CRATES "anofox_ml_ffi"
)

# Set extension name here
set(TARGET_NAME anofox_ml)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# --- Telemetry (disabled on MinGW and WASM) ---
# MinGW: vcpkg cannot build OpenSSL there.
# WASM: the telemetry load event is a raw HTTPS request, which DuckDB-Wasm
# has no socket layer for (as in anofox-statistics).
if(NOT MINGW AND NOT EMSCRIPTEN)
    find_package(OpenSSL REQUIRED)

    add_library(posthog_telemetry_obj OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/src/telemetry.cpp
    )

    target_include_directories(posthog_telemetry_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include
        ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/httplib
        ${OPENSSL_INCLUDE_DIR}
    )

    target_compile_features(posthog_telemetry_obj PUBLIC cxx_std_17)
    target_compile_definitions(posthog_telemetry_obj PRIVATE ANOFOX_TELEMETRY_ENABLED)
    set(ANOFOX_TELEMETRY_ENABLED ON)
else()
    message(STATUS "Telemetry disabled (MinGW or WASM)")
    set(ANOFOX_TELEMETRY_ENABLED OFF)
endif()

include_directories(src/include)

# Extension source files
set(EXTENSION_SOURCES
    src/anofox_ml_extension.cpp
    src/functions/ml_functions.cpp
    src/functions/ml_metrics.cpp
    src/functions/ml_cv.cpp
    src/functions/ml_macros.cpp
    src/functions/ml_text.cpp
    src/functions/ml_stats.cpp
)

# Build static extension
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

# Build loadable extension
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# posthog-telemetry reads the machine id from the IORegistry on macOS, which
# needs IOKit and CoreFoundation linked explicitly.
if(APPLE AND ANOFOX_TELEMETRY_ENABLED)
    target_link_libraries(${EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
endif()

# Link to Rust library
target_link_libraries(${EXTENSION_NAME} anofox_ml_ffi-static)
target_link_libraries(${LOADABLE_EXTENSION_NAME} anofox_ml_ffi-static)

# Telemetry header is always needed (provides no-op stubs when telemetry is disabled)
target_include_directories(${EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)

# Add telemetry objects and OpenSSL linking (when available)
if(ANOFOX_TELEMETRY_ENABLED)
    target_sources(${EXTENSION_NAME} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)
    target_sources(${LOADABLE_EXTENSION_NAME} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)

    target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

    target_compile_definitions(${EXTENSION_NAME} PRIVATE ANOFOX_TELEMETRY_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE ANOFOX_TELEMETRY_ENABLED)
endif()

# Link Windows system libraries
if(WIN32)
    # bcrypt: required by Rust's getrandom crate
    target_link_libraries(${EXTENSION_NAME} bcrypt)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} bcrypt)
    if(ANOFOX_TELEMETRY_ENABLED)
        # iphlpapi: required by telemetry GetAdaptersInfo
        target_link_libraries(${EXTENSION_NAME} iphlpapi)
        target_link_libraries(${LOADABLE_EXTENSION_NAME} iphlpapi)
    endif()
endif()

# Install
install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
