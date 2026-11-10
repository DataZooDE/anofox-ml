#pragma once

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

class ExtensionLoader;

void RegisterMlFunctions(ExtensionLoader &loader);
void RegisterMlMetrics(ExtensionLoader &loader);
void RegisterMlCv(ExtensionLoader &loader);
void RegisterMlMacros(ExtensionLoader &loader);
void RegisterMlText(ExtensionLoader &loader);
void RegisterMlStats(ExtensionLoader &loader);
void RegisterMlImpute(ExtensionLoader &loader);

// Scalar functions that report invalid models or arguments as errors are
// registered as fallible (DuckDB 2.0 turns an error of any other function
// into an internal error).
inline ScalarFunction Fallible(ScalarFunction function) {
    function.SetFallible();
    return function;
}

class AnofoxMlExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
