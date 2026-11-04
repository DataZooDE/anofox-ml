#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

void RegisterMlFunctions(ExtensionLoader &loader);
void RegisterMlMetrics(ExtensionLoader &loader);
void RegisterMlCv(ExtensionLoader &loader);

class AnofoxMlExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
