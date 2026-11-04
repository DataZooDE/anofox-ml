#define DUCKDB_EXTENSION_MAIN

#include "include/anofox_ml_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "telemetry.hpp"

namespace duckdb {

static std::string ExtensionVersion() {
#ifdef EXT_VERSION_ANOFOX_ML
    return EXT_VERSION_ANOFOX_ML;
#else
    return "0.1.0";
#endif
}

#ifdef ANOFOX_TELEMETRY_ENABLED

namespace {

void OnTelemetryEnabled(ClientContext &context, SetScope scope, Value &parameter) {
    if (parameter.IsNull()) {
        throw InvalidInputException("anofox_telemetry_enabled cannot be NULL");
    }
    PostHogTelemetry::Instance().SetEnabled(BooleanValue::Get(parameter));
}

void OnTelemetryKey(ClientContext &context, SetScope scope, Value &parameter) {
    if (parameter.IsNull()) {
        throw InvalidInputException("anofox_telemetry_key cannot be NULL");
    }
    PostHogTelemetry::Instance().SetAPIKey(StringValue::Get(parameter));
}

} // anonymous namespace

static const char *TELEMETRY_KEY = "phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t";

static void RegisterTelemetryOptions(ExtensionLoader &loader) {
    auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
    config.AddExtensionOption("anofox_telemetry_enabled", "Enable or disable anonymous usage telemetry",
                              LogicalType::BOOLEAN, Value::BOOLEAN(true), OnTelemetryEnabled);
    config.AddExtensionOption("anofox_telemetry_key", "PostHog API key for telemetry", LogicalType::VARCHAR,
                              Value(TELEMETRY_KEY), OnTelemetryKey);
}

#endif // ANOFOX_TELEMETRY_ENABLED

void LoadInternal(ExtensionLoader &loader) {
#ifdef ANOFOX_TELEMETRY_ENABLED
    RegisterTelemetryOptions(loader);
    auto &telemetry = PostHogTelemetry::Instance();
    telemetry.SetAPIKey(TELEMETRY_KEY);
    telemetry.CaptureExtensionLoad("anofox_ml", ExtensionVersion());
#endif
    RegisterMlFunctions(loader);
    RegisterMlMetrics(loader);
}

void AnofoxMlExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
}

std::string AnofoxMlExtension::Name() {
    return "anofox_ml";
}

std::string AnofoxMlExtension::Version() const {
    return ExtensionVersion();
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_ml, loader) {
    duckdb::LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *anofox_ml_version() {
#ifdef EXT_VERSION_ANOFOX_ML
    return EXT_VERSION_ANOFOX_ML;
#else
    return "0.1.0";
#endif
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
