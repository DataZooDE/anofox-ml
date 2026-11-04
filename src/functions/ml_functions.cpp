// SQL functions of the anofox_ml extension: thin wrappers over the generic
// C ABI in anofox_ml_ffi.h (estimator registry of anofox-ml).

#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

namespace {

// Throw the FFI error message (and free it) when `err` is non-NULL.
void Check(char *err, const char *function) {
    if (err) {
        std::string msg(err);
        anofox_ml_free_string(err);
        throw InvalidInputException("%s: %s", function, msg);
    }
}

// Take ownership of an FFI string.
std::string TakeString(char *s) {
    std::string out(s ? s : "");
    anofox_ml_free_string(s);
    return out;
}

void CaptureUse(const char *function) {
#ifdef ANOFOX_TELEMETRY_ENABLED
    PostHogTelemetry::Instance().CaptureFunctionExecution(function);
#endif
}

//===--------------------------------------------------------------------===//
// ml_version() -> VARCHAR
//===--------------------------------------------------------------------===//
void MlVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, anofox_ml_ffi_version());
}

//===--------------------------------------------------------------------===//
// ml_model_kind(model BLOB) -> VARCHAR
//===--------------------------------------------------------------------===//
void MlModelKindFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t blob) {
        char *kind = nullptr;
        Check(anofox_ml_model_kind(reinterpret_cast<const uint8_t *>(blob.GetData()), blob.GetSize(), &kind),
              "ml_model_kind");
        return StringVector::AddString(result, TakeString(kind));
    });
}

//===--------------------------------------------------------------------===//
// ml_estimators() -> TABLE(kind VARCHAR, default_params VARCHAR)
//===--------------------------------------------------------------------===//
struct EstimatorsState : public GlobalTableFunctionState {
    std::vector<std::string> kinds;
    idx_t next = 0;
};

unique_ptr<FunctionData> EstimatorsBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &types,
                                        vector<string> &names) {
    names = {"kind", "default_params"};
    types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
    return nullptr;
}

unique_ptr<GlobalTableFunctionState> EstimatorsInit(ClientContext &, TableFunctionInitInput &) {
    auto state = make_uniq<EstimatorsState>();
    char *joined = nullptr;
    Check(anofox_ml_kinds(&joined), "ml_estimators");
    std::string all = TakeString(joined);
    size_t start = 0;
    while (start < all.size()) {
        size_t end = all.find('\n', start);
        if (end == std::string::npos) {
            end = all.size();
        }
        state->kinds.push_back(all.substr(start, end - start));
        start = end + 1;
    }
    return std::move(state);
}

void EstimatorsFunction(ClientContext &, TableFunctionInput &input, DataChunk &output) {
    auto &state = input.global_state->Cast<EstimatorsState>();
    idx_t count = 0;
    while (state.next < state.kinds.size() && count < STANDARD_VECTOR_SIZE) {
        const auto &kind = state.kinds[state.next++];
        char *params = nullptr;
        Check(anofox_ml_default_params(kind.c_str(), &params), "ml_estimators");
        output.SetValue(0, count, Value(kind));
        output.SetValue(1, count, Value(TakeString(params)));
        count++;
    }
    output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// ml_fit_agg(kind VARCHAR, y DOUBLE, x DOUBLE[] [, params VARCHAR]) -> BLOB
//===--------------------------------------------------------------------===//
struct FitState {
    bool initialized;
    std::string *kind;
    std::string *params;
    idx_t n_cols;
    std::vector<double> *x;
    std::vector<double> *y;
};

void FitInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<FitState *>(state_p);
    s.initialized = false;
    s.kind = new std::string();
    s.params = new std::string();
    s.n_cols = 0;
    s.x = new std::vector<double>();
    s.y = new std::vector<double>();
}

void FitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        delete s.kind;
        delete s.params;
        delete s.x;
        delete s.y;
    }
}

void FitUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat kind_data, y_data, x_data, params_data, sdata;
    inputs[0].ToUnifiedFormat(count, kind_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    inputs[2].ToUnifiedFormat(count, x_data);
    const bool has_params = input_count > 3;
    if (has_params) {
        inputs[3].ToUnifiedFormat(count, params_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);

    auto kinds = UnifiedVectorFormat::GetData<string_t>(kind_data);
    auto ys = UnifiedVectorFormat::GetData<double>(y_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(inputs[2]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[2]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);
    auto states = reinterpret_cast<FitState **>(sdata.data);

    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ki = kind_data.sel->get_index(i);
        auto yi = y_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!kind_data.validity.RowIsValid(ki) || !y_data.validity.RowIsValid(yi) ||
            !x_data.validity.RowIsValid(xi)) {
            continue; // rows with a NULL kind, target or feature list are skipped
        }
        const auto &entry = lists[xi];
        if (!s.initialized) {
            s.initialized = true;
            *s.kind = kinds[ki].GetString();
            if (has_params) {
                auto pi = params_data.sel->get_index(i);
                if (params_data.validity.RowIsValid(pi)) {
                    *s.params = UnifiedVectorFormat::GetData<string_t>(params_data)[pi].GetString();
                }
            }
            s.n_cols = entry.length;
        }
        if (entry.length != s.n_cols) {
            throw InvalidInputException("ml_fit_agg: every feature list must have %llu values, got %llu",
                                        (unsigned long long)s.n_cols, (unsigned long long)entry.length);
        }
        for (idx_t j = 0; j < entry.length; j++) {
            auto vi = child_data.sel->get_index(entry.offset + j);
            if (!child_data.validity.RowIsValid(vi)) {
                throw InvalidInputException("ml_fit_agg: feature lists must not contain NULL");
            }
            s.x->push_back(values[vi]);
        }
        s.y->push_back(ys[yi]);
    }
}

void FitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<FitState *>(source_vector);
    auto targets = FlatVector::GetData<FitState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (!src.initialized) {
            continue;
        }
        if (!tgt.initialized) {
            tgt.initialized = true;
            *tgt.kind = *src.kind;
            *tgt.params = *src.params;
            tgt.n_cols = src.n_cols;
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("ml_fit_agg: feature lists of different lengths");
        }
        tgt.x->insert(tgt.x->end(), src.x->begin(), src.x->end());
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
    }
}

void FitFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.y->empty()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_fit_agg");
        AnofoxMlBytes blob {nullptr, 0};
        Check(anofox_ml_fit(s.kind->c_str(), s.params->c_str(), s.x->data(), s.y->size(), s.n_cols,
                            s.y->data(), nullptr, &blob),
              "ml_fit_agg");
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

//===--------------------------------------------------------------------===//
// ml_predict(model BLOB, x DOUBLE[]) -> DOUBLE
// Consecutive rows with the same model are predicted in one call.
//===--------------------------------------------------------------------===//
void MlPredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    idx_t count = args.size();
    UnifiedVectorFormat model_data, x_data;
    args.data[0].ToUnifiedFormat(count, model_data);
    args.data[1].ToUnifiedFormat(count, x_data);
    auto models = UnifiedVectorFormat::GetData<string_t>(model_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(args.data[1]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(args.data[1]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);

    result.SetVectorType(VectorType::FLAT_VECTOR);
    auto out = FlatVector::GetData<double>(result);
    auto &validity = FlatVector::Validity(result);
    CaptureUse("ml_predict");

    std::vector<idx_t> rows;
    std::vector<double> x;
    string_t current;
    idx_t n_cols = 0;
    auto flush = [&]() {
        if (rows.empty()) {
            return;
        }
        AnofoxMlDoubles pred {nullptr, 0};
        Check(anofox_ml_predict(reinterpret_cast<const uint8_t *>(current.GetData()), current.GetSize(), x.data(),
                                rows.size(), n_cols, &pred),
              "ml_predict");
        for (idx_t r = 0; r < rows.size(); r++) {
            out[rows[r]] = pred.data[r];
        }
        anofox_ml_free_doubles(pred);
        rows.clear();
        x.clear();
    };

    for (idx_t i = 0; i < count; i++) {
        auto mi = model_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!model_data.validity.RowIsValid(mi) || !x_data.validity.RowIsValid(xi)) {
            validity.SetInvalid(i);
            continue;
        }
        const auto &model = models[mi];
        const auto &entry = lists[xi];
        bool same = !rows.empty() && model.GetSize() == current.GetSize() &&
                    (model.GetData() == current.GetData() ||
                     memcmp(model.GetData(), current.GetData(), model.GetSize()) == 0);
        if (!same || entry.length != n_cols) {
            flush();
            current = model;
            n_cols = entry.length;
        }
        for (idx_t j = 0; j < entry.length; j++) {
            auto vi = child_data.sel->get_index(entry.offset + j);
            if (!child_data.validity.RowIsValid(vi)) {
                throw InvalidInputException("ml_predict: feature lists must not contain NULL");
            }
            x.push_back(values[vi]);
        }
        rows.push_back(i);
    }
    flush();
}

} // anonymous namespace

void RegisterMlFunctions(ExtensionLoader &loader) {
    loader.RegisterFunction(ScalarFunction("ml_version", {}, LogicalType::VARCHAR, MlVersionFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_kind", {LogicalType::BLOB}, LogicalType::VARCHAR, MlModelKindFunction));
    loader.RegisterFunction(ScalarFunction("ml_predict", {LogicalType::BLOB, LogicalType::LIST(LogicalType::DOUBLE)},
                                           LogicalType::DOUBLE, MlPredictFunction));
    loader.RegisterFunction(TableFunction("ml_estimators", {}, EstimatorsFunction, EstimatorsBind, EstimatorsInit));

    AggregateFunctionSet fit_set("ml_fit_agg");
    auto make_fit = [](vector<LogicalType> args) {
        AggregateFunction f("ml_fit_agg", std::move(args), LogicalType::BLOB, AggregateFunction::StateSize<FitState>,
                            FitInitialize, FitUpdate, FitCombine, FitFinalize, nullptr, nullptr, FitDestroy);
        f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
        return f;
    };
    fit_set.AddFunction(
        make_fit({LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}));
    fit_set.AddFunction(make_fit(
        {LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR}));
    loader.RegisterFunction(fit_set);
}

} // namespace duckdb
