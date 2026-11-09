// SQL functions of the anofox_ml extension: thin wrappers over the generic
// C ABI in anofox_ml_ffi.h (estimator registry of anofox-ml).

#include <cmath>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

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

// True when a feature list holds a NULL element; such rows are treated like
// a NULL feature list (skipped when fitting, NULL when predicting).
bool HasNullElement(const UnifiedVectorFormat &child_data, const list_entry_t &entry) {
    for (idx_t j = 0; j < entry.length; j++) {
        if (!child_data.validity.RowIsValid(child_data.sel->get_index(entry.offset + j))) {
            return true;
        }
    }
    return false;
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
// ml_estimators() -> TABLE(kind VARCHAR, default_params VARCHAR, capabilities VARCHAR[])
//===--------------------------------------------------------------------===//
struct EstimatorsState : public GlobalTableFunctionState {
    std::vector<std::string> kinds;
    idx_t next = 0;
};

unique_ptr<FunctionData> EstimatorsBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &types,
                                        vector<string> &names) {
    names = {"kind", "default_params", "capabilities"};
    types = {LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::VARCHAR),
             LogicalType::LIST(LogicalType(LogicalTypeId::VARCHAR))};
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
        char *caps = nullptr;
        Check(anofox_ml_kind_capabilities(kind.c_str(), &caps), "ml_estimators");
        vector<Value> list;
        std::string all = TakeString(caps);
        for (size_t start = 0; start < all.size();) {
            size_t end = all.find('\n', start);
            if (end == std::string::npos) {
                end = all.size();
            }
            list.push_back(Value(all.substr(start, end - start)));
            start = end + 1;
        }
        output.SetValue(2, count, Value::LIST(LogicalType(LogicalTypeId::VARCHAR), std::move(list)));
        count++;
    }
    output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// ml_fit_agg(kind VARCHAR, y DOUBLE | VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> BLOB
// ml_fit_agg(kind VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> BLOB (unsupervised)
//===--------------------------------------------------------------------===//
struct FitState {
    bool initialized;
    std::string *kind;
    std::string *params;
    bool has_y;
    bool has_w;
    idx_t n_cols;
    idx_t n_rows;
    std::vector<double> *x;
    std::vector<double> *y;
    std::vector<double> *w;
    std::vector<std::string> *labels; // string targets (has_labels)
    std::vector<uint8_t> *unlabelled; // per label: 1 for a NULL target of a semi-supervised kind
    bool has_labels;
    std::vector<std::string> *sx; // VARCHAR features of an encoder (string_x)
    bool string_x;
    std::string *options; // ml_tune_agg options
    std::vector<uint8_t> *val; // per used row: 1 = validation row (ml_fit_with_validation_agg)
    // One entry per input row: 1 when the row was used for fitting. Lets the
    // per-row aggregates (ml_cluster_agg, ...) return NULL for skipped rows.
    std::vector<uint8_t> *used;
};

void FitInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<FitState *>(state_p);
    s.initialized = false;
    s.kind = new std::string();
    s.params = new std::string();
    s.has_y = false;
    s.has_w = false;
    s.n_cols = 0;
    s.n_rows = 0;
    s.x = new std::vector<double>();
    s.y = new std::vector<double>();
    s.w = new std::vector<double>();
    s.labels = new std::vector<std::string>();
    s.unlabelled = new std::vector<uint8_t>();
    s.has_labels = false;
    s.sx = new std::vector<std::string>();
    s.string_x = false;
    s.options = new std::string();
    s.used = new std::vector<uint8_t>();
    s.val = new std::vector<uint8_t>();
}

void FitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        delete s.kind;
        delete s.params;
        delete s.x;
        delete s.y;
        delete s.w;
        delete s.labels;
        delete s.unlabelled;
        delete s.sx;
        delete s.options;
        delete s.used;
        delete s.val;
    }
}

// The memory guard of the fitting aggregates, read from the
// anofox_ml_max_fit_values setting at bind time.
// Settings of the fitting aggregates, read at bind time: the memory guard
// (anofox_ml_max_fit_values) and the thread budget of a fit
// (anofox_ml_threads, 0 = DuckDB's threads).
struct FitBindData : public FunctionData {
    FitBindData(uint64_t max_values_p, idx_t threads_p) : max_values(max_values_p), threads(threads_p) {
    }
    uint64_t max_values; // 0 = unlimited
    idx_t threads;
    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<FitBindData>(max_values, threads);
    }
    bool Equals(const FunctionData &other) const override {
        auto &o = other.Cast<FitBindData>();
        return max_values == o.max_values && threads == o.threads;
    }
};

unique_ptr<FunctionData> FitBind(ClientContext &context, AggregateFunction &, vector<unique_ptr<Expression>> &) {
    Value v;
    uint64_t max_values = 100000000;
    if (context.TryGetCurrentSetting("anofox_ml_max_fit_values", v) && !v.IsNull()) {
        max_values = v.GetValue<uint64_t>();
    }
    idx_t threads = 0;
    if (context.TryGetCurrentSetting("anofox_ml_threads", v) && !v.IsNull()) {
        threads = v.GetValue<uint64_t>();
    }
    if (threads == 0) {
        threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
    }
    return make_uniq<FitBindData>(max_values, threads);
}

idx_t FitThreads(const AggregateInputData &aggr) {
    return aggr.bind_data ? aggr.bind_data->Cast<FitBindData>().threads : 1;
}

void CheckFitSize(const AggregateInputData &aggr, idx_t n_values) {
    if (!aggr.bind_data) {
        return;
    }
    auto max_values = aggr.bind_data->Cast<FitBindData>().max_values;
    if (max_values > 0 && n_values > max_values) {
        throw InvalidInputException(
            "the training set of one group exceeds anofox_ml_max_fit_values (%llu rows x features); sample it "
            "(e.g. FROM t USING SAMPLE 1000000 ROWS) or raise the limit with SET anofox_ml_max_fit_values = ...",
            (unsigned long long)max_values);
    }
}

// Inputs: kind, [y,] x [, params [, weight]]; y is a string label when LABELS.
// With UNLABELLED, a NULL y of a semi-supervised kind is kept as an
// unlabelled row (y = -1, or a NULL label) instead of skipping the row.
// With HAS_VAL, a BOOLEAN validation flag follows x (rows with a NULL flag
// are skipped).
template <bool HAS_Y, bool HAS_W = false, bool LABELS = false, bool UNLABELLED = true, bool HAS_VAL = false>
void FitUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &state_vector, idx_t count) {
    constexpr idx_t X = HAS_Y ? 2 : 1;
    constexpr idx_t P = X + 1 + (HAS_VAL ? 1 : 0); // params, then weight
    UnifiedVectorFormat kind_data, y_data, x_data, val_data, params_data, w_data, sdata;
    inputs[0].ToUnifiedFormat(count, kind_data);
    if (HAS_Y) {
        inputs[1].ToUnifiedFormat(count, y_data);
    }
    inputs[X].ToUnifiedFormat(count, x_data);
    if (HAS_VAL) {
        inputs[X + 1].ToUnifiedFormat(count, val_data);
    }
    const bool has_params = input_count > P;
    if (has_params) {
        inputs[P].ToUnifiedFormat(count, params_data);
    }
    if (HAS_W) {
        inputs[P + 1].ToUnifiedFormat(count, w_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);

    auto kinds = UnifiedVectorFormat::GetData<string_t>(kind_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(inputs[X]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[X]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);
    auto states = reinterpret_cast<FitState **>(sdata.data);

    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ki = kind_data.sel->get_index(i);
        auto yi = HAS_Y ? y_data.sel->get_index(i) : 0;
        auto xi = x_data.sel->get_index(i);
        auto wi = HAS_W ? w_data.sel->get_index(i) : 0;
        auto vi = HAS_VAL ? val_data.sel->get_index(i) : 0;
        const bool no_y = HAS_Y && !y_data.validity.RowIsValid(yi);
        const bool unlabelled = UNLABELLED && no_y && kind_data.validity.RowIsValid(ki) &&
                                anofox_ml_accepts_unlabelled(kinds[ki].GetString().c_str());
        if (!kind_data.validity.RowIsValid(ki) || (no_y && !unlabelled) || !x_data.validity.RowIsValid(xi) ||
            (HAS_W && !w_data.validity.RowIsValid(wi)) || (HAS_VAL && !val_data.validity.RowIsValid(vi))) {
            s.used->push_back(0); // rows with a NULL kind, target, feature list or weight are skipped
            continue;
        }
        const auto &entry = lists[xi];
        if (HasNullElement(child_data, entry)) {
            s.used->push_back(0);
            continue;
        }
        if (!s.initialized) {
            s.initialized = true;
            s.has_y = HAS_Y;
            s.has_w = HAS_W;
            s.has_labels = LABELS;
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
            s.x->push_back(values[child_data.sel->get_index(entry.offset + j)]);
        }
        if (HAS_Y && LABELS) {
            s.labels->push_back(unlabelled ? std::string()
                                           : UnifiedVectorFormat::GetData<string_t>(y_data)[yi].GetString());
            s.unlabelled->push_back(unlabelled ? 1 : 0);
        } else if (HAS_Y) {
            s.y->push_back(unlabelled ? -1.0 : UnifiedVectorFormat::GetData<double>(y_data)[yi]);
        }
        if (HAS_W) {
            s.w->push_back(UnifiedVectorFormat::GetData<double>(w_data)[wi]);
        }
        if (HAS_VAL) {
            s.val->push_back(UnifiedVectorFormat::GetData<bool>(val_data)[vi] ? 1 : 0);
        }
        s.n_rows++;
        s.used->push_back(1);
        CheckFitSize(aggr, s.x->size());
    }
}

// ml_fit_agg(kind VARCHAR, x VARCHAR[] [, params VARCHAR]) for the categorical
// encoders; rows with a NULL list or a NULL element are skipped.
void EncodeUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat kind_data, x_data, params_data, sdata;
    inputs[0].ToUnifiedFormat(count, kind_data);
    inputs[1].ToUnifiedFormat(count, x_data);
    const bool has_params = input_count > 2;
    if (has_params) {
        inputs[2].ToUnifiedFormat(count, params_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto kinds = UnifiedVectorFormat::GetData<string_t>(kind_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(inputs[1]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[1]), child_data);
    auto values = UnifiedVectorFormat::GetData<string_t>(child_data);
    auto states = reinterpret_cast<FitState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ki = kind_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!kind_data.validity.RowIsValid(ki) || !x_data.validity.RowIsValid(xi) ||
            HasNullElement(child_data, lists[xi])) {
            s.used->push_back(0);
            continue;
        }
        const auto &entry = lists[xi];
        if (!s.initialized) {
            s.initialized = true;
            s.string_x = true;
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
            s.sx->push_back(values[child_data.sel->get_index(entry.offset + j)].GetString());
        }
        s.n_rows++;
        s.used->push_back(1);
        CheckFitSize(aggr, s.sx->size());
    }
}

void FitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr, idx_t count) {
    auto sources = FlatVector::GetData<FitState *>(source_vector);
    auto targets = FlatVector::GetData<FitState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        tgt.used->insert(tgt.used->end(), src.used->begin(), src.used->end());
        if (!src.initialized) {
            continue;
        }
        if (!tgt.initialized) {
            tgt.initialized = true;
            tgt.has_y = src.has_y;
            tgt.has_w = src.has_w;
            tgt.has_labels = src.has_labels;
            tgt.string_x = src.string_x;
            *tgt.options = *src.options;
            *tgt.kind = *src.kind;
            *tgt.params = *src.params;
            tgt.n_cols = src.n_cols;
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("ml_fit_agg: feature lists of different lengths");
        }
        tgt.x->insert(tgt.x->end(), src.x->begin(), src.x->end());
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
        tgt.w->insert(tgt.w->end(), src.w->begin(), src.w->end());
        tgt.labels->insert(tgt.labels->end(), src.labels->begin(), src.labels->end());
        tgt.unlabelled->insert(tgt.unlabelled->end(), src.unlabelled->begin(), src.unlabelled->end());
        tgt.val->insert(tgt.val->end(), src.val->begin(), src.val->end());
        tgt.sx->insert(tgt.sx->end(), src.sx->begin(), src.sx->end());
        tgt.n_rows += src.n_rows;
        CheckFitSize(aggr, tgt.x->size() + tgt.sx->size());
    }
}

void FitFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_fit_agg");
        AnofoxMlBytes blob {nullptr, 0};
        const double *w = s.has_w ? s.w->data() : nullptr;
        if (s.string_x) {
            std::vector<const char *> x;
            x.reserve(s.sx->size());
            for (const auto &v : *s.sx) {
                x.push_back(v.c_str());
            }
            Check(anofox_ml_fit_encoder(s.kind->c_str(), s.params->c_str(), x.data(), s.n_rows, s.n_cols, &blob),
                  "ml_fit_agg");
        } else if (s.has_labels) {
            std::vector<const char *> labels;
            labels.reserve(s.labels->size());
            for (idx_t j = 0; j < s.labels->size(); j++) {
                labels.push_back((*s.unlabelled)[j] ? nullptr : (*s.labels)[j].c_str());
            }
            Check(anofox_ml_fit_labels(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                       labels.data(), w, FitThreads(aggr), &blob),
                  "ml_fit_agg");
        } else {
            Check(anofox_ml_fit(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                s.has_y ? s.y->data() : nullptr, w, FitThreads(aggr), &blob),
                  "ml_fit_agg");
        }
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

//===--------------------------------------------------------------------===//
// ml_fit_with_validation_agg(kind VARCHAR, y DOUBLE | VARCHAR, x DOUBLE[], is_validation BOOLEAN
//                            [, params VARCHAR [, weight DOUBLE]]) -> BLOB
// Rows with is_validation = true are the validation set of early stopping.
//===--------------------------------------------------------------------===//
void FitValidationFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count,
                           idx_t offset) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_fit_with_validation_agg");
        AnofoxMlBytes blob {nullptr, 0};
        std::vector<const char *> labels;
        for (const auto &l : *s.labels) {
            labels.push_back(l.c_str());
        }
        Check(anofox_ml_fit_validation(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                       s.has_labels ? nullptr : s.y->data(), s.has_labels ? labels.data() : nullptr,
                                       s.has_w ? s.w->data() : nullptr, s.val->data(), FitThreads(aggr), &blob),
              "ml_fit_with_validation_agg");
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

//===--------------------------------------------------------------------===//
// ml_tune_agg(kind VARCHAR, y DOUBLE | VARCHAR, x DOUBLE[], search_space VARCHAR [, options VARCHAR])
//   -> STRUCT(best_params VARCHAR, best_score DOUBLE, model BLOB, trials VARCHAR)
// Grid / random search with k-fold CV; the search space rides in `params`.
//===--------------------------------------------------------------------===//
template <bool LABELS>
void TuneUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &state_vector, idx_t count) {
    FitUpdate<true, false, LABELS, false>(inputs, aggr, 4, state_vector, count);
    if (input_count < 5) {
        return;
    }
    UnifiedVectorFormat options_data, sdata;
    inputs[4].ToUnifiedFormat(count, options_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<FitState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto oi = options_data.sel->get_index(i);
        if (s.options->empty() && options_data.validity.RowIsValid(oi)) {
            *s.options = UnifiedVectorFormat::GetData<string_t>(options_data)[oi].GetString();
        }
    }
}

LogicalType TuneType() {
    child_list_t<LogicalType> fields;
    fields.emplace_back("best_params", LogicalType(LogicalTypeId::VARCHAR));
    fields.emplace_back("best_score", LogicalType(LogicalTypeId::DOUBLE));
    fields.emplace_back("model", LogicalType(LogicalTypeId::BLOB));
    fields.emplace_back("trials", LogicalType(LogicalTypeId::VARCHAR));
    return LogicalType::STRUCT(std::move(fields));
}

void TuneFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    auto &fields = StructVector::GetEntries(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_tune_agg");
        std::vector<const char *> labels;
        for (const auto &l : *s.labels) {
            labels.push_back(l.c_str());
        }
        AnofoxMlTuneResult r {};
        Check(anofox_ml_tune(s.kind->c_str(), s.params->c_str(), s.options->c_str(), s.x->data(), s.n_rows,
                             s.n_cols, s.has_labels ? nullptr : s.y->data(), s.has_labels ? labels.data() : nullptr,
                             FitThreads(aggr), &r),
              "ml_tune_agg");
        FlatVector::GetData<string_t>(*fields[0])[row] = StringVector::AddString(*fields[0], r.best_params);
        FlatVector::GetData<double>(*fields[1])[row] = r.best_score;
        FlatVector::GetData<string_t>(*fields[2])[row] =
            StringVector::AddStringOrBlob(*fields[2], reinterpret_cast<const char *>(r.model.data), r.model.len);
        FlatVector::GetData<string_t>(*fields[3])[row] = StringVector::AddString(*fields[3], r.trials);
        anofox_ml_free_tune_result(r);
    }
}

//===--------------------------------------------------------------------===//
// ml_cluster_agg(kind VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> BIGINT[]
// ml_outlier_score_agg(kind VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> DOUBLE[]
// Fit on the group and return one value per input row, in input order (use
// ORDER BY inside the aggregate to fix it); skipped rows get NULL.
//===--------------------------------------------------------------------===//
template <const char *FUNCTION, const char *METHODS, class T>
void RowsFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<FitState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse(FUNCTION);
        AnofoxMlDoubles out {nullptr, 0};
        size_t out_cols = 0;
        Check(anofox_ml_fit_apply(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                  s.has_y ? s.y->data() : nullptr, s.has_w ? s.w->data() : nullptr, METHODS,
                                  FitThreads(aggr), &out, &out_cols),
              FUNCTION);
        if (out_cols != 1) {
            anofox_ml_free_doubles(out);
            throw InvalidInputException("%s: %s returns %llu values per row", FUNCTION, s.kind->c_str(),
                                        (unsigned long long)out_cols);
        }
        idx_t list_offset = ListVector::GetListSize(result);
        idx_t n = s.used->size();
        ListVector::Reserve(result, list_offset + n);
        auto &child = ListVector::GetEntry(result);
        auto child_data = FlatVector::GetData<T>(child);
        auto &child_validity = FlatVector::Validity(child);
        idx_t next = 0;
        for (idx_t r = 0; r < n; r++) {
            if ((*s.used)[r]) {
                child_data[list_offset + r] = static_cast<T>(out.data[next++]);
            } else {
                child_validity.SetInvalid(list_offset + r);
            }
        }
        anofox_ml_free_doubles(out);
        ListVector::SetListSize(result, list_offset + n);
        entries[row] = list_entry_t(list_offset, n);
    }
}

//===--------------------------------------------------------------------===//
// ml_fit_transform_agg(kind VARCHAR, [y DOUBLE,] x DOUBLE[] [, params VARCHAR]) -> DOUBLE[][]
// The output of a fit on the group for each input row: the stored training
// embedding (t-SNE, Isomap, LLE, MDS; TargetEncoder's cross-fitted encoding)
// or else `transform` of the training rows. Row order as in RowsFinalize.
//===--------------------------------------------------------------------===//
void FitTransformFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    static constexpr const char *FUNCTION = "ml_fit_transform_agg";
    auto states = FlatVector::GetData<FitState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse(FUNCTION);
        AnofoxMlDoubles out {nullptr, 0};
        size_t out_cols = 0;
        Check(anofox_ml_fit_apply(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                  s.has_y ? s.y->data() : nullptr, s.has_w ? s.w->data() : nullptr,
                                  "embedding,transform", FitThreads(aggr), &out, &out_cols),
              FUNCTION);
        // Outer list: one entry per input row; inner list: out_cols values.
        idx_t n = s.used->size();
        idx_t outer_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, outer_offset + n);
        auto &rows = ListVector::GetEntry(result);
        auto row_entries = FlatVector::GetData<list_entry_t>(rows);
        auto &row_validity = FlatVector::Validity(rows);
        idx_t value_offset = ListVector::GetListSize(rows);
        ListVector::Reserve(rows, value_offset + s.n_rows * out_cols);
        auto values = FlatVector::GetData<double>(ListVector::GetEntry(rows));
        idx_t next = 0;
        for (idx_t r = 0; r < n; r++) {
            if ((*s.used)[r]) {
                memcpy(values + value_offset, out.data + next * out_cols, out_cols * sizeof(double));
                row_entries[outer_offset + r] = list_entry_t(value_offset, out_cols);
                value_offset += out_cols;
                next++;
            } else {
                row_validity.SetInvalid(outer_offset + r);
            }
        }
        anofox_ml_free_doubles(out);
        ListVector::SetListSize(rows, value_offset);
        ListVector::SetListSize(result, outer_offset + n);
        entries[row] = list_entry_t(outer_offset, n);
    }
}

constexpr char kClusterAgg[] = "ml_cluster_agg";
constexpr char kClusterMethods[] = "labels,predict";
constexpr char kOutlierAgg[] = "ml_outlier_score_agg";
constexpr char kOutlierMethods[] = "score_samples,decision_function";

//===--------------------------------------------------------------------===//
// Row-wise model application. Consecutive rows with the same model BLOB are
// sent to the FFI in one call; `emit(row, values, n_values)` writes one
// output row.
//===--------------------------------------------------------------------===//
// `call(model, x, rows, n_cols)` runs the FFI on one batch of rows.
template <class CALL>
void ApplyBatchedWith(DataChunk &args, const char *function, Vector &result, CALL &&call) {
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
    auto &validity = FlatVector::Validity(result);
    CaptureUse(function);

    std::vector<idx_t> rows;
    std::vector<double> x;
    string_t current;
    idx_t n_cols = 0;
    auto flush = [&]() {
        if (rows.empty()) {
            return;
        }
        call(current, x, rows, n_cols);
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
        if (HasNullElement(child_data, entry)) {
            validity.SetInvalid(i);
            continue;
        }
        bool same = !rows.empty() && model.GetSize() == current.GetSize() &&
                    (model.GetData() == current.GetData() ||
                     memcmp(model.GetData(), current.GetData(), model.GetSize()) == 0);
        if (!same || entry.length != n_cols) {
            flush();
            current = model;
            n_cols = entry.length;
        }
        for (idx_t j = 0; j < entry.length; j++) {
            x.push_back(values[child_data.sel->get_index(entry.offset + j)]);
        }
        rows.push_back(i);
    }
    flush();
}

template <class EMIT>
void ApplyBatched(DataChunk &args, const char *function, const char *method, Vector &result, EMIT &&emit) {
    ApplyBatchedWith(args, function, result,
                     [&](string_t model, std::vector<double> &x, std::vector<idx_t> &rows, idx_t n_cols) {
                         AnofoxMlDoubles out {nullptr, 0};
                         size_t out_cols = 0;
                         Check(anofox_ml_apply(reinterpret_cast<const uint8_t *>(model.GetData()), model.GetSize(),
                                               method, x.data(), rows.size(), n_cols, &out, &out_cols),
                               function);
                         for (idx_t r = 0; r < rows.size(); r++) {
                             emit(rows[r], out.data + r * out_cols, out_cols);
                         }
                         anofox_ml_free_doubles(out);
                     });
}

//===--------------------------------------------------------------------===//
// ml_predict_label(model BLOB, x DOUBLE[]) -> VARCHAR (string class label)
//===--------------------------------------------------------------------===//
void MlPredictLabelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto out = FlatVector::GetData<string_t>(result);
    ApplyBatchedWith(args, "ml_predict_label", result,
                     [&](string_t model, std::vector<double> &x, std::vector<idx_t> &rows, idx_t n_cols) {
                         AnofoxMlStrings labels {nullptr, 0};
                         Check(anofox_ml_predict_labels(reinterpret_cast<const uint8_t *>(model.GetData()),
                                                        model.GetSize(), x.data(), rows.size(), n_cols, &labels),
                               "ml_predict_label");
                         for (idx_t r = 0; r < rows.size(); r++) {
                             out[rows[r]] = StringVector::AddString(result, labels.data[r]);
                         }
                         anofox_ml_free_strings(labels);
                     });
}

//===--------------------------------------------------------------------===//
// ml_predict_interval(model BLOB, x DOUBLE[], alpha DOUBLE) -> STRUCT(lower, upper)
// ml_predict_set(model BLOB, x DOUBLE[], alpha DOUBLE) -> VARCHAR[]
// Conformal predictions of a model calibrated with ml_conformalize_agg.
//===--------------------------------------------------------------------===//

// Run `call(rows, alpha)` on the rows of one model batch, split into runs of
// equal alpha; rows with a NULL alpha become NULL.
template <class CALL>
void ByAlpha(DataChunk &args, Vector &result, std::vector<idx_t> &rows, std::vector<double> &x, idx_t n_cols,
             CALL &&call) {
    UnifiedVectorFormat alpha_data;
    args.data[2].ToUnifiedFormat(args.size(), alpha_data);
    auto alphas = UnifiedVectorFormat::GetData<double>(alpha_data);
    std::vector<idx_t> run_rows;
    std::vector<double> run_x;
    double run_alpha = 0;
    auto flush = [&]() {
        if (!run_rows.empty()) {
            call(run_rows, run_x, run_alpha);
        }
        run_rows.clear();
        run_x.clear();
    };
    for (idx_t r = 0; r < rows.size(); r++) {
        auto ai = alpha_data.sel->get_index(rows[r]);
        if (!alpha_data.validity.RowIsValid(ai)) {
            FlatVector::SetNull(result, rows[r], true);
            continue;
        }
        if (!run_rows.empty() && alphas[ai] != run_alpha) {
            flush();
        }
        run_alpha = alphas[ai];
        run_rows.push_back(rows[r]);
        run_x.insert(run_x.end(), x.begin() + r * n_cols, x.begin() + (r + 1) * n_cols);
    }
    flush();
}

LogicalType IntervalType() {
    child_list_t<LogicalType> fields;
    fields.emplace_back("lower", LogicalType(LogicalTypeId::DOUBLE));
    fields.emplace_back("upper", LogicalType(LogicalTypeId::DOUBLE));
    return LogicalType::STRUCT(std::move(fields));
}

void MlPredictIntervalFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &fields = StructVector::GetEntries(result);
    auto lower = FlatVector::GetData<double>(*fields[0]);
    auto upper = FlatVector::GetData<double>(*fields[1]);
    ApplyBatchedWith(
        args, "ml_predict_interval", result,
        [&](string_t model, std::vector<double> &x, std::vector<idx_t> &rows, idx_t n_cols) {
            ByAlpha(args, result, rows, x, n_cols, [&](std::vector<idx_t> &run, std::vector<double> &rx, double alpha) {
                AnofoxMlDoubles out {nullptr, 0};
                Check(anofox_ml_predict_interval(reinterpret_cast<const uint8_t *>(model.GetData()), model.GetSize(),
                                                 rx.data(), run.size(), n_cols, alpha, &out),
                      "ml_predict_interval");
                for (idx_t r = 0; r < run.size(); r++) {
                    lower[run[r]] = out.data[2 * r];
                    upper[run[r]] = out.data[2 * r + 1];
                }
                anofox_ml_free_doubles(out);
            });
        });
}

void MlPredictSetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    // Collect per row first: the list child vector is appended to in row order.
    std::vector<std::pair<idx_t, std::vector<std::string>>> sets;
    ApplyBatchedWith(
        args, "ml_predict_set", result,
        [&](string_t model, std::vector<double> &x, std::vector<idx_t> &rows, idx_t n_cols) {
            ByAlpha(args, result, rows, x, n_cols, [&](std::vector<idx_t> &run, std::vector<double> &rx, double alpha) {
                AnofoxMlStrings members {nullptr, 0};
                AnofoxMlDoubles lengths {nullptr, 0};
                Check(anofox_ml_predict_set(reinterpret_cast<const uint8_t *>(model.GetData()), model.GetSize(),
                                            rx.data(), run.size(), n_cols, alpha, &members, &lengths),
                      "ml_predict_set");
                size_t next = 0;
                for (idx_t r = 0; r < run.size(); r++) {
                    std::vector<std::string> set;
                    for (size_t j = 0; j < static_cast<size_t>(lengths.data[r]); j++) {
                        set.emplace_back(members.data[next++]);
                    }
                    sets.emplace_back(run[r], std::move(set));
                }
                anofox_ml_free_strings(members);
                anofox_ml_free_doubles(lengths);
            });
        });
    auto entries = FlatVector::GetData<list_entry_t>(result);
    idx_t offset = ListVector::GetListSize(result);
    for (auto &row : sets) {
        entries[row.first] = list_entry_t(offset, row.second.size());
        for (auto &m : row.second) {
            ListVector::PushBack(result, Value(m));
        }
        offset += row.second.size();
    }
}

//===--------------------------------------------------------------------===//
// Aggregates over (model BLOB, y DOUBLE | VARCHAR, x DOUBLE[] [, options]):
// ml_conformalize_agg(model, y, x) -> BLOB calibrates a fitted model on
// held-out rows (split conformal); ml_permutation_importance_agg(model, y, x
// [, options]) -> STRUCT(feature, importance_mean, importance_std)[].
//===--------------------------------------------------------------------===//
struct ConformalState {
    std::string *model;
    std::string *options;
    std::vector<double> *x;
    std::vector<double> *y;
    std::vector<std::string> *labels;
    idx_t n_rows;
    idx_t n_cols;
};

void ConformalInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<ConformalState *>(state_p);
    s.model = new std::string();
    s.options = new std::string();
    s.x = new std::vector<double>();
    s.y = new std::vector<double>();
    s.labels = new std::vector<std::string>();
    s.n_rows = 0;
    s.n_cols = 0;
}

void ConformalDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<ConformalState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->model;
        delete states[i]->options;
        delete states[i]->x;
        delete states[i]->y;
        delete states[i]->labels;
    }
}

// Rows of (model, y, x [, options]) for one model per group; also used by
// ml_warm_start_agg (options = params). FUNCTION names the aggregate in errors.
constexpr char kConformalize[] = "ml_conformalize_agg";
constexpr char kWarmStart[] = "ml_warm_start_agg";

template <bool LABELS, const char *FUNCTION = kConformalize>
void ConformalUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat model_data, y_data, x_data, options_data, sdata;
    if (input_count > 3) {
        inputs[3].ToUnifiedFormat(count, options_data);
    }
    inputs[0].ToUnifiedFormat(count, model_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    inputs[2].ToUnifiedFormat(count, x_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto models = UnifiedVectorFormat::GetData<string_t>(model_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(inputs[2]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[2]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);
    auto states = reinterpret_cast<ConformalState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto mi = model_data.sel->get_index(i);
        auto yi = y_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!model_data.validity.RowIsValid(mi) || !y_data.validity.RowIsValid(yi) ||
            !x_data.validity.RowIsValid(xi) || HasNullElement(child_data, lists[xi])) {
            continue;
        }
        const auto &entry = lists[xi];
        if (s.n_rows == 0) {
            *s.model = models[mi].GetString();
            s.n_cols = entry.length;
            if (input_count > 3) {
                auto oi = options_data.sel->get_index(i);
                if (options_data.validity.RowIsValid(oi)) {
                    *s.options = UnifiedVectorFormat::GetData<string_t>(options_data)[oi].GetString();
                }
            }
        } else if (models[mi].GetSize() != s.model->size() ||
                   memcmp(models[mi].GetData(), s.model->data(), s.model->size()) != 0) {
            throw InvalidInputException("%s: every row of a group must carry the same model", FUNCTION);
        } else if (entry.length != s.n_cols) {
            throw InvalidInputException("%s: feature lists of different lengths", FUNCTION);
        }
        for (idx_t j = 0; j < entry.length; j++) {
            s.x->push_back(values[child_data.sel->get_index(entry.offset + j)]);
        }
        if (LABELS) {
            s.labels->push_back(UnifiedVectorFormat::GetData<string_t>(y_data)[yi].GetString());
        } else {
            s.y->push_back(UnifiedVectorFormat::GetData<double>(y_data)[yi]);
        }
        s.n_rows++;
    }
}

template <const char *FUNCTION = kConformalize>
void ConformalCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<ConformalState *>(source_vector);
    auto targets = FlatVector::GetData<ConformalState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (src.n_rows == 0) {
            continue;
        }
        if (tgt.n_rows == 0) {
            *tgt.model = *src.model;
            *tgt.options = *src.options;
            tgt.n_cols = src.n_cols;
        } else if (*tgt.model != *src.model) {
            throw InvalidInputException("%s: every row of a group must carry the same model", FUNCTION);
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("%s: feature lists of different lengths", FUNCTION);
        }
        tgt.x->insert(tgt.x->end(), src.x->begin(), src.x->end());
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
        tgt.labels->insert(tgt.labels->end(), src.labels->begin(), src.labels->end());
        tgt.n_rows += src.n_rows;
    }
}

//===--------------------------------------------------------------------===//
// ml_warm_start_agg(model BLOB, y DOUBLE | VARCHAR, x DOUBLE[] [, params VARCHAR]) -> BLOB
// Continue training a boosting model on the group's rows.
//===--------------------------------------------------------------------===//
void WarmStartFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<ConformalState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_warm_start_agg");
        AnofoxMlBytes blob {nullptr, 0};
        std::vector<const char *> labels;
        for (const auto &l : *s.labels) {
            labels.push_back(l.c_str());
        }
        const bool has_labels = !s.labels->empty();
        Check(anofox_ml_warm_start(reinterpret_cast<const uint8_t *>(s.model->data()), s.model->size(),
                                   s.options->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                   has_labels ? nullptr : s.y->data(), has_labels ? labels.data() : nullptr,
                                   FitThreads(aggr), &blob),
              "ml_warm_start_agg");
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

void ConformalFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<ConformalState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_conformalize_agg");
        AnofoxMlBytes blob {nullptr, 0};
        auto model = reinterpret_cast<const uint8_t *>(s.model->data());
        if (!s.labels->empty()) {
            std::vector<const char *> labels;
            for (const auto &l : *s.labels) {
                labels.push_back(l.c_str());
            }
            Check(anofox_ml_conformalize_labels(model, s.model->size(), s.x->data(), s.n_rows, s.n_cols,
                                                labels.data(), &blob),
                  "ml_conformalize_agg");
        } else {
            Check(anofox_ml_conformalize(model, s.model->size(), s.x->data(), s.n_rows, s.n_cols, s.y->data(), &blob),
                  "ml_conformalize_agg");
        }
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

void ImportanceFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<ConformalState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.n_rows == 0) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        CaptureUse("ml_permutation_importance_agg");
        std::vector<const char *> labels;
        for (const auto &l : *s.labels) {
            labels.push_back(l.c_str());
        }
        AnofoxMlDoubles out {nullptr, 0};
        Check(anofox_ml_permutation_importance(reinterpret_cast<const uint8_t *>(s.model->data()), s.model->size(),
                                               s.x->data(), s.n_rows, s.n_cols,
                                               s.labels->empty() ? s.y->data() : nullptr,
                                               s.labels->empty() ? nullptr : labels.data(), s.options->c_str(), &out),
              "ml_permutation_importance_agg");
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + s.n_cols);
        auto &fields = StructVector::GetEntries(ListVector::GetEntry(result));
        auto feature = FlatVector::GetData<int32_t>(*fields[0]);
        auto mean = FlatVector::GetData<double>(*fields[1]);
        auto stddev = FlatVector::GetData<double>(*fields[2]);
        for (idx_t j = 0; j < s.n_cols; j++) {
            feature[list_offset + j] = static_cast<int32_t>(j + 1); // 1-based, as x[feature]
            mean[list_offset + j] = out.data[2 * j];
            stddev[list_offset + j] = out.data[2 * j + 1];
        }
        anofox_ml_free_doubles(out);
        ListVector::SetListSize(result, list_offset + s.n_cols);
        entries[row] = list_entry_t(list_offset, s.n_cols);
    }
}

//===--------------------------------------------------------------------===//
// ml_transform(encoder BLOB, x VARCHAR[]) -> DOUBLE[] (categorical encoders)
//===--------------------------------------------------------------------===//
void MlEncodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    idx_t count = args.size();
    UnifiedVectorFormat model_data, x_data;
    args.data[0].ToUnifiedFormat(count, model_data);
    args.data[1].ToUnifiedFormat(count, x_data);
    auto models = UnifiedVectorFormat::GetData<string_t>(model_data);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(args.data[1]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(args.data[1]), child_data);
    auto values = UnifiedVectorFormat::GetData<string_t>(child_data);
    result.SetVectorType(VectorType::FLAT_VECTOR);
    CaptureUse("ml_transform");

    std::vector<std::pair<idx_t, std::vector<double>>> rows_out;
    std::vector<idx_t> rows;
    std::vector<std::string> x;
    string_t current;
    idx_t n_cols = 0;
    auto flush = [&]() {
        if (rows.empty()) {
            return;
        }
        std::vector<const char *> ptrs;
        for (const auto &v : x) {
            ptrs.push_back(v.c_str());
        }
        AnofoxMlDoubles out {nullptr, 0};
        size_t out_cols = 0;
        Check(anofox_ml_encode(reinterpret_cast<const uint8_t *>(current.GetData()), current.GetSize(), ptrs.data(),
                               rows.size(), n_cols, &out, &out_cols),
              "ml_transform");
        for (idx_t r = 0; r < rows.size(); r++) {
            rows_out.emplace_back(rows[r], std::vector<double>(out.data + r * out_cols, out.data + (r + 1) * out_cols));
        }
        anofox_ml_free_doubles(out);
        rows.clear();
        x.clear();
    };
    for (idx_t i = 0; i < count; i++) {
        auto mi = model_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!model_data.validity.RowIsValid(mi) || !x_data.validity.RowIsValid(xi) ||
            HasNullElement(child_data, lists[xi])) {
            FlatVector::SetNull(result, i, true);
            continue;
        }
        const auto &model = models[mi];
        const auto &entry = lists[xi];
        bool same = !rows.empty() && model.GetSize() == current.GetSize() &&
                    memcmp(model.GetData(), current.GetData(), model.GetSize()) == 0;
        if (!same || entry.length != n_cols) {
            flush();
            current = model;
            n_cols = entry.length;
        }
        for (idx_t j = 0; j < entry.length; j++) {
            x.push_back(values[child_data.sel->get_index(entry.offset + j)].GetString());
        }
        rows.push_back(i);
    }
    flush();
    auto entries = FlatVector::GetData<list_entry_t>(result);
    idx_t offset = ListVector::GetListSize(result);
    for (auto &r : rows_out) {
        entries[r.first] = list_entry_t(offset, r.second.size());
        for (double d : r.second) {
            ListVector::PushBack(result, Value::DOUBLE(d));
        }
        offset += r.second.size();
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

//===--------------------------------------------------------------------===//
// ml_predict(model BLOB, x DOUBLE[]) -> DOUBLE
//===--------------------------------------------------------------------===//
void MlPredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto out = FlatVector::GetData<double>(result);
    ApplyBatched(args, "ml_predict", "predict", result, [&](idx_t row, const double *v, size_t n) {
        if (n != 1) {
            throw InvalidInputException("ml_predict: the model returns %llu values per row; use ml_transform",
                                        (unsigned long long)n);
        }
        out[row] = v[0];
    });
}

//===--------------------------------------------------------------------===//
// ml_predict_proba / ml_decision_function / ml_transform /
// ml_inverse_transform / ml_score_samples (model BLOB, x DOUBLE[]) -> DOUBLE[]
//===--------------------------------------------------------------------===//
scalar_function_t MakeListFunction(const char *function, const char *method) {
    return [function, method](DataChunk &args, ExpressionState &state, Vector &result) {
        // Collect per row first: the list child vector is appended to in row order.
        std::vector<std::pair<idx_t, std::vector<double>>> rows;
        ApplyBatched(args, function, method, result, [&](idx_t row, const double *v, size_t n) {
            rows.emplace_back(row, std::vector<double>(v, v + n));
        });
        auto entries = FlatVector::GetData<list_entry_t>(result);
        idx_t offset = ListVector::GetListSize(result);
        for (auto &r : rows) {
            entries[r.first] = list_entry_t(offset, r.second.size());
            for (double d : r.second) {
                ListVector::PushBack(result, Value::DOUBLE(d));
            }
            offset += r.second.size();
        }
        if (args.AllConstant()) {
            result.SetVectorType(VectorType::CONSTANT_VECTOR);
        }
    };
}

//===--------------------------------------------------------------------===//
// ml_model_fitted(model BLOB) -> VARCHAR (JSON of the fitted state)
//===--------------------------------------------------------------------===//
void MlModelFittedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t blob) {
        char *json = nullptr;
        Check(anofox_ml_model_fitted(reinterpret_cast<const uint8_t *>(blob.GetData()), blob.GetSize(), &json),
              "ml_model_fitted");
        return StringVector::AddString(result, TakeString(json));
    });
}

//===--------------------------------------------------------------------===//
// Fitted arrays: ml_coefficients(model) -> STRUCT(coef DOUBLE[][], intercept DOUBLE[]),
// ml_cluster_centers(model) -> DOUBLE[][], ml_explained_variance_ratio(model) -> DOUBLE[],
// ml_feature_importances(model) -> DOUBLE[]
//===--------------------------------------------------------------------===//
// Read a fitted array; false when `optional` and the model has none.
bool ModelArray(const char *function, string_t blob, const char *keys, bool optional, std::vector<double> &data,
                size_t &rows, size_t &cols) {
    AnofoxMlDoubles out {nullptr, 0};
    char *err = anofox_ml_model_array(reinterpret_cast<const uint8_t *>(blob.GetData()), blob.GetSize(), keys, &out,
                                      &rows, &cols);
    if (err) {
        if (optional) {
            anofox_ml_free_string(err);
            return false;
        }
        Check(err, function);
    }
    data.assign(out.data, out.data + out.len);
    anofox_ml_free_doubles(out);
    return true;
}

Value MatrixValue(const std::vector<double> &data, size_t rows, size_t cols) {
    const auto D = LogicalType(LogicalTypeId::DOUBLE);
    vector<Value> out;
    for (size_t r = 0; r < rows; r++) {
        vector<Value> row;
        for (size_t c = 0; c < cols; c++) {
            row.push_back(Value::DOUBLE(data[r * cols + c]));
        }
        out.push_back(Value::LIST(D, std::move(row)));
    }
    return Value::LIST(LogicalType::LIST(D), std::move(out));
}

Value ListValue(const std::vector<double> &data) {
    vector<Value> out;
    for (double d : data) {
        out.push_back(Value::DOUBLE(d));
    }
    return Value::LIST(LogicalType(LogicalTypeId::DOUBLE), std::move(out));
}

template <class MAKE>
void ModelValueFunction(DataChunk &args, Vector &result, MAKE &&make) {
    UnifiedVectorFormat model_data;
    args.data[0].ToUnifiedFormat(args.size(), model_data);
    auto models = UnifiedVectorFormat::GetData<string_t>(model_data);
    for (idx_t i = 0; i < args.size(); i++) {
        auto mi = model_data.sel->get_index(i);
        result.SetValue(i, model_data.validity.RowIsValid(mi) ? make(models[mi]) : Value(result.GetType()));
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

LogicalType CoefficientsType() {
    const auto D = LogicalType(LogicalTypeId::DOUBLE);
    child_list_t<LogicalType> fields;
    fields.emplace_back("coef", LogicalType::LIST(LogicalType::LIST(D)));
    fields.emplace_back("intercept", LogicalType::LIST(D));
    return LogicalType::STRUCT(std::move(fields));
}

void MlCoefficientsFunction(DataChunk &args, ExpressionState &, Vector &result) {
    ModelValueFunction(args, result, [](string_t blob) {
        std::vector<double> coef, intercept;
        size_t r = 0, c = 0, ir = 0, ic = 0;
        ModelArray("ml_coefficients", blob, "coef,weights", false, coef, r, c);
        bool has_intercept =
            ModelArray("ml_coefficients", blob, "intercept,intercepts,bias,biases", true, intercept, ir, ic);
        child_list_t<Value> fields;
        fields.emplace_back("coef", MatrixValue(coef, r, c));
        fields.emplace_back("intercept", has_intercept ? ListValue(intercept)
                                                       : Value(LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))));
        return Value::STRUCT(std::move(fields));
    });
}

void MlClusterCentersFunction(DataChunk &args, ExpressionState &, Vector &result) {
    ModelValueFunction(args, result, [](string_t blob) {
        std::vector<double> data;
        size_t r = 0, c = 0;
        ModelArray("ml_cluster_centers", blob, "centroids,cluster_centers,subcluster_centers,means", false, data, r, c);
        return MatrixValue(data, r, c);
    });
}

LogicalType CovarianceType() {
    const auto D = LogicalType(LogicalTypeId::DOUBLE);
    child_list_t<LogicalType> fields;
    fields.emplace_back("location", LogicalType::LIST(D));
    fields.emplace_back("covariance", LogicalType::LIST(LogicalType::LIST(D)));
    fields.emplace_back("precision", LogicalType::LIST(LogicalType::LIST(D)));
    return LogicalType::STRUCT(std::move(fields));
}

// ml_covariance(model) -> STRUCT(location, covariance, precision) of a
// covariance estimator (EmpiricalCovariance, LedoitWolf, MinCovDet, ...,
// EllipticEnvelope).
void MlCovarianceFunction(DataChunk &args, ExpressionState &, Vector &result) {
    ModelValueFunction(args, result, [](string_t blob) {
        std::vector<double> location, covariance, precision;
        size_t lr = 0, lc = 0, cr = 0, cc = 0, pr = 0, pc = 0;
        ModelArray("ml_covariance", blob, "location", false, location, lr, lc);
        ModelArray("ml_covariance", blob, "covariance", false, covariance, cr, cc);
        ModelArray("ml_covariance", blob, "precision", false, precision, pr, pc);
        child_list_t<Value> fields;
        fields.emplace_back("location", ListValue(location));
        fields.emplace_back("covariance", MatrixValue(covariance, cr, cc));
        fields.emplace_back("precision", MatrixValue(precision, pr, pc));
        return Value::STRUCT(std::move(fields));
    });
}

void MlFeatureImportancesFunction(DataChunk &args, ExpressionState &, Vector &result) {
    ModelValueFunction(args, result, [](string_t blob) {
        std::vector<double> data;
        size_t r = 0, c = 0;
        ModelArray("ml_feature_importances", blob, "feature_importances", false, data, r, c);
        return ListValue(data);
    });
}

void MlExplainedVarianceRatioFunction(DataChunk &args, ExpressionState &, Vector &result) {
    ModelValueFunction(args, result, [](string_t blob) {
        std::vector<double> data;
        size_t r = 0, c = 0;
        ModelArray("ml_explained_variance_ratio", blob, "explained_variance_ratio", false, data, r, c);
        return ListValue(data);
    });
}

//===--------------------------------------------------------------------===//
// ml_model_info(model BLOB) -> STRUCT(kind, n_features, classes, capabilities,
//                                     params, format_version, crate_version, labels,
//                                     n_iter, inertia, oob_score)
//===--------------------------------------------------------------------===//
LogicalType ModelInfoType() {
    child_list_t<LogicalType> fields;
    fields.emplace_back("kind", LogicalType(LogicalTypeId::VARCHAR));
    fields.emplace_back("n_features", LogicalType(LogicalTypeId::BIGINT));
    fields.emplace_back("classes", LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)));
    fields.emplace_back("capabilities", LogicalType::LIST(LogicalType(LogicalTypeId::VARCHAR)));
    fields.emplace_back("params", LogicalType(LogicalTypeId::VARCHAR));
    fields.emplace_back("format_version", LogicalType(LogicalTypeId::INTEGER));
    fields.emplace_back("crate_version", LogicalType(LogicalTypeId::VARCHAR));
    fields.emplace_back("labels", LogicalType::LIST(LogicalType(LogicalTypeId::VARCHAR)));
    fields.emplace_back("n_iter", LogicalType(LogicalTypeId::BIGINT));
    fields.emplace_back("inertia", LogicalType(LogicalTypeId::DOUBLE));
    fields.emplace_back("oob_score", LogicalType(LogicalTypeId::DOUBLE));
    return LogicalType::STRUCT(std::move(fields));
}

void MlModelInfoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    idx_t count = args.size();
    UnifiedVectorFormat model_data;
    args.data[0].ToUnifiedFormat(count, model_data);
    auto models = UnifiedVectorFormat::GetData<string_t>(model_data);
    for (idx_t i = 0; i < count; i++) {
        auto mi = model_data.sel->get_index(i);
        if (!model_data.validity.RowIsValid(mi)) {
            result.SetValue(i, Value(ModelInfoType()));
            continue;
        }
        AnofoxMlModelInfo info {};
        Check(anofox_ml_model_info(reinterpret_cast<const uint8_t *>(models[mi].GetData()), models[mi].GetSize(),
                                   &info),
              "ml_model_info");
        Value classes(LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)));
        if (info.classes.data) {
            vector<Value> c;
            for (size_t j = 0; j < info.classes.len; j++) {
                c.push_back(Value::DOUBLE(info.classes.data[j]));
            }
            classes = Value::LIST(LogicalType(LogicalTypeId::DOUBLE), std::move(c));
        }
        vector<Value> caps;
        std::string all(info.capabilities ? info.capabilities : "");
        for (size_t start = 0; start < all.size();) {
            size_t end = all.find('\n', start);
            if (end == std::string::npos) {
                end = all.size();
            }
            caps.push_back(Value(all.substr(start, end - start)));
            start = end + 1;
        }
        child_list_t<Value> fields;
        fields.emplace_back("kind", Value(info.kind ? info.kind : ""));
        fields.emplace_back("n_features", Value::BIGINT(info.n_features));
        fields.emplace_back("classes", std::move(classes));
        fields.emplace_back("capabilities", Value::LIST(LogicalType(LogicalTypeId::VARCHAR), std::move(caps)));
        fields.emplace_back("params", Value(info.params ? info.params : ""));
        fields.emplace_back("format_version", Value::INTEGER(info.format_version));
        fields.emplace_back("crate_version", Value(info.crate_version ? info.crate_version : ""));
        vector<Value> labels;
        for (size_t j = 0; j < info.labels.len; j++) {
            labels.push_back(Value(info.labels.data[j]));
        }
        fields.emplace_back("labels", Value::LIST(LogicalType(LogicalTypeId::VARCHAR), std::move(labels)));
        fields.emplace_back("n_iter", std::isnan(info.n_iter) ? Value(LogicalType(LogicalTypeId::BIGINT))
                                                              : Value::BIGINT(static_cast<int64_t>(info.n_iter)));
        auto nullable = [](double v) { return std::isnan(v) ? Value(LogicalType(LogicalTypeId::DOUBLE)) : Value::DOUBLE(v); };
        fields.emplace_back("inertia", nullable(info.inertia));
        fields.emplace_back("oob_score", nullable(info.oob_score));
        anofox_ml_free_model_info(info);
        result.SetValue(i, Value::STRUCT(std::move(fields)));
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

} // anonymous namespace

void RegisterMlFunctions(ExtensionLoader &loader) {
    loader.RegisterFunction(ScalarFunction("ml_version", {}, LogicalType(LogicalTypeId::VARCHAR), MlVersionFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_kind", {LogicalType(LogicalTypeId::BLOB)}, LogicalType(LogicalTypeId::VARCHAR), MlModelKindFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_info", {LogicalType(LogicalTypeId::BLOB)}, ModelInfoType(), MlModelInfoFunction));
    const auto BL = LogicalType(LogicalTypeId::BLOB), DL = LogicalType(LogicalTypeId::DOUBLE);
    loader.RegisterFunction(ScalarFunction("ml_coefficients", {BL}, CoefficientsType(), MlCoefficientsFunction));
    loader.RegisterFunction(ScalarFunction("ml_cluster_centers", {BL}, LogicalType::LIST(LogicalType::LIST(DL)),
                                           MlClusterCentersFunction));
    loader.RegisterFunction(ScalarFunction("ml_explained_variance_ratio", {BL}, LogicalType::LIST(DL),
                                           MlExplainedVarianceRatioFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_feature_importances", {BL}, LogicalType::LIST(DL), MlFeatureImportancesFunction));
    loader.RegisterFunction(ScalarFunction("ml_covariance", {BL}, CovarianceType(), MlCovarianceFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_fitted", {LogicalType(LogicalTypeId::BLOB)}, LogicalType(LogicalTypeId::VARCHAR), MlModelFittedFunction));
    const vector<LogicalType> model_and_x {LogicalType(LogicalTypeId::BLOB), LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))};
    loader.RegisterFunction(ScalarFunction("ml_predict", model_and_x, LogicalType(LogicalTypeId::DOUBLE), MlPredictFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_predict_label", model_and_x, LogicalType(LogicalTypeId::VARCHAR), MlPredictLabelFunction));
    const vector<LogicalType> model_x_alpha {LogicalType(LogicalTypeId::BLOB), LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)),
                                             LogicalType(LogicalTypeId::DOUBLE)};
    loader.RegisterFunction(
        ScalarFunction("ml_predict_interval", model_x_alpha, IntervalType(), MlPredictIntervalFunction));
    loader.RegisterFunction(ScalarFunction("ml_predict_set", model_x_alpha, LogicalType::LIST(LogicalType(LogicalTypeId::VARCHAR)),
                                           MlPredictSetFunction));
    AggregateFunctionSet conformal_set("ml_conformalize_agg");
    for (bool labels : {false, true}) {
        AggregateFunction f("ml_conformalize_agg",
                            {LogicalType(LogicalTypeId::BLOB), LogicalType(labels ? LogicalTypeId::VARCHAR : LogicalTypeId::DOUBLE),
                             LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))},
                            LogicalType(LogicalTypeId::BLOB), AggregateFunction::StateSize<ConformalState>, ConformalInitialize,
                            labels ? ConformalUpdate<true> : ConformalUpdate<false>, ConformalCombine<>,
                            ConformalFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr,
                            ConformalDestroy);
        conformal_set.AddFunction(f);
    }
    loader.RegisterFunction(conformal_set);
    AggregateFunctionSet warm_set("ml_warm_start_agg");
    for (bool labels : {false, true}) {
        for (bool params : {false, true}) {
            vector<LogicalType> args {LogicalType(LogicalTypeId::BLOB),
                                      LogicalType(labels ? LogicalTypeId::VARCHAR : LogicalTypeId::DOUBLE),
                                      LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))};
            if (params) {
                args.push_back(LogicalType(LogicalTypeId::VARCHAR));
            }
            AggregateFunction f("ml_warm_start_agg", args, LogicalType(LogicalTypeId::BLOB),
                                AggregateFunction::StateSize<ConformalState>, ConformalInitialize,
                                labels ? ConformalUpdate<true, kWarmStart> : ConformalUpdate<false, kWarmStart>,
                                ConformalCombine<kWarmStart>, WarmStartFinalize, FunctionNullHandling::SPECIAL_HANDLING,
                                nullptr, FitBind, ConformalDestroy);
            warm_set.AddFunction(f);
        }
    }
    loader.RegisterFunction(warm_set);

    child_list_t<LogicalType> importance_fields;
    importance_fields.emplace_back("feature", LogicalType(LogicalTypeId::INTEGER));
    importance_fields.emplace_back("importance_mean", LogicalType(LogicalTypeId::DOUBLE));
    importance_fields.emplace_back("importance_std", LogicalType(LogicalTypeId::DOUBLE));
    const auto importance_type = LogicalType::LIST(LogicalType::STRUCT(std::move(importance_fields)));
    AggregateFunctionSet importance_set("ml_permutation_importance_agg");
    for (bool labels : {false, true}) {
        for (bool options : {false, true}) {
            vector<LogicalType> args {LogicalType(LogicalTypeId::BLOB),
                                      LogicalType(labels ? LogicalTypeId::VARCHAR : LogicalTypeId::DOUBLE),
                                      LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))};
            if (options) {
                args.push_back(LogicalType(LogicalTypeId::VARCHAR));
            }
            AggregateFunction f("ml_permutation_importance_agg", args, importance_type,
                                AggregateFunction::StateSize<ConformalState>, ConformalInitialize,
                                labels ? ConformalUpdate<true> : ConformalUpdate<false>, ConformalCombine<>,
                                ImportanceFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr,
                                ConformalDestroy);
            importance_set.AddFunction(f);
        }
    }
    loader.RegisterFunction(importance_set);
    const std::pair<const char *, const char *> list_functions[] = {
        {"ml_predict_proba", "predict_proba"}, {"ml_decision_function", "decision_function"},
{"ml_predict_log_proba", "predict_log_proba"}, {"ml_inverse_transform", "inverse_transform"},
        {"ml_score_samples", "score_samples"}, {"ml_staged_predict", "staged_predict"},
    };
    for (const auto &f : list_functions) {
        loader.RegisterFunction(ScalarFunction(f.first, model_and_x, LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)),
                                               MakeListFunction(f.first, f.second)));
    }
    ScalarFunctionSet transform_set("ml_transform");
    transform_set.AddFunction(ScalarFunction(model_and_x, LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)),
                                             MakeListFunction("ml_transform", "transform")));
    transform_set.AddFunction(ScalarFunction({LogicalType(LogicalTypeId::BLOB),
                                              LogicalType::LIST(LogicalType(LogicalTypeId::VARCHAR))},
                                             LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)), MlEncodeFunction));
    loader.RegisterFunction(transform_set);
    loader.RegisterFunction(TableFunction("ml_estimators", {}, EstimatorsFunction, EstimatorsBind, EstimatorsInit));

    AggregateFunctionSet fit_set("ml_fit_agg");
    auto make_fit = [](vector<LogicalType> args, aggregate_update_t update) {
        AggregateFunction f("ml_fit_agg", std::move(args), LogicalType(LogicalTypeId::BLOB), AggregateFunction::StateSize<FitState>,
                            FitInitialize, update, FitCombine, FitFinalize, nullptr, FitBind, FitDestroy);
        f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
        return f;
    };
    const auto features = LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE));
    fit_set.AddFunction(make_fit({LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features}, FitUpdate<true>));
    fit_set.AddFunction(
        make_fit({LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features, LogicalType(LogicalTypeId::VARCHAR)}, FitUpdate<true>));
    fit_set.AddFunction(make_fit({LogicalType(LogicalTypeId::VARCHAR), features}, FitUpdate<false>));
    fit_set.AddFunction(make_fit({LogicalType(LogicalTypeId::VARCHAR), features, LogicalType(LogicalTypeId::VARCHAR)}, FitUpdate<false>));
    fit_set.AddFunction(make_fit(
        {LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features, LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE)},
        FitUpdate<true, true>));
    fit_set.AddFunction(
        make_fit({LogicalType(LogicalTypeId::VARCHAR), features, LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE)}, FitUpdate<false, true>));
    // String class labels.
    const auto V = LogicalType(LogicalTypeId::VARCHAR);
    fit_set.AddFunction(make_fit({V, V, features}, FitUpdate<true, false, true>));
    fit_set.AddFunction(make_fit({V, V, features, V}, FitUpdate<true, false, true>));
    fit_set.AddFunction(make_fit({V, V, features, V, LogicalType(LogicalTypeId::DOUBLE)}, FitUpdate<true, true, true>));
    AggregateFunctionSet tune_set("ml_tune_agg");
    for (bool labels : {false, true}) {
        for (bool options : {false, true}) {
            vector<LogicalType> args {V, LogicalType(labels ? LogicalTypeId::VARCHAR : LogicalTypeId::DOUBLE), features,
                                      V};
            if (options) {
                args.push_back(V);
            }
            AggregateFunction f("ml_tune_agg", args, TuneType(), AggregateFunction::StateSize<FitState>,
                                FitInitialize, labels ? TuneUpdate<true> : TuneUpdate<false>, FitCombine,
                                TuneFinalize, nullptr, FitBind, FitDestroy);
            f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
            tune_set.AddFunction(f);
        }
    }
    loader.RegisterFunction(tune_set);
    // Categorical encoders on VARCHAR features.
    const auto string_features = LogicalType::LIST(V);
    fit_set.AddFunction(make_fit({V, string_features}, EncodeUpdate));
    fit_set.AddFunction(make_fit({V, string_features, V}, EncodeUpdate));
    loader.RegisterFunction(fit_set);

    AggregateFunctionSet validation_set("ml_fit_with_validation_agg");
    const auto B = LogicalType(LogicalTypeId::BOOLEAN), D = LogicalType(LogicalTypeId::DOUBLE);
    auto make_validation = [](vector<LogicalType> args, aggregate_update_t update) {
        AggregateFunction f("ml_fit_with_validation_agg", std::move(args), LogicalType(LogicalTypeId::BLOB),
                            AggregateFunction::StateSize<FitState>, FitInitialize, update, FitCombine,
                            FitValidationFinalize, nullptr, FitBind, FitDestroy);
        f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
        return f;
    };
    validation_set.AddFunction(make_validation({V, D, features, B}, FitUpdate<true, false, false, false, true>));
    validation_set.AddFunction(make_validation({V, D, features, B, V}, FitUpdate<true, false, false, false, true>));
    validation_set.AddFunction(make_validation({V, D, features, B, V, D}, FitUpdate<true, true, false, false, true>));
    validation_set.AddFunction(make_validation({V, V, features, B}, FitUpdate<true, false, true, false, true>));
    validation_set.AddFunction(make_validation({V, V, features, B, V}, FitUpdate<true, false, true, false, true>));
    validation_set.AddFunction(make_validation({V, V, features, B, V, D}, FitUpdate<true, true, true, false, true>));
    loader.RegisterFunction(validation_set);

    using Overload = std::pair<vector<LogicalType>, aggregate_update_t>;
    auto make_rows = [&](const char *name, LogicalType result, aggregate_finalize_t finalize,
                         const vector<Overload> &overloads) {
        AggregateFunctionSet set(name);
        for (const auto &o : overloads) {
            AggregateFunction f(name, o.first, result, AggregateFunction::StateSize<FitState>, FitInitialize,
                                o.second, FitCombine, finalize, nullptr, FitBind, FitDestroy);
            f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
            set.AddFunction(f);
        }
        loader.RegisterFunction(set);
    };
    const vector<Overload> unsupervised {
        {{LogicalType(LogicalTypeId::VARCHAR), features}, FitUpdate<false>},
        {{LogicalType(LogicalTypeId::VARCHAR), features, LogicalType(LogicalTypeId::VARCHAR)}, FitUpdate<false>},
        {{LogicalType(LogicalTypeId::VARCHAR), features, LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE)}, FitUpdate<false, true>},
    };
    make_rows(kClusterAgg, LogicalType::LIST(LogicalType(LogicalTypeId::BIGINT)),
              RowsFinalize<kClusterAgg, kClusterMethods, int64_t>, unsupervised);
    make_rows(kOutlierAgg, LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)),
              RowsFinalize<kOutlierAgg, kOutlierMethods, double>, unsupervised);
    auto fit_transform = unsupervised;
    fit_transform.push_back({{LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features}, FitUpdate<true>});
    fit_transform.push_back(
        {{LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features, LogicalType(LogicalTypeId::VARCHAR)}, FitUpdate<true>});
    fit_transform.push_back(
        {{LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE), features, LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::DOUBLE)},
         FitUpdate<true, true>});
    make_rows("ml_fit_transform_agg", LogicalType::LIST(LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE))),
              FitTransformFinalize, fit_transform);
}

} // namespace duckdb
