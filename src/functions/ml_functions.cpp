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
    bool has_labels;
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
    s.has_labels = false;
    s.used = new std::vector<uint8_t>();
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
        delete s.used;
    }
}

// Inputs: kind, [y,] x [, params [, weight]]; y is a string label when LABELS.
template <bool HAS_Y, bool HAS_W = false, bool LABELS = false>
void FitUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    constexpr idx_t X = HAS_Y ? 2 : 1;
    UnifiedVectorFormat kind_data, y_data, x_data, params_data, w_data, sdata;
    inputs[0].ToUnifiedFormat(count, kind_data);
    if (HAS_Y) {
        inputs[1].ToUnifiedFormat(count, y_data);
    }
    inputs[X].ToUnifiedFormat(count, x_data);
    const bool has_params = input_count > X + 1;
    if (has_params) {
        inputs[X + 1].ToUnifiedFormat(count, params_data);
    }
    if (HAS_W) {
        inputs[X + 2].ToUnifiedFormat(count, w_data);
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
        if (!kind_data.validity.RowIsValid(ki) || (HAS_Y && !y_data.validity.RowIsValid(yi)) ||
            !x_data.validity.RowIsValid(xi) || (HAS_W && !w_data.validity.RowIsValid(wi))) {
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
            s.labels->push_back(UnifiedVectorFormat::GetData<string_t>(y_data)[yi].GetString());
        } else if (HAS_Y) {
            s.y->push_back(UnifiedVectorFormat::GetData<double>(y_data)[yi]);
        }
        if (HAS_W) {
            s.w->push_back(UnifiedVectorFormat::GetData<double>(w_data)[wi]);
        }
        s.n_rows++;
        s.used->push_back(1);
    }
}

void FitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
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
        tgt.n_rows += src.n_rows;
    }
}

void FitFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
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
        if (s.has_labels) {
            std::vector<const char *> labels;
            labels.reserve(s.labels->size());
            for (const auto &l : *s.labels) {
                labels.push_back(l.c_str());
            }
            Check(anofox_ml_fit_labels(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                       labels.data(), w, &blob),
                  "ml_fit_agg");
        } else {
            Check(anofox_ml_fit(s.kind->c_str(), s.params->c_str(), s.x->data(), s.n_rows, s.n_cols,
                                s.has_y ? s.y->data() : nullptr, w, &blob),
                  "ml_fit_agg");
        }
        FlatVector::GetData<string_t>(result)[row] =
            StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(blob.data), blob.len);
        anofox_ml_free_bytes(blob);
    }
}

//===--------------------------------------------------------------------===//
// ml_cluster_agg(kind VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> BIGINT[]
// ml_outlier_score_agg(kind VARCHAR, x DOUBLE[] [, params VARCHAR [, weight DOUBLE]]) -> DOUBLE[]
// Fit on the group and return one value per input row, in input order (use
// ORDER BY inside the aggregate to fix it); skipped rows get NULL.
//===--------------------------------------------------------------------===//
template <const char *FUNCTION, const char *METHODS, class T>
void RowsFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
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
                                  s.has_y ? s.y->data() : nullptr, s.has_w ? s.w->data() : nullptr, METHODS, &out,
                                  &out_cols),
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
void FitTransformFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
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
                                  "embedding,transform", &out, &out_cols),
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
// ml_model_info(model BLOB) -> STRUCT(kind, n_features, classes, capabilities,
//                                     params, format_version, crate_version)
//===--------------------------------------------------------------------===//
LogicalType ModelInfoType() {
    child_list_t<LogicalType> fields;
    fields.emplace_back("kind", LogicalType::VARCHAR);
    fields.emplace_back("n_features", LogicalType::BIGINT);
    fields.emplace_back("classes", LogicalType::LIST(LogicalType::DOUBLE));
    fields.emplace_back("capabilities", LogicalType::LIST(LogicalType::VARCHAR));
    fields.emplace_back("params", LogicalType::VARCHAR);
    fields.emplace_back("format_version", LogicalType::INTEGER);
    fields.emplace_back("crate_version", LogicalType::VARCHAR);
    fields.emplace_back("labels", LogicalType::LIST(LogicalType::VARCHAR));
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
        Value classes(LogicalType::LIST(LogicalType::DOUBLE));
        if (info.classes.data) {
            vector<Value> c;
            for (size_t j = 0; j < info.classes.len; j++) {
                c.push_back(Value::DOUBLE(info.classes.data[j]));
            }
            classes = Value::LIST(LogicalType::DOUBLE, std::move(c));
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
        fields.emplace_back("capabilities", Value::LIST(LogicalType::VARCHAR, std::move(caps)));
        fields.emplace_back("params", Value(info.params ? info.params : ""));
        fields.emplace_back("format_version", Value::INTEGER(info.format_version));
        fields.emplace_back("crate_version", Value(info.crate_version ? info.crate_version : ""));
        vector<Value> labels;
        for (size_t j = 0; j < info.labels.len; j++) {
            labels.push_back(Value(info.labels.data[j]));
        }
        fields.emplace_back("labels", Value::LIST(LogicalType::VARCHAR, std::move(labels)));
        anofox_ml_free_model_info(info);
        result.SetValue(i, Value::STRUCT(std::move(fields)));
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

} // anonymous namespace

void RegisterMlFunctions(ExtensionLoader &loader) {
    loader.RegisterFunction(ScalarFunction("ml_version", {}, LogicalType::VARCHAR, MlVersionFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_kind", {LogicalType::BLOB}, LogicalType::VARCHAR, MlModelKindFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_info", {LogicalType::BLOB}, ModelInfoType(), MlModelInfoFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_model_fitted", {LogicalType::BLOB}, LogicalType::VARCHAR, MlModelFittedFunction));
    const vector<LogicalType> model_and_x {LogicalType::BLOB, LogicalType::LIST(LogicalType::DOUBLE)};
    loader.RegisterFunction(ScalarFunction("ml_predict", model_and_x, LogicalType::DOUBLE, MlPredictFunction));
    loader.RegisterFunction(
        ScalarFunction("ml_predict_label", model_and_x, LogicalType::VARCHAR, MlPredictLabelFunction));
    const std::pair<const char *, const char *> list_functions[] = {
        {"ml_predict_proba", "predict_proba"}, {"ml_decision_function", "decision_function"},
        {"ml_transform", "transform"},         {"ml_inverse_transform", "inverse_transform"},
        {"ml_score_samples", "score_samples"},
    };
    for (const auto &f : list_functions) {
        loader.RegisterFunction(ScalarFunction(f.first, model_and_x, LogicalType::LIST(LogicalType::DOUBLE),
                                               MakeListFunction(f.first, f.second)));
    }
    loader.RegisterFunction(TableFunction("ml_estimators", {}, EstimatorsFunction, EstimatorsBind, EstimatorsInit));

    AggregateFunctionSet fit_set("ml_fit_agg");
    auto make_fit = [](vector<LogicalType> args, aggregate_update_t update) {
        AggregateFunction f("ml_fit_agg", std::move(args), LogicalType::BLOB, AggregateFunction::StateSize<FitState>,
                            FitInitialize, update, FitCombine, FitFinalize, nullptr, nullptr, FitDestroy);
        f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
        return f;
    };
    const auto features = LogicalType::LIST(LogicalType::DOUBLE);
    fit_set.AddFunction(make_fit({LogicalType::VARCHAR, LogicalType::DOUBLE, features}, FitUpdate<true>));
    fit_set.AddFunction(
        make_fit({LogicalType::VARCHAR, LogicalType::DOUBLE, features, LogicalType::VARCHAR}, FitUpdate<true>));
    fit_set.AddFunction(make_fit({LogicalType::VARCHAR, features}, FitUpdate<false>));
    fit_set.AddFunction(make_fit({LogicalType::VARCHAR, features, LogicalType::VARCHAR}, FitUpdate<false>));
    fit_set.AddFunction(make_fit(
        {LogicalType::VARCHAR, LogicalType::DOUBLE, features, LogicalType::VARCHAR, LogicalType::DOUBLE},
        FitUpdate<true, true>));
    fit_set.AddFunction(
        make_fit({LogicalType::VARCHAR, features, LogicalType::VARCHAR, LogicalType::DOUBLE}, FitUpdate<false, true>));
    // String class labels.
    const auto V = LogicalType::VARCHAR;
    fit_set.AddFunction(make_fit({V, V, features}, FitUpdate<true, false, true>));
    fit_set.AddFunction(make_fit({V, V, features, V}, FitUpdate<true, false, true>));
    fit_set.AddFunction(make_fit({V, V, features, V, LogicalType::DOUBLE}, FitUpdate<true, true, true>));
    loader.RegisterFunction(fit_set);

    using Overload = std::pair<vector<LogicalType>, aggregate_update_t>;
    auto make_rows = [&](const char *name, LogicalType result, aggregate_finalize_t finalize,
                         const vector<Overload> &overloads) {
        AggregateFunctionSet set(name);
        for (const auto &o : overloads) {
            AggregateFunction f(name, o.first, result, AggregateFunction::StateSize<FitState>, FitInitialize,
                                o.second, FitCombine, finalize, nullptr, nullptr, FitDestroy);
            f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
            set.AddFunction(f);
        }
        loader.RegisterFunction(set);
    };
    const vector<Overload> unsupervised {
        {{LogicalType::VARCHAR, features}, FitUpdate<false>},
        {{LogicalType::VARCHAR, features, LogicalType::VARCHAR}, FitUpdate<false>},
        {{LogicalType::VARCHAR, features, LogicalType::VARCHAR, LogicalType::DOUBLE}, FitUpdate<false, true>},
    };
    make_rows(kClusterAgg, LogicalType::LIST(LogicalType::BIGINT),
              RowsFinalize<kClusterAgg, kClusterMethods, int64_t>, unsupervised);
    make_rows(kOutlierAgg, LogicalType::LIST(LogicalType::DOUBLE),
              RowsFinalize<kOutlierAgg, kOutlierMethods, double>, unsupervised);
    auto fit_transform = unsupervised;
    fit_transform.push_back({{LogicalType::VARCHAR, LogicalType::DOUBLE, features}, FitUpdate<true>});
    fit_transform.push_back(
        {{LogicalType::VARCHAR, LogicalType::DOUBLE, features, LogicalType::VARCHAR}, FitUpdate<true>});
    fit_transform.push_back(
        {{LogicalType::VARCHAR, LogicalType::DOUBLE, features, LogicalType::VARCHAR, LogicalType::DOUBLE},
         FitUpdate<true, true>});
    make_rows("ml_fit_transform_agg", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
              FitTransformFinalize, fit_transform);
}

} // namespace duckdb
