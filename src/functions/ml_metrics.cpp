// Metric aggregates of the anofox_ml extension (ml_r2, ml_f1, ml_roc_auc, ...):
// collect the group, then call anofox_ml_metric once in finalize.

#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"

namespace duckdb {

namespace {

struct MetricState {
    std::vector<double> *y;
    std::vector<double> *p; // n_rows x n_cols, row-major
    std::vector<double> *w;
    std::string *options;
    idx_t n_cols;
    bool has_w;
};

void MetricInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<MetricState *>(state_p);
    s.y = new std::vector<double>();
    s.p = new std::vector<double>();
    s.w = new std::vector<double>();
    s.options = new std::string();
    s.n_cols = 0;
    s.has_w = false;
}

void MetricDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<MetricState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->y;
        delete states[i]->p;
        delete states[i]->w;
        delete states[i]->options;
    }
}

// How the arguments map onto (y_true, y_pred rows, options, weight).
enum class Shape {
    PAIR,       // (y DOUBLE, pred DOUBLE | DOUBLE[] [, options VARCHAR [, weight DOUBLE]])
    SILHOUETTE, // (x DOUBLE[], label DOUBLE)
    INTERVAL,   // (y DOUBLE, lower DOUBLE, upper DOUBLE [, alpha DOUBLE])
};

struct Metric {
    const char *function;
    const char *name; // metric name in anofox_ml_metric
};

template <Shape SHAPE, bool LIST_PRED>
void MetricUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat a, b, c, d, sdata;
    inputs[0].ToUnifiedFormat(count, a);
    inputs[1].ToUnifiedFormat(count, b);
    if (input_count > 2) {
        inputs[2].ToUnifiedFormat(count, c);
    }
    if (input_count > 3) {
        inputs[3].ToUnifiedFormat(count, d);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<MetricState **>(sdata.data);

    // The list argument: pred for PAIR with LIST_PRED, x for SILHOUETTE.
    constexpr bool HAS_LIST = SHAPE == Shape::SILHOUETTE || LIST_PRED;
    constexpr idx_t LIST_ARG = SHAPE == Shape::SILHOUETTE ? 0 : 1;
    constexpr idx_t Y_ARG = SHAPE == Shape::SILHOUETTE ? 1 : 0;
    UnifiedVectorFormat child_data;
    const double *child_values = nullptr;
    if (HAS_LIST) {
        auto &child = ListVector::GetEntry(inputs[LIST_ARG]);
        child.ToUnifiedFormat(ListVector::GetListSize(inputs[LIST_ARG]), child_data);
        child_values = UnifiedVectorFormat::GetData<double>(child_data);
    }
    auto &y_fmt = Y_ARG == 0 ? a : b;
    auto &list_fmt = LIST_ARG == 0 ? a : b;
    auto ys = UnifiedVectorFormat::GetData<double>(y_fmt);

    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto yi = y_fmt.sel->get_index(i);
        auto bi = b.sel->get_index(i);
        auto ai = a.sel->get_index(i);
        if (!a.validity.RowIsValid(ai) || !b.validity.RowIsValid(bi)) {
            continue; // rows with a NULL argument are skipped
        }
        std::vector<double> row;
        if (HAS_LIST) {
            auto li = list_fmt.sel->get_index(i);
            const auto &entry = UnifiedVectorFormat::GetData<list_entry_t>(list_fmt)[li];
            bool null_element = false;
            for (idx_t j = 0; j < entry.length; j++) {
                auto vi = child_data.sel->get_index(entry.offset + j);
                if (!child_data.validity.RowIsValid(vi)) {
                    null_element = true;
                    break;
                }
                row.push_back(child_values[vi]);
            }
            if (null_element) {
                continue;
            }
        } else {
            row.push_back(UnifiedVectorFormat::GetData<double>(b)[bi]);
        }
        if (SHAPE == Shape::INTERVAL) {
            auto ci = c.sel->get_index(i);
            if (!c.validity.RowIsValid(ci)) {
                continue;
            }
            row.push_back(UnifiedVectorFormat::GetData<double>(c)[ci]);
            if (input_count > 3 && s.options->empty()) {
                auto di = d.sel->get_index(i);
                if (d.validity.RowIsValid(di)) {
                    *s.options =
                        StringUtil::Format("{\"alpha\": %.17g}", UnifiedVectorFormat::GetData<double>(d)[di]);
                }
            }
        }
        double weight = 1.0;
        if (SHAPE == Shape::PAIR) {
            if (input_count > 2 && s.options->empty()) {
                auto ci = c.sel->get_index(i);
                if (c.validity.RowIsValid(ci)) {
                    *s.options = UnifiedVectorFormat::GetData<string_t>(c)[ci].GetString();
                }
            }
            if (input_count > 3) {
                auto di = d.sel->get_index(i);
                if (!d.validity.RowIsValid(di)) {
                    continue;
                }
                weight = UnifiedVectorFormat::GetData<double>(d)[di];
                s.has_w = true;
            }
        }
        if (s.y->empty()) {
            s.n_cols = row.size();
        } else if (row.size() != s.n_cols) {
            throw InvalidInputException("every row must have %llu values, got %llu", (unsigned long long)s.n_cols,
                                        (unsigned long long)row.size());
        }
        s.y->push_back(ys[yi]);
        s.p->insert(s.p->end(), row.begin(), row.end());
        if (s.has_w) {
            s.w->push_back(weight);
        }
    }
}

void MetricCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<MetricState *>(source_vector);
    auto targets = FlatVector::GetData<MetricState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (src.y->empty()) {
            continue;
        }
        if (tgt.y->empty()) {
            tgt.n_cols = src.n_cols;
            tgt.has_w = src.has_w;
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("rows of different lengths");
        }
        if (tgt.options->empty()) {
            *tgt.options = *src.options;
        }
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
        tgt.p->insert(tgt.p->end(), src.p->begin(), src.p->end());
        tgt.w->insert(tgt.w->end(), src.w->begin(), src.w->end());
    }
}

// The metric of an aggregate: attached as function_info at registration and
// handed to finalize as bind data.
struct MetricInfo : public AggregateFunctionInfo {
    explicit MetricInfo(Metric metric_p) : metric(metric_p) {
    }
    Metric metric;
};

struct MetricBindData : public FunctionData {
    explicit MetricBindData(Metric metric_p) : metric(metric_p) {
    }
    Metric metric;
    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<MetricBindData>(metric);
    }
    bool Equals(const FunctionData &other) const override {
        return std::string(metric.name) == other.Cast<MetricBindData>().metric.name;
    }
};

unique_ptr<FunctionData> MetricBind(ClientContext &, AggregateFunction &function, vector<unique_ptr<Expression>> &) {
    return make_uniq<MetricBindData>(function.function_info->Cast<MetricInfo>().metric);
}

void MetricFinalize(Vector &state_vector, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    auto &metric = aggr.bind_data->Cast<MetricBindData>().metric;
    auto states = FlatVector::GetData<MetricState *>(state_vector);
    auto out = FlatVector::GetData<double>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.y->empty()) {
            FlatVector::SetNull(result, row, true); // empty group
            continue;
        }
        char *err = anofox_ml_metric(metric.name, s.options->c_str(), s.y->data(), s.p->data(), s.y->size(),
                                     s.n_cols, s.has_w ? s.w->data() : nullptr, &out[row]);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("%s: %s", metric.function, msg);
        }
    }
}

} // anonymous namespace

void RegisterMlMetrics(ExtensionLoader &loader) {
    const auto D = LogicalType::DOUBLE;
    const auto L = LogicalType::LIST(LogicalType::DOUBLE);
    const auto V = LogicalType::VARCHAR;


    // Metrics differ only in name, argument shape and the optional arguments
    // they accept; the metric name travels as bind data.
    struct Spec {
        Metric metric;
        Shape shape;
        bool scalar_pred; // pred may be DOUBLE
        bool list_pred;   // pred may be DOUBLE[] (probabilities per class)
        bool options;     // accepts an options JSON
        bool weight;      // accepts a sample weight
    };
    const Spec specs[] = {
        // regression: (y, y_pred)
        {{"ml_r2", "r2"}, Shape::PAIR, true, false, false, false},
        {{"ml_mse", "mse"}, Shape::PAIR, true, false, false, false},
        {{"ml_rmse", "rmse"}, Shape::PAIR, true, false, false, false},
        {{"ml_mae", "mae"}, Shape::PAIR, true, false, false, false},
        {{"ml_mape", "mape"}, Shape::PAIR, true, false, false, false},
        {{"ml_median_absolute_error", "median_absolute_error"}, Shape::PAIR, true, false, false, false},
        {{"ml_max_error", "max_error"}, Shape::PAIR, true, false, false, false},
        {{"ml_explained_variance", "explained_variance"}, Shape::PAIR, true, false, false, false},
        {{"ml_mean_squared_log_error", "mean_squared_log_error"}, Shape::PAIR, true, false, false, false},
        // classification on labels: (y, y_pred [, options [, weight]])
        {{"ml_accuracy", "accuracy"}, Shape::PAIR, true, false, true, true},
        {{"ml_precision", "precision"}, Shape::PAIR, true, false, true, true},
        {{"ml_recall", "recall"}, Shape::PAIR, true, false, true, true},
        {{"ml_f1", "f1"}, Shape::PAIR, true, false, true, true},
        {{"ml_fbeta", "fbeta"}, Shape::PAIR, true, false, true, true},
        {{"ml_balanced_accuracy", "balanced_accuracy"}, Shape::PAIR, true, false, false, false},
        {{"ml_matthews_corrcoef", "matthews_corrcoef"}, Shape::PAIR, true, false, false, false},
        {{"ml_cohen_kappa", "cohen_kappa"}, Shape::PAIR, true, false, false, false},
        // scores and probabilities
        {{"ml_roc_auc", "roc_auc"}, Shape::PAIR, true, true, true, true},
        {{"ml_log_loss", "log_loss"}, Shape::PAIR, true, true, true, true},
        {{"ml_brier_score", "brier_score"}, Shape::PAIR, true, false, true, true},
        {{"ml_average_precision", "average_precision"}, Shape::PAIR, true, false, false, false},
        // clustering
        {{"ml_adjusted_rand", "adjusted_rand"}, Shape::PAIR, true, false, false, false},
        {{"ml_normalized_mutual_info", "normalized_mutual_info"}, Shape::PAIR, true, false, false, false},
        {{"ml_silhouette", "silhouette"}, Shape::SILHOUETTE, false, false, false, false},
        // prediction intervals
        {{"ml_coverage", "coverage"}, Shape::INTERVAL, false, false, false, false},
        {{"ml_interval_score", "interval_score"}, Shape::INTERVAL, false, false, false, false},
    };

    for (const auto &spec : specs) {
        AggregateFunctionSet set(spec.metric.function);
        auto add = [&](vector<LogicalType> args, aggregate_update_t update) {
            AggregateFunction f(spec.metric.function, std::move(args), LogicalType::DOUBLE,
                                AggregateFunction::StateSize<MetricState>, MetricInitialize, update, MetricCombine,
                                MetricFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, MetricBind,
                                MetricDestroy);
            f.function_info = make_shared_ptr<MetricInfo>(spec.metric);
            f.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
            set.AddFunction(f);
        };
        switch (spec.shape) {
        case Shape::PAIR:
            for (bool list : {false, true}) {
                if ((list && !spec.list_pred) || (!list && !spec.scalar_pred)) {
                    continue;
                }
                auto pred = list ? L : D;
                auto update = list ? MetricUpdate<Shape::PAIR, true> : MetricUpdate<Shape::PAIR, false>;
                add({D, pred}, update);
                if (spec.options) {
                    add({D, pred, V}, update);
                }
                if (spec.weight) {
                    add({D, pred, V, D}, update);
                }
            }
            break;
        case Shape::SILHOUETTE:
            add({L, D}, MetricUpdate<Shape::SILHOUETTE, false>);
            break;
        case Shape::INTERVAL:
            add({D, D, D}, MetricUpdate<Shape::INTERVAL, false>);
            add({D, D, D, D}, MetricUpdate<Shape::INTERVAL, false>);
            break;
        }
        loader.RegisterFunction(set);
    }
}

} // namespace duckdb
