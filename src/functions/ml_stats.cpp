// Statistic aggregates over feature lists:
//   ml_mutual_info_agg(y DOUBLE | VARCHAR, x DOUBLE[] [, options]) -> DOUBLE[]
//   ml_covariance_agg(x DOUBLE[] [, options]) -> STRUCT(covariance DOUBLE[][], shrinkage DOUBLE)
//   ml_estimate_bandwidth_agg(x DOUBLE[] [, options]) -> DOUBLE
// Rows with a NULL argument or a NULL feature are skipped; an empty group gives NULL.

#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"

namespace duckdb {

namespace {

struct StatState {
    std::vector<double> *x;
    std::vector<double> *y;
    std::vector<std::string> *labels;
    std::string *options;
    idx_t n_rows;
    idx_t n_cols;
};

void StatInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<StatState *>(state_p);
    s.x = new std::vector<double>();
    s.y = new std::vector<double>();
    s.labels = new std::vector<std::string>();
    s.options = new std::string();
    s.n_rows = 0;
    s.n_cols = 0;
}

void StatDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<StatState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->x;
        delete states[i]->y;
        delete states[i]->labels;
        delete states[i]->options;
    }
}

// Y: 0 = no target, 1 = DOUBLE, 2 = VARCHAR. Inputs: [y,] x [, options].
template <int Y>
void StatUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    constexpr idx_t X = Y ? 1 : 0;
    UnifiedVectorFormat y_data, x_data, o_data, sdata;
    if (Y) {
        inputs[0].ToUnifiedFormat(count, y_data);
    }
    inputs[X].ToUnifiedFormat(count, x_data);
    const bool has_options = input_count > X + 1;
    if (has_options) {
        inputs[X + 1].ToUnifiedFormat(count, o_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = ListVector::GetEntry(inputs[X]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[X]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);
    auto states = reinterpret_cast<StatState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto xi = x_data.sel->get_index(i);
        auto yi = Y ? y_data.sel->get_index(i) : 0;
        if (!x_data.validity.RowIsValid(xi) || (Y && !y_data.validity.RowIsValid(yi))) {
            continue;
        }
        const auto &entry = lists[xi];
        bool null_element = false;
        for (idx_t j = 0; j < entry.length; j++) {
            null_element = null_element || !child_data.validity.RowIsValid(child_data.sel->get_index(entry.offset + j));
        }
        if (null_element) {
            continue;
        }
        if (s.n_rows == 0) {
            s.n_cols = entry.length;
            if (has_options) {
                auto oi = o_data.sel->get_index(i);
                if (o_data.validity.RowIsValid(oi)) {
                    *s.options = UnifiedVectorFormat::GetData<string_t>(o_data)[oi].GetString();
                }
            }
        } else if (entry.length != s.n_cols) {
            throw InvalidInputException("every feature list must have %llu values", (unsigned long long)s.n_cols);
        }
        for (idx_t j = 0; j < entry.length; j++) {
            s.x->push_back(values[child_data.sel->get_index(entry.offset + j)]);
        }
        if (Y == 1) {
            s.y->push_back(UnifiedVectorFormat::GetData<double>(y_data)[yi]);
        } else if (Y == 2) {
            s.labels->push_back(UnifiedVectorFormat::GetData<string_t>(y_data)[yi].GetString());
        }
        s.n_rows++;
    }
}

void StatCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<StatState *>(source_vector);
    auto targets = FlatVector::GetData<StatState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (src.n_rows == 0) {
            continue;
        }
        if (tgt.n_rows == 0) {
            tgt.n_cols = src.n_cols;
            *tgt.options = *src.options;
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("feature lists of different lengths");
        }
        tgt.x->insert(tgt.x->end(), src.x->begin(), src.x->end());
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
        tgt.labels->insert(tgt.labels->end(), src.labels->begin(), src.labels->end());
        tgt.n_rows += src.n_rows;
    }
}

// Run statistic NAME on a state; returns false for an empty group.
bool RunStat(const char *function, const char *name, StatState &s, AnofoxMlDoubles &out) {
    if (s.n_rows == 0) {
        return false;
    }
    std::vector<const char *> labels;
    for (const auto &l : *s.labels) {
        labels.push_back(l.c_str());
    }
    char *err = anofox_ml_stat(name, s.options->c_str(), s.x->data(), s.n_rows, s.n_cols,
                               s.y->empty() ? nullptr : s.y->data(), labels.empty() ? nullptr : labels.data(), &out);
    if (err) {
        std::string msg(err);
        anofox_ml_free_string(err);
        throw InvalidInputException("%s: %s", function, msg);
    }
    return true;
}

void MutualInfoFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<StatState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        idx_t row = i + offset;
        AnofoxMlDoubles out {nullptr, 0};
        if (!RunStat("ml_mutual_info_agg", "mutual_info", *states[i], out)) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + out.len);
        memcpy(FlatVector::GetData<double>(ListVector::GetEntry(result)) + list_offset, out.data,
               out.len * sizeof(double));
        ListVector::SetListSize(result, list_offset + out.len);
        FlatVector::GetData<list_entry_t>(result)[row] = list_entry_t(list_offset, out.len);
        anofox_ml_free_doubles(out);
    }
}

void CovarianceFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<StatState *>(state_vector);
    auto &fields = StructVector::GetEntries(result);
    auto &cov = *fields[0];
    for (idx_t i = 0; i < count; i++) {
        idx_t row = i + offset;
        AnofoxMlDoubles out {nullptr, 0};
        if (!RunStat("ml_covariance_agg", "covariance", *states[i], out)) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        idx_t d = states[i]->n_cols;
        idx_t outer = ListVector::GetListSize(cov);
        ListVector::Reserve(cov, outer + d);
        auto &rows = ListVector::GetEntry(cov);
        idx_t inner = ListVector::GetListSize(rows);
        ListVector::Reserve(rows, inner + d * d);
        memcpy(FlatVector::GetData<double>(ListVector::GetEntry(rows)) + inner, out.data, d * d * sizeof(double));
        for (idx_t r = 0; r < d; r++) {
            FlatVector::GetData<list_entry_t>(rows)[outer + r] = list_entry_t(inner + r * d, d);
        }
        ListVector::SetListSize(rows, inner + d * d);
        ListVector::SetListSize(cov, outer + d);
        FlatVector::GetData<list_entry_t>(cov)[row] = list_entry_t(outer, d);
        FlatVector::GetData<double>(*fields[1])[row] = out.data[d * d];
        anofox_ml_free_doubles(out);
    }
}

void BandwidthFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<StatState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        idx_t row = i + offset;
        AnofoxMlDoubles out {nullptr, 0};
        if (!RunStat("ml_estimate_bandwidth_agg", "bandwidth", *states[i], out)) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        FlatVector::GetData<double>(result)[row] = out.data[0];
        anofox_ml_free_doubles(out);
    }
}

} // anonymous namespace

void RegisterMlStats(ExtensionLoader &loader) {
    const auto D = LogicalType(LogicalTypeId::DOUBLE), V = LogicalType(LogicalTypeId::VARCHAR);
    const auto L = LogicalType::LIST(D);
    auto make = [](const char *name, vector<LogicalType> args, LogicalType result, aggregate_update_t update,
                   aggregate_finalize_t finalize) {
        AggregateFunction f(name, std::move(args), std::move(result), AggregateFunction::StateSize<StatState>,
                            StatInitialize, update, StatCombine, finalize, FunctionNullHandling::SPECIAL_HANDLING,
                            nullptr, nullptr, StatDestroy);
        return f;
    };
    AggregateFunctionSet mi("ml_mutual_info_agg");
    for (bool options : {false, true}) {
        vector<LogicalType> num {D, L}, lab {V, L};
        if (options) {
            num.push_back(V);
            lab.push_back(V);
        }
        mi.AddFunction(make("ml_mutual_info_agg", num, L, StatUpdate<1>, MutualInfoFinalize));
        mi.AddFunction(make("ml_mutual_info_agg", lab, L, StatUpdate<2>, MutualInfoFinalize));
    }
    loader.RegisterFunction(mi);

    child_list_t<LogicalType> cov_fields;
    cov_fields.emplace_back("covariance", LogicalType::LIST(L));
    cov_fields.emplace_back("shrinkage", D);
    const auto cov_type = LogicalType::STRUCT(std::move(cov_fields));
    AggregateFunctionSet cov("ml_covariance_agg"), bw("ml_estimate_bandwidth_agg");
    for (auto args : {vector<LogicalType> {L}, vector<LogicalType> {L, V}}) {
        cov.AddFunction(make("ml_covariance_agg", args, cov_type, StatUpdate<0>, CovarianceFinalize));
        bw.AddFunction(make("ml_estimate_bandwidth_agg", args, D, StatUpdate<0>, BandwidthFinalize));
    }
    loader.RegisterFunction(cov);
    loader.RegisterFunction(bw);
}

} // namespace duckdb
