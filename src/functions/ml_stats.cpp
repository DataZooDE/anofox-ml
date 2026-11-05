// Statistic aggregates over feature lists:
//   ml_mutual_info_agg(y DOUBLE | VARCHAR, x DOUBLE[] [, options]) -> DOUBLE[]
//   ml_covariance_agg(x DOUBLE[] [, options]) -> STRUCT(covariance DOUBLE[][], shrinkage DOUBLE)
//   ml_estimate_bandwidth_agg(x DOUBLE[] [, options]) -> DOUBLE
// Rows with a NULL argument or a NULL feature are skipped; an empty group gives NULL.

#include <map>
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

//===--------------------------------------------------------------------===//
// ml_spectral_embedding_agg(src BIGINT, dst BIGINT [, weight DOUBLE [, options VARCHAR]])
//   -> STRUCT(node BIGINT, embedding DOUBLE[])[]
// Spectral embedding of the graph given by the group's edges (undirected).
//===--------------------------------------------------------------------===//
struct EdgeState {
    std::vector<int64_t> *src;
    std::vector<int64_t> *dst;
    std::vector<double> *w;
    std::string *options;
};

void EdgeInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<EdgeState *>(state_p);
    s.src = new std::vector<int64_t>();
    s.dst = new std::vector<int64_t>();
    s.w = new std::vector<double>();
    s.options = new std::string();
}

void EdgeDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<EdgeState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->src;
        delete states[i]->dst;
        delete states[i]->w;
        delete states[i]->options;
    }
}

void EdgeUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat a, b, w, o, sdata;
    inputs[0].ToUnifiedFormat(count, a);
    inputs[1].ToUnifiedFormat(count, b);
    if (input_count > 2) {
        inputs[2].ToUnifiedFormat(count, w);
    }
    if (input_count > 3) {
        inputs[3].ToUnifiedFormat(count, o);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<EdgeState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ai = a.sel->get_index(i), bi = b.sel->get_index(i);
        auto wi = input_count > 2 ? w.sel->get_index(i) : 0;
        if (!a.validity.RowIsValid(ai) || !b.validity.RowIsValid(bi) ||
            (input_count > 2 && !w.validity.RowIsValid(wi))) {
            continue;
        }
        if (input_count > 3 && s.options->empty()) {
            auto oi = o.sel->get_index(i);
            if (o.validity.RowIsValid(oi)) {
                *s.options = UnifiedVectorFormat::GetData<string_t>(o)[oi].GetString();
            }
        }
        s.src->push_back(UnifiedVectorFormat::GetData<int64_t>(a)[ai]);
        s.dst->push_back(UnifiedVectorFormat::GetData<int64_t>(b)[bi]);
        s.w->push_back(input_count > 2 ? UnifiedVectorFormat::GetData<double>(w)[wi] : 1.0);
    }
}

void EdgeCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<EdgeState *>(source_vector);
    auto targets = FlatVector::GetData<EdgeState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (tgt.options->empty()) {
            *tgt.options = *src.options;
        }
        tgt.src->insert(tgt.src->end(), src.src->begin(), src.src->end());
        tgt.dst->insert(tgt.dst->end(), src.dst->begin(), src.dst->end());
        tgt.w->insert(tgt.w->end(), src.w->begin(), src.w->end());
    }
}

void EdgeFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<EdgeState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.src->empty()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        // Node ids -> dense indices in id order.
        std::map<int64_t, uint64_t> index;
        for (auto v : *s.src) {
            index.emplace(v, 0);
        }
        for (auto v : *s.dst) {
            index.emplace(v, 0);
        }
        std::vector<int64_t> nodes;
        for (auto &kv : index) {
            kv.second = nodes.size();
            nodes.push_back(kv.first);
        }
        std::vector<uint64_t> a, b;
        for (idx_t e = 0; e < s.src->size(); e++) {
            a.push_back(index[(*s.src)[e]]);
            b.push_back(index[(*s.dst)[e]]);
        }
        AnofoxMlDoubles out {nullptr, 0};
        size_t cols = 0;
        char *err = anofox_ml_spectral_embedding(s.options->c_str(), nodes.size(), a.data(), b.data(), s.w->data(),
                                                 a.size(), &out, &cols);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("ml_spectral_embedding_agg: %s", msg);
        }
        idx_t n = nodes.size();
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + n);
        auto &fields = StructVector::GetEntries(ListVector::GetEntry(result));
        auto node_out = FlatVector::GetData<int64_t>(*fields[0]);
        auto &emb = *fields[1];
        idx_t e_offset = ListVector::GetListSize(emb);
        ListVector::Reserve(emb, e_offset + n * cols);
        memcpy(FlatVector::GetData<double>(ListVector::GetEntry(emb)) + e_offset, out.data, n * cols * sizeof(double));
        for (idx_t r = 0; r < n; r++) {
            node_out[list_offset + r] = nodes[r];
            FlatVector::GetData<list_entry_t>(emb)[list_offset + r] = list_entry_t(e_offset + r * cols, cols);
        }
        ListVector::SetListSize(emb, e_offset + n * cols);
        anofox_ml_free_doubles(out);
        ListVector::SetListSize(result, list_offset + n);
        entries[row] = list_entry_t(list_offset, n);
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

    const auto B = LogicalType(LogicalTypeId::BIGINT);
    child_list_t<LogicalType> emb_fields;
    emb_fields.emplace_back("node", B);
    emb_fields.emplace_back("embedding", L);
    const auto emb_type = LogicalType::LIST(LogicalType::STRUCT(std::move(emb_fields)));
    AggregateFunctionSet emb("ml_spectral_embedding_agg");
    for (auto args : {vector<LogicalType> {B, B}, vector<LogicalType> {B, B, D}, vector<LogicalType> {B, B, D, V}}) {
        AggregateFunction f("ml_spectral_embedding_agg", args, emb_type, AggregateFunction::StateSize<EdgeState>,
                            EdgeInitialize, EdgeUpdate, EdgeCombine, EdgeFinalize,
                            FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr, EdgeDestroy);
        emb.AddFunction(f);
    }
    loader.RegisterFunction(emb);
    loader.RegisterFunction(bw);
}

} // namespace duckdb
