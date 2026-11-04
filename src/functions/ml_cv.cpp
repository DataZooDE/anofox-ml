// Cross-validation splits as an aggregate:
//   ml_cv_splits_agg(id BIGINT, method VARCHAR, options VARCHAR [, y DOUBLE [, groups BIGINT | VARCHAR]]
//                    ORDER BY ...) -> STRUCT(id BIGINT, split INTEGER, role VARCHAR)[]
// one entry per (row, split) in which the row is used; unnest it to get a
// fold table. Row order (ORDER BY) is the time order of time-series methods.

#include <string>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"

namespace duckdb {

namespace {

constexpr const char *kFunction = "ml_cv_splits_agg";

struct CvState {
    bool initialized;
    std::string *method;
    std::string *options;
    std::vector<int64_t> *ids;
    std::vector<double> *y; // NaN = NULL
    std::vector<std::string> *groups;
    std::vector<uint8_t> *group_valid;
    bool has_y;
    bool has_groups;
};

void CvInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<CvState *>(state_p);
    s.initialized = false;
    s.method = new std::string();
    s.options = new std::string();
    s.ids = new std::vector<int64_t>();
    s.y = new std::vector<double>();
    s.groups = new std::vector<std::string>();
    s.group_valid = new std::vector<uint8_t>();
    s.has_y = false;
    s.has_groups = false;
}

void CvDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<CvState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        delete s.method;
        delete s.options;
        delete s.ids;
        delete s.y;
        delete s.groups;
        delete s.group_valid;
    }
}

template <class GROUP>
std::string GroupKey(const GROUP &g);
template <>
std::string GroupKey<int64_t>(const int64_t &g) {
    return std::to_string(g);
}
template <>
std::string GroupKey<string_t>(const string_t &g) {
    return g.GetString();
}

// Inputs: id, method, options [, y [, groups]].
template <class GROUP>
void CvUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat id_data, method_data, options_data, y_data, group_data, sdata;
    inputs[0].ToUnifiedFormat(count, id_data);
    inputs[1].ToUnifiedFormat(count, method_data);
    inputs[2].ToUnifiedFormat(count, options_data);
    const bool has_y = input_count > 3;
    const bool has_groups = input_count > 4;
    if (has_y) {
        inputs[3].ToUnifiedFormat(count, y_data);
    }
    if (has_groups) {
        inputs[4].ToUnifiedFormat(count, group_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<CvState **>(sdata.data);
    auto ids = UnifiedVectorFormat::GetData<int64_t>(id_data);

    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ii = id_data.sel->get_index(i);
        auto mi = method_data.sel->get_index(i);
        if (!id_data.validity.RowIsValid(ii) || !method_data.validity.RowIsValid(mi)) {
            continue; // rows with a NULL id or method are left out
        }
        if (!s.initialized) {
            s.initialized = true;
            s.has_y = has_y;
            s.has_groups = has_groups;
            *s.method = UnifiedVectorFormat::GetData<string_t>(method_data)[mi].GetString();
            auto oi = options_data.sel->get_index(i);
            if (options_data.validity.RowIsValid(oi)) {
                *s.options = UnifiedVectorFormat::GetData<string_t>(options_data)[oi].GetString();
            }
        }
        s.ids->push_back(ids[ii]);
        if (has_y) {
            auto yi = y_data.sel->get_index(i);
            s.y->push_back(y_data.validity.RowIsValid(yi) ? UnifiedVectorFormat::GetData<double>(y_data)[yi]
                                                          : std::numeric_limits<double>::quiet_NaN());
        }
        if (has_groups) {
            auto gi = group_data.sel->get_index(i);
            bool valid = group_data.validity.RowIsValid(gi);
            s.groups->push_back(valid ? GroupKey<GROUP>(UnifiedVectorFormat::GetData<GROUP>(group_data)[gi]) : "");
            s.group_valid->push_back(valid ? 1 : 0);
        }
    }
}

void CvCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<CvState *>(source_vector);
    auto targets = FlatVector::GetData<CvState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (!src.initialized) {
            continue;
        }
        if (!tgt.initialized) {
            tgt.initialized = true;
            tgt.has_y = src.has_y;
            tgt.has_groups = src.has_groups;
            *tgt.method = *src.method;
            *tgt.options = *src.options;
        }
        tgt.ids->insert(tgt.ids->end(), src.ids->begin(), src.ids->end());
        tgt.y->insert(tgt.y->end(), src.y->begin(), src.y->end());
        tgt.groups->insert(tgt.groups->end(), src.groups->begin(), src.groups->end());
        tgt.group_valid->insert(tgt.group_valid->end(), src.group_valid->begin(), src.group_valid->end());
    }
}

void CvFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<CvState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.ids->empty()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        // Group keys -> dense codes in order of first appearance, so the
        // codes of group_time_series follow the ORDER BY of the aggregate.
        std::vector<uint64_t> codes;
        if (s.has_groups) {
            std::unordered_map<std::string, uint64_t> seen;
            codes.reserve(s.groups->size());
            for (idx_t r = 0; r < s.groups->size(); r++) {
                if (!(*s.group_valid)[r]) {
                    codes.push_back(UINT64_MAX);
                    continue;
                }
                auto it = seen.emplace((*s.groups)[r], seen.size()).first;
                codes.push_back(it->second);
            }
        }
        AnofoxMlSplits splits {nullptr, nullptr, nullptr, nullptr, 0};
        char *err = anofox_ml_cv_splits(s.method->c_str(), s.options->c_str(), s.ids->size(),
                                        s.has_y ? s.y->data() : nullptr, s.has_groups ? codes.data() : nullptr,
                                        &splits);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("%s: %s", kFunction, msg);
        }
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + splits.len);
        auto &child = ListVector::GetEntry(result);
        auto &fields = StructVector::GetEntries(child);
        auto id_out = FlatVector::GetData<int64_t>(*fields[0]);
        auto split_out = FlatVector::GetData<int32_t>(*fields[1]);
        auto role_out = FlatVector::GetData<string_t>(*fields[2]);
        for (idx_t e = 0; e < splits.len; e++) {
            id_out[list_offset + e] = (*s.ids)[splits.row[e]];
            split_out[list_offset + e] = static_cast<int32_t>(splits.split[e]);
            role_out[list_offset + e] = string_t(splits.test[e] ? "test" : "train"); // inlined (< 12 bytes)
        }
        anofox_ml_free_splits(splits);
        ListVector::SetListSize(result, list_offset + splits.len);
        entries[row] = list_entry_t(list_offset, splits.len);
    }
}

//===--------------------------------------------------------------------===//
// ml_purged_cv_splits_agg(id BIGINT, t_start DOUBLE, t_end DOUBLE, method VARCHAR, options VARCHAR
//                         ORDER BY ...) -> STRUCT(id, split, role, path INTEGER)[]
// Purged k-fold and combinatorial purged CV (López de Prado): row i's label
// spans [t_start, t_end]; rows must arrive in time order (ORDER BY).
//===--------------------------------------------------------------------===//
constexpr const char *kPurged = "ml_purged_cv_splits_agg";

struct PurgedState {
    bool initialized;
    std::string *method;
    std::string *options;
    std::vector<int64_t> *ids;
    std::vector<double> *t_start;
    std::vector<double> *t_end;
};

void PurgedInitialize(const AggregateFunction &, data_ptr_t state_p) {
    auto &s = *reinterpret_cast<PurgedState *>(state_p);
    s.initialized = false;
    s.method = new std::string();
    s.options = new std::string();
    s.ids = new std::vector<int64_t>();
    s.t_start = new std::vector<double>();
    s.t_end = new std::vector<double>();
}

void PurgedDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<PurgedState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->method;
        delete states[i]->options;
        delete states[i]->ids;
        delete states[i]->t_start;
        delete states[i]->t_end;
    }
}

void PurgedUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat d[5], sdata;
    for (idx_t a = 0; a < 5; a++) {
        inputs[a].ToUnifiedFormat(count, d[a]);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<PurgedState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        idx_t k[5];
        bool valid = true;
        for (idx_t a = 0; a < 4; a++) {
            k[a] = d[a].sel->get_index(i);
            valid = valid && d[a].validity.RowIsValid(k[a]);
        }
        if (!valid) {
            continue; // rows with a NULL id, time or method are left out
        }
        if (!s.initialized) {
            s.initialized = true;
            *s.method = UnifiedVectorFormat::GetData<string_t>(d[3])[k[3]].GetString();
            k[4] = d[4].sel->get_index(i);
            if (d[4].validity.RowIsValid(k[4])) {
                *s.options = UnifiedVectorFormat::GetData<string_t>(d[4])[k[4]].GetString();
            }
        }
        s.ids->push_back(UnifiedVectorFormat::GetData<int64_t>(d[0])[k[0]]);
        s.t_start->push_back(UnifiedVectorFormat::GetData<double>(d[1])[k[1]]);
        s.t_end->push_back(UnifiedVectorFormat::GetData<double>(d[2])[k[2]]);
    }
}

void PurgedCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<PurgedState *>(source_vector);
    auto targets = FlatVector::GetData<PurgedState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (!src.initialized) {
            continue;
        }
        if (!tgt.initialized) {
            tgt.initialized = true;
            *tgt.method = *src.method;
            *tgt.options = *src.options;
        }
        tgt.ids->insert(tgt.ids->end(), src.ids->begin(), src.ids->end());
        tgt.t_start->insert(tgt.t_start->end(), src.t_start->begin(), src.t_start->end());
        tgt.t_end->insert(tgt.t_end->end(), src.t_end->begin(), src.t_end->end());
    }
}

void PurgedFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<PurgedState *>(state_vector);
    auto entries = FlatVector::GetData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (!s.initialized || s.ids->empty()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        AnofoxMlSplits splits {nullptr, nullptr, nullptr, nullptr, 0};
        char *err = anofox_ml_purged_cv_splits(s.method->c_str(), s.options->c_str(), s.ids->size(),
                                               s.t_start->data(), s.t_end->data(), &splits);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("%s: %s", kPurged, msg);
        }
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + splits.len);
        auto &child = ListVector::GetEntry(result);
        auto &fields = StructVector::GetEntries(child);
        auto id_out = FlatVector::GetData<int64_t>(*fields[0]);
        auto split_out = FlatVector::GetData<int32_t>(*fields[1]);
        auto role_out = FlatVector::GetData<string_t>(*fields[2]);
        auto path_out = FlatVector::GetData<int32_t>(*fields[3]);
        auto &path_validity = FlatVector::Validity(*fields[3]);
        for (idx_t e = 0; e < splits.len; e++) {
            id_out[list_offset + e] = (*s.ids)[splits.row[e]];
            split_out[list_offset + e] = static_cast<int32_t>(splits.split[e]);
            role_out[list_offset + e] = string_t(splits.test[e] ? "test" : "train");
            if (splits.path[e] < 0) {
                path_validity.SetInvalid(list_offset + e);
            } else {
                path_out[list_offset + e] = splits.path[e];
            }
        }
        anofox_ml_free_splits(splits);
        ListVector::SetListSize(result, list_offset + splits.len);
        entries[row] = list_entry_t(list_offset, splits.len);
    }
}

} // anonymous namespace

void RegisterMlCv(ExtensionLoader &loader) {
    child_list_t<LogicalType> fields;
    fields.emplace_back("id", LogicalType::BIGINT);
    fields.emplace_back("split", LogicalType::INTEGER);
    fields.emplace_back("role", LogicalType::VARCHAR);
    const auto result = LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
    const auto B = LogicalType::BIGINT, V = LogicalType::VARCHAR, D = LogicalType::DOUBLE;

    AggregateFunctionSet set(kFunction);
    auto add = [&](vector<LogicalType> args, aggregate_update_t update) {
        AggregateFunction f(kFunction, std::move(args), result, AggregateFunction::StateSize<CvState>, CvInitialize,
                            update, CvCombine, CvFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr,
                            CvDestroy);
        set.AddFunction(f);
    };
    add({B, V, V}, CvUpdate<int64_t>);
    add({B, V, V, D}, CvUpdate<int64_t>);
    add({B, V, V, D, B}, CvUpdate<int64_t>);
    add({B, V, V, D, V}, CvUpdate<string_t>);
    loader.RegisterFunction(set);

    child_list_t<LogicalType> purged_fields;
    purged_fields.emplace_back("id", LogicalType(LogicalTypeId::BIGINT));
    purged_fields.emplace_back("split", LogicalType(LogicalTypeId::INTEGER));
    purged_fields.emplace_back("role", LogicalType(LogicalTypeId::VARCHAR));
    purged_fields.emplace_back("path", LogicalType(LogicalTypeId::INTEGER));
    AggregateFunction purged(kPurged, {B, D, D, V, V}, LogicalType::LIST(LogicalType::STRUCT(std::move(purged_fields))),
                             AggregateFunction::StateSize<PurgedState>, PurgedInitialize, PurgedUpdate, PurgedCombine,
                             PurgedFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr, PurgedDestroy);
    loader.RegisterFunction(purged);
}

} // namespace duckdb
