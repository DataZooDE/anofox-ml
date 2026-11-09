// ml_multiple_impute_agg(id BIGINT, x DOUBLE[] [, options VARCHAR] ORDER BY id)
//   -> STRUCT(imputation INTEGER, id BIGINT, x DOUBLE[])[]
// Multiple imputation (Hmisc aregImpute): a NULL element of x is a missing
// value; the result holds n_impute completed copies of the group's rows.

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"
#include "../include/duckdb_compat.hpp"

namespace duckdb {

namespace {

constexpr const char *kImpute = "ml_multiple_impute_agg";

struct ImputeState {
    std::vector<int64_t> *ids;
    std::vector<double> *x; // NaN = missing
    std::string *options;
    idx_t n_cols;
};

void ImputeInitialize(ImputeState &s) {
    s.ids = new std::vector<int64_t>();
    s.x = new std::vector<double>();
    s.options = new std::string();
    s.n_cols = 0;
}

void ImputeDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    auto states = FlatVector::GetData<ImputeState *>(state_vector);
    for (idx_t i = 0; i < count; i++) {
        delete states[i]->ids;
        delete states[i]->x;
        delete states[i]->options;
    }
}

void ImputeUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat id_data, x_data, o_data, sdata;
    inputs[0].ToUnifiedFormat(count, id_data);
    inputs[1].ToUnifiedFormat(count, x_data);
    if (input_count > 2) {
        inputs[2].ToUnifiedFormat(count, o_data);
    }
    state_vector.ToUnifiedFormat(count, sdata);
    auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &child = compat::ListChild(inputs[1]);
    UnifiedVectorFormat child_data;
    child.ToUnifiedFormat(ListVector::GetListSize(inputs[1]), child_data);
    auto values = UnifiedVectorFormat::GetData<double>(child_data);
    auto states = UnifiedVectorFormat::GetData<ImputeState *>(sdata);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        auto ii = id_data.sel->get_index(i);
        auto xi = x_data.sel->get_index(i);
        if (!id_data.validity.RowIsValid(ii) || !x_data.validity.RowIsValid(xi)) {
            continue; // rows with a NULL id or feature list are left out
        }
        const auto &entry = lists[xi];
        if (s.ids->empty()) {
            s.n_cols = entry.length;
            if (input_count > 2) {
                auto oi = o_data.sel->get_index(i);
                if (o_data.validity.RowIsValid(oi)) {
                    *s.options = UnifiedVectorFormat::GetData<string_t>(o_data)[oi].GetString();
                }
            }
        } else if (entry.length != s.n_cols) {
            throw InvalidInputException("%s: every feature list must have %llu values", kImpute,
                                        (unsigned long long)s.n_cols);
        }
        s.ids->push_back(UnifiedVectorFormat::GetData<int64_t>(id_data)[ii]);
        for (idx_t j = 0; j < entry.length; j++) {
            auto vi = child_data.sel->get_index(entry.offset + j);
            s.x->push_back(child_data.validity.RowIsValid(vi) ? values[vi] : std::numeric_limits<double>::quiet_NaN());
        }
    }
}

void ImputeCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    auto sources = FlatVector::GetData<ImputeState *>(source_vector);
    auto targets = FlatVector::GetData<ImputeState *>(target_vector);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[i];
        auto &tgt = *targets[i];
        if (src.ids->empty()) {
            continue;
        }
        if (tgt.ids->empty()) {
            tgt.n_cols = src.n_cols;
            *tgt.options = *src.options;
        } else if (tgt.n_cols != src.n_cols) {
            throw InvalidInputException("%s: feature lists of different lengths", kImpute);
        }
        tgt.ids->insert(tgt.ids->end(), src.ids->begin(), src.ids->end());
        tgt.x->insert(tgt.x->end(), src.x->begin(), src.x->end());
    }
}

void ImputeFinalize(Vector &state_vector, compat::FinalizeInput &, Vector &result, idx_t count, idx_t offset) {
    auto states = FlatVector::GetData<ImputeState *>(state_vector);
    auto entries = compat::FlatData<list_entry_t>(result);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[i];
        idx_t row = i + offset;
        if (s.ids->empty()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }
        AnofoxMlDoubles out {nullptr, 0};
        size_t m = 0;
        char *err = anofox_ml_multiple_impute(s.options->c_str(), s.x->data(), s.ids->size(), s.n_cols, &out, &m);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("%s: %s", kImpute, msg);
        }
        idx_t n = s.ids->size(), d = s.n_cols, total = m * n;
        idx_t list_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, list_offset + total);
        auto &structs = compat::ListChild(result);
        auto imp = compat::FlatData<int32_t>(compat::StructField(structs, 0));
        auto ids = compat::FlatData<int64_t>(compat::StructField(structs, 1));
        auto &xs = compat::StructField(structs, 2);
        idx_t x_offset = ListVector::GetListSize(xs);
        ListVector::Reserve(xs, x_offset + total * d);
        memcpy(compat::FlatData<double>(compat::ListChild(xs)) + x_offset, out.data, total * d * sizeof(double));
        for (idx_t e = 0; e < total; e++) {
            imp[list_offset + e] = static_cast<int32_t>(e / n + 1);
            ids[list_offset + e] = (*s.ids)[e % n];
            compat::FlatData<list_entry_t>(xs)[list_offset + e] = list_entry_t(x_offset + e * d, d);
        }
        ListVector::SetListSize(xs, x_offset + total * d);
        anofox_ml_free_doubles(out);
        ListVector::SetListSize(result, list_offset + total);
        entries[row] = list_entry_t(list_offset, total);
    }
}

} // anonymous namespace

void RegisterMlImpute(ExtensionLoader &loader) {
    const auto B = LogicalType(LogicalTypeId::BIGINT), V = LogicalType(LogicalTypeId::VARCHAR);
    const auto L = LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE));
    child_list_t<LogicalType> fields;
    fields.emplace_back("imputation", LogicalType(LogicalTypeId::INTEGER));
    fields.emplace_back("id", B);
    fields.emplace_back("x", L);
    const auto result = LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
    AggregateFunctionSet set(kImpute);
    for (auto args : {vector<LogicalType> {B, L}, vector<LogicalType> {B, L, V}}) {
        set.AddFunction(AggregateFunction(kImpute, args, result, AggregateFunction::StateSize<ImputeState>,
                                          compat::StateInit<ImputeInitialize>, ImputeUpdate, ImputeCombine,
                                          ImputeFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, nullptr,
                                          ImputeDestroy));
    }
    loader.RegisterFunction(set);
}

} // namespace duckdb
