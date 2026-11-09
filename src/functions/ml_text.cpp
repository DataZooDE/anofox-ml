// Text features: ml_hashing_vectorize(text, n_features [, options]) -> DOUBLE[]
// and ml_hashing_vectorize_sparse(...) -> STRUCT(indices INTEGER[], "values" DOUBLE[])
// (scikit-learn HashingVectorizer; stateless, so no model is needed).

#include <string>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/anofox_ml_ffi.h"
#include "../include/duckdb_compat.hpp"

namespace duckdb {

namespace {

template <bool DENSE>
void HashingFunction(DataChunk &args, ExpressionState &, Vector &result) {
    const char *function = DENSE ? "ml_hashing_vectorize" : "ml_hashing_vectorize_sparse";
    idx_t count = args.size();
    UnifiedVectorFormat text_data, n_data, options_data;
    args.data[0].ToUnifiedFormat(count, text_data);
    args.data[1].ToUnifiedFormat(count, n_data);
    const bool has_options = args.ColumnCount() > 2;
    if (has_options) {
        args.data[2].ToUnifiedFormat(count, options_data);
    }
    auto texts = UnifiedVectorFormat::GetData<string_t>(text_data);
    auto ns = UnifiedVectorFormat::GetData<int64_t>(n_data);
    result.SetVectorType(VectorType::FLAT_VECTOR);

    // Children of the result: dense -> the list child; sparse -> the two
    // list fields of the struct.
    Vector *lists[2] = {&result, nullptr};
    if (!DENSE) {
        lists[0] = &compat::StructField(result, 0);
        lists[1] = &compat::StructField(result, 1);
    }
    for (idx_t i = 0; i < count; i++) {
        auto ti = text_data.sel->get_index(i);
        auto ni = n_data.sel->get_index(i);
        if (!text_data.validity.RowIsValid(ti) || !n_data.validity.RowIsValid(ni)) {
            FlatVector::SetNull(result, i, true);
            continue;
        }
        if (ns[ni] <= 0) {
            throw InvalidInputException("%s: n_features must be positive", function);
        }
        std::string options;
        if (has_options) {
            auto oi = options_data.sel->get_index(i);
            if (options_data.validity.RowIsValid(oi)) {
                options = UnifiedVectorFormat::GetData<string_t>(options_data)[oi].GetString();
            }
        }
        std::string text = texts[ti].GetString();
        AnofoxMlDoubles out {nullptr, 0};
        char *err = anofox_ml_hashing_vectorize(text.c_str(), static_cast<size_t>(ns[ni]), options.c_str(), DENSE, &out);
        if (err) {
            std::string msg(err);
            anofox_ml_free_string(err);
            throw InvalidInputException("%s: %s", function, msg);
        }
        if (DENSE) {
            idx_t offset = ListVector::GetListSize(result);
            ListVector::Reserve(result, offset + out.len);
            memcpy(compat::FlatData<double>(compat::ListChild(result)) + offset, out.data, out.len * sizeof(double));
            ListVector::SetListSize(result, offset + out.len);
            compat::FlatData<list_entry_t>(result)[i] = list_entry_t(offset, out.len);
        } else {
            idx_t nnz = out.len / 2;
            for (idx_t f = 0; f < 2; f++) {
                auto &list = *lists[f];
                idx_t offset = ListVector::GetListSize(list);
                ListVector::Reserve(list, offset + nnz);
                auto &child = compat::ListChild(list);
                for (idx_t j = 0; j < nnz; j++) {
                    if (f == 0) {
                        compat::FlatData<int32_t>(child)[offset + j] = static_cast<int32_t>(out.data[j]);
                    } else {
                        compat::FlatData<double>(child)[offset + j] = out.data[nnz + j];
                    }
                }
                ListVector::SetListSize(list, offset + nnz);
                compat::FlatData<list_entry_t>(list)[i] = list_entry_t(offset, nnz);
            }
        }
        anofox_ml_free_doubles(out);
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

} // anonymous namespace

void RegisterMlText(ExtensionLoader &loader) {
    const auto V = LogicalType(LogicalTypeId::VARCHAR);
    const auto B = LogicalType(LogicalTypeId::BIGINT);
    child_list_t<LogicalType> fields;
    fields.emplace_back("indices", LogicalType::LIST(LogicalType(LogicalTypeId::INTEGER)));
    fields.emplace_back("values", LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)));
    const auto sparse = LogicalType::STRUCT(std::move(fields));
    ScalarFunctionSet dense_set("ml_hashing_vectorize"), sparse_set("ml_hashing_vectorize_sparse");
    for (auto args : {vector<LogicalType> {V, B}, vector<LogicalType> {V, B, V}}) {
        dense_set.AddFunction(Fallible(
            ScalarFunction(args, LogicalType::LIST(LogicalType(LogicalTypeId::DOUBLE)), HashingFunction<true>)));
        sparse_set.AddFunction(Fallible(ScalarFunction(args, sparse, HashingFunction<false>)));
    }
    loader.RegisterFunction(dense_set);
    loader.RegisterFunction(sparse_set);
}

} // namespace duckdb
