// Thin adapters over the parts of DuckDB's C++ API that differ between
// DuckDB 1.5 and 2.0, so that one source tree builds against both. Function
// bodies use these instead of the version-specific spellings.
//
// DuckDB 2.0 is detected by its vector headers under duckdb/common/vector/
// (1.5 declares ListVector, StructVector, ... in duckdb/common/types/vector.hpp).

#pragma once

#include <type_traits>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parser.hpp"

#if __has_include("duckdb/common/vector/list_vector.hpp")
#define ANOFOX_DUCKDB_V2 1
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector/unified_vector_format.hpp"
#else
#define ANOFOX_DUCKDB_V2 0
#endif

namespace duckdb {
namespace compat {

//===--------------------------------------------------------------------===//
// Vectors. 2.0 splits read and write access: FlatVector::GetData and
// FlatVector::Validity are const; writes go through the *Mutable accessors.
//===--------------------------------------------------------------------===//
// Writable data of a flat (or constant) result vector.
template <class T>
T *FlatData(Vector &vector) {
#if ANOFOX_DUCKDB_V2
    return FlatVector::GetDataMutable<T>(vector);
#else
    return FlatVector::GetData<T>(vector);
#endif
}

// Writable validity mask of a flat result vector.
inline ValidityMask &FlatValidity(Vector &vector) {
#if ANOFOX_DUCKDB_V2
    return FlatVector::ValidityMutable(vector);
#else
    return FlatVector::Validity(vector);
#endif
}

// The child vector of a LIST vector.
inline const Vector &ListChild(const Vector &vector) {
#if ANOFOX_DUCKDB_V2
    return ListVector::GetChild(vector);
#else
    return ListVector::GetEntry(vector);
#endif
}

inline Vector &ListChild(Vector &vector) {
#if ANOFOX_DUCKDB_V2
    return ListVector::GetChildMutable(vector);
#else
    return ListVector::GetEntry(vector);
#endif
}

// Field `index` of a STRUCT vector (2.0 holds the fields by value).
inline Vector &StructField(Vector &vector, idx_t index) {
#if ANOFOX_DUCKDB_V2
    return StructVector::GetEntries(vector)[index];
#else
    return *StructVector::GetEntries(vector)[index];
#endif
}

//===--------------------------------------------------------------------===//
// Aggregate callbacks
//===--------------------------------------------------------------------===//
// State initialization, written once as `void INIT(STATE &)` and registered
// as StateInit<INIT>. 2.0 initializes states in batches. (MSVC deduces a
// function reference for `auto INIT`, hence the decay.)
template <class F>
struct InitState;
template <class STATE>
struct InitState<void (*)(STATE &)> {
    using type = STATE;
};

#if ANOFOX_DUCKDB_V2
template <auto INIT>
void StateInit(AggregateStateInput &, data_ptr_t *states, idx_t count) {
    using STATE = typename InitState<std::decay_t<decltype(INIT)>>::type;
    for (idx_t i = 0; i < count; i++) {
        INIT(*reinterpret_cast<STATE *>(states[i]));
    }
}
#else
template <auto INIT>
void StateInit(const AggregateFunction &, data_ptr_t state) {
    using STATE = typename InitState<std::decay_t<decltype(INIT)>>::type;
    INIT(*reinterpret_cast<STATE *>(state));
}
#endif

// The input of a finalize callback (2.0 adds a finalize-local state).
#if ANOFOX_DUCKDB_V2
using FinalizeInput = AggregateFinalizeInputData;
#else
using FinalizeInput = AggregateInputData;
#endif

// An aggregate bind, written once as BIND(context, info); `info` is the
// function info attached with SetFunctionInfo, or nullptr.
using aggregate_bind_t = unique_ptr<FunctionData> (*)(ClientContext &context,
                                                      optional_ptr<AggregateFunctionInfo> info);

#if ANOFOX_DUCKDB_V2
template <aggregate_bind_t BIND>
unique_ptr<FunctionData> AggregateBind(BindAggregateFunctionInput &input) {
    auto &function = input.GetBoundFunction();
    return BIND(input.GetClientContext(),
                function.HasExtraFunctionInfo() ? &function.GetExtraFunctionInfo() : nullptr);
}
#else
template <aggregate_bind_t BIND>
unique_ptr<FunctionData> AggregateBind(ClientContext &context, AggregateFunction &function,
                                       vector<unique_ptr<Expression>> &) {
    return BIND(context, function.function_info.get());
}
#endif

inline void SetFunctionInfo(AggregateFunction &function, shared_ptr<AggregateFunctionInfo> info) {
#if ANOFOX_DUCKDB_V2
    function.SetExtraFunctionInfo(std::move(info));
#else
    function.function_info = std::move(info);
#endif
}

//===--------------------------------------------------------------------===//
// Table functions: the column names a bind returns (2.0: identifiers).
//===--------------------------------------------------------------------===//
#if ANOFOX_DUCKDB_V2
using ColumnNames = vector<Identifier>;
#else
using ColumnNames = vector<string>;
#endif

//===--------------------------------------------------------------------===//
// Parsing: a parser with DuckDB's built-in settings, independent of any
// connection (2.0 has no default constructor).
//===--------------------------------------------------------------------===//
inline Parser MakeParser() {
#if ANOFOX_DUCKDB_V2
    return Parser::GetBuiltinParser();
#else
    return Parser();
#endif
}

} // namespace compat
} // namespace duckdb
