// One-shot table macros (ml_fit_predict_by, ml_cluster_by, ...): plain SQL over
// the aggregates, parsed and registered at load time.
//
// Column arguments (group_col, id_col, y_col, x_cols) are substituted as
// expressions, so they are referenced unqualified; the macros' own columns
// start with '_' to stay unambiguous. GROUP BY ALL keeps a constant group
// expression from being read as a column position.

#include "duckdb.hpp"
#include "duckdb/catalog/default/default_functions.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

#include "../include/anofox_ml_extension.hpp"

namespace duckdb {

namespace {

struct TableMacro {
    const char *name;
    vector<const char *> parameters;
    vector<std::pair<const char *, const char *>> named; // name, default
    const char *sql;
};

// Per-row aggregate results joined back to the source rows by (group, id).
#define ML_ROWS_BY(AGG, COLUMN)                                                                                        \
    "WITH _src AS (FROM query_table(source::VARCHAR)),\n"                                                              \
    "_l AS (SELECT group_col AS _g, unnest(list(id_col ORDER BY id_col)) AS _id,\n"                                    \
    "              unnest(" AGG "(kind, x_cols, params ORDER BY id_col)) AS _v\n"                                      \
    "       FROM _src GROUP BY ALL)\n"                                                                                 \
    "SELECT _src.*, _l._v AS " COLUMN "\n"                                                                             \
    "FROM _src LEFT JOIN _l ON group_col IS NOT DISTINCT FROM _l._g AND id_col = _l._id"

const char *kFitPredict = R"(
WITH _src AS (FROM query_table(source::VARCHAR)),
_m AS (SELECT group_col AS _g, ml_fit_agg(kind, y_col, x_cols, params) AS _model
       FROM _src WHERE y_col IS NOT NULL GROUP BY ALL)
SELECT _src.*, ml_predict(_m._model, x_cols) AS prediction
FROM _src LEFT JOIN _m ON group_col IS NOT DISTINCT FROM _m._g
)";

const char *kFitPredictProba = R"(
WITH _src AS (FROM query_table(source::VARCHAR)),
_m AS (SELECT group_col AS _g, ml_fit_agg(kind, y_col, x_cols, params) AS _model
       FROM _src WHERE y_col IS NOT NULL GROUP BY ALL)
SELECT _src.*, ml_predict(_m._model, x_cols) AS prediction, ml_predict_label(_m._model, x_cols) AS label,
       ml_predict_proba(_m._model, x_cols) AS proba, ml_model_info(_m._model).classes AS classes,
       ml_model_info(_m._model).labels AS labels
FROM _src LEFT JOIN _m ON group_col IS NOT DISTINCT FROM _m._g
)";

const char *kCrossValPredict = R"(
WITH _src AS (FROM query_table(source::VARCHAR) WHERE y_col IS NOT NULL),
_f AS (SELECT group_col AS _g, unnest(ml_cv_splits_agg(id_col, method, options, y_col ORDER BY id_col)) AS _s
       FROM _src GROUP BY ALL),
_d AS (SELECT _src.*, _f._g, _f._s.split AS _split, _f._s.role AS _role
       FROM _src JOIN _f ON group_col IS NOT DISTINCT FROM _f._g AND id_col = _f._s.id),
_m AS (SELECT _g, _split, ml_fit_agg(kind, y_col, x_cols, params) AS _model
       FROM _d WHERE _role = 'train' GROUP BY ALL)
SELECT _d.* EXCLUDE (_g, _split, _role), _d._split AS split, ml_predict(_m._model, x_cols) AS prediction
FROM _d JOIN _m ON _d._g IS NOT DISTINCT FROM _m._g AND _d._split = _m._split
WHERE _d._role = 'test'
)";

const TableMacro kMacros[] = {
    {"ml_fit_predict_by", {"source", "group_col", "y_col", "x_cols", "kind"}, {{"params", "'{}'"}}, kFitPredict},
    {"ml_fit_predict_proba_by",
     {"source", "group_col", "y_col", "x_cols", "kind"},
     {{"params", "'{}'"}},
     kFitPredictProba},
    {"ml_cluster_by",
     {"source", "group_col", "id_col", "x_cols", "kind"},
     {{"params", "'{}'"}},
     ML_ROWS_BY("ml_cluster_agg", "cluster")},
    {"ml_outlier_score_by",
     {"source", "group_col", "id_col", "x_cols", "kind"},
     {{"params", "'{}'"}},
     ML_ROWS_BY("ml_outlier_score_agg", "outlier_score")},
    {"ml_fit_transform_by",
     {"source", "group_col", "id_col", "x_cols", "kind"},
     {{"params", "'{}'"}},
     ML_ROWS_BY("ml_fit_transform_agg", "features")},
    {"ml_cross_val_predict_by",
     {"source", "group_col", "id_col", "y_col", "x_cols", "kind"},
     {{"params", "'{}'"}, {"method", "'kfold'"}, {"options", "'{}'"}},
     kCrossValPredict},
};

unique_ptr<CreateMacroInfo> MakeTableMacro(const TableMacro &def) {
    Parser parser;
    parser.ParseQuery(def.sql);
    if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
        throw InternalException("%s: expected a single SELECT", def.name);
    }
    auto function = make_uniq<TableMacroFunction>(std::move(parser.statements[0]->Cast<SelectStatement>().node));
    for (auto p : def.parameters) {
        function->parameters.push_back(make_uniq<ColumnRefExpression>(p));
    }
    for (const auto &p : def.named) {
        function->parameters.push_back(make_uniq<ColumnRefExpression>(p.first));
        auto defaults = Parser::ParseExpressionList(p.second);
        function->default_parameters.insert(make_pair(string(p.first), std::move(defaults[0])));
    }
    auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
    info->schema = DEFAULT_SCHEMA;
    info->name = def.name;
    info->temporary = true;
    info->internal = true;
    info->macros.push_back(std::move(function));
    return info;
}

} // anonymous namespace

// Aggregate metrics of conformal prediction sets (ml_predict_set's VARCHAR[]).
// A numeric y matches its label text as a number ("1" for 1.0).
const DefaultMacro kScalarMacros[] = {
    {DEFAULT_SCHEMA,
     "ml_set_coverage",
     {"y", "prediction_set", nullptr},
     {{nullptr, nullptr}},
     "avg(CASE WHEN y IS NULL OR prediction_set IS NULL THEN NULL "
     "WHEN list_contains(prediction_set, y::VARCHAR) OR coalesce(list_contains(list_transform(prediction_set, "
     "_e -> TRY_CAST(_e AS DOUBLE)), TRY_CAST(y AS DOUBLE)), false) THEN 1.0 ELSE 0.0 END)"},
    {DEFAULT_SCHEMA, "ml_mean_set_size", {"prediction_set", nullptr}, {{nullptr, nullptr}}, "avg(len(prediction_set))"},
};

void RegisterMlMacros(ExtensionLoader &loader) {
    for (const auto &def : kScalarMacros) {
        loader.RegisterFunction(*DefaultFunctionGenerator::CreateInternalMacroInfo(def));
    }
    for (const auto &def : kMacros) {
        loader.RegisterFunction(*MakeTableMacro(def));
    }
}

} // namespace duckdb
