// One-shot table macros (ml_fit_predict_by, ml_cluster_by, ...): plain SQL over
// the aggregates, parsed as CREATE MACRO statements at load time.
//
// Column arguments (group_col, id_col, y_col, x_cols) are substituted as
// expressions, so they are referenced unqualified; the macros' own columns
// start with '_' to stay unambiguous. GROUP BY ALL keeps a constant group
// expression from being read as a column position; lists are unnested outside
// the grouping query (DuckDB 2.0's GROUP BY ALL would group on the unnest).

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/create_statement.hpp"

#include "../include/anofox_ml_extension.hpp"
#include "../include/duckdb_compat.hpp"

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
    "_a AS (SELECT group_col AS _g, list(id_col ORDER BY id_col) AS _ids,\n"                                           \
    "              " AGG "(kind, x_cols, params ORDER BY id_col) AS _vs\n"                                             \
    "       FROM _src GROUP BY ALL),\n"                                                                                \
    "_l AS (SELECT _g, unnest(_ids) AS _id, unnest(_vs) AS _v FROM _a)\n"                                              \
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
_a AS (SELECT group_col AS _g, ml_cv_splits_agg(id_col, method, options, y_col ORDER BY id_col) AS _splits
       FROM _src GROUP BY ALL),
_f AS (SELECT _g, unnest(_splits) AS _s FROM _a),
_d AS (SELECT _src.*, _f._g, _f._s.split AS _split, _f._s.role AS _role
       FROM _src JOIN _f ON group_col IS NOT DISTINCT FROM _f._g AND id_col = _f._s.id),
_m AS (SELECT _g, _split, ml_fit_agg(kind, y_col, x_cols, params) AS _model
       FROM _d WHERE _role = 'train' GROUP BY ALL)
SELECT _d.* EXCLUDE (_g, _split, _role), _d._split AS split, ml_predict(_m._model, x_cols) AS prediction
FROM _d JOIN _m ON _d._g IS NOT DISTINCT FROM _m._g AND _d._split = _m._split
WHERE _d._role = 'test'
)";

// Domain-classifier drift (evidently's "model" method, NannyML's domain
// classifier): per chunk, a classifier learns to tell the chunk's rows from
// the reference chunk's; its out-of-fold ROC AUC is about 0.5 without drift.
const char *kDomainClassifierDrift = R"(
WITH _s AS (SELECT chunk_col AS _k, id_col AS _id, x_cols AS _x FROM query_table(source::VARCHAR)),
_c AS (SELECT DISTINCT _k AS _chunk FROM _s WHERE _k IS DISTINCT FROM reference),
_d AS (SELECT c._chunk, s._id, s._x, CASE WHEN s._k IS NOT DISTINCT FROM c._chunk THEN 1.0 ELSE 0.0 END AS _y
       FROM _c c JOIN _s s ON s._k IS NOT DISTINCT FROM c._chunk OR s._k IS NOT DISTINCT FROM reference),
_f AS (SELECT _chunk, unnest(ml_cv_splits_agg(_id, 'stratified_kfold', printf('{"n_splits": %d, "seed": %d}', folds, seed),
                                               _y ORDER BY _id)) AS _sp
       FROM _d GROUP BY _chunk),
_j AS (SELECT _d.*, _f._sp.split AS _split, _f._sp.role AS _role
       FROM _d JOIN _f ON _d._chunk IS NOT DISTINCT FROM _f._chunk AND _d._id = _f._sp.id),
_m AS (SELECT _chunk, _split, ml_fit_agg(kind, _y, _x, params) AS _model FROM _j WHERE _role = 'train' GROUP BY ALL),
_p AS (SELECT _j._chunk, _j._y, ml_predict_proba(_m._model, _j._x)[2] AS _p
       FROM _j JOIN _m ON _j._chunk IS NOT DISTINCT FROM _m._chunk AND _j._split = _m._split WHERE _j._role = 'test')
SELECT _chunk AS chunk, count(*) FILTER (WHERE _y = 0) AS n_reference, count(*) FILTER (WHERE _y = 1) AS n_rows,
       ml_roc_auc(_y, _p) AS auc, threshold::DOUBLE AS threshold, ml_roc_auc(_y, _p) > threshold AS drifted
FROM _p GROUP BY _chunk ORDER BY _chunk
)";

// Direct loss estimation (NannyML's DLE): on labelled reference rows, models
// learn the absolute and the squared error of a regression model from its
// features and prediction; averaged over the rows of an unlabelled chunk they
// estimate its MAE and RMSE. Sampling errors: the spread of the reference
// errors over sqrt(rows) (RMSE by the delta method).
const char *kDle = R"(
WITH _r AS (SELECT y_col AS _y, pred_col AS _p, list_append(x_cols, pred_col::DOUBLE) AS _x FROM query_table(reference::VARCHAR)),
_m AS (SELECT ml_fit_agg(kind, abs(_y - _p), _x, params) AS _mae, ml_fit_agg(kind, (_y - _p) ^ 2, _x, params) AS _mse,
              stddev_pop(abs(_y - _p)) AS _sd_ae, stddev_pop((_y - _p) ^ 2) AS _sd_se
       FROM _r WHERE _y IS NOT NULL AND _p IS NOT NULL),
_a AS (SELECT chunk_col AS _chunk, list_append(x_cols, pred_col::DOUBLE) AS _x FROM query_table(analysis::VARCHAR)),
_e AS (SELECT _chunk, count(*) AS _n, avg(greatest(ml_predict(_mae, _x), 0)) AS _ae, avg(greatest(ml_predict(_mse, _x), 0)) AS _se,
              any_value(_sd_ae) AS _sd_ae, any_value(_sd_se) AS _sd_se
       FROM _a, _m WHERE _x IS NOT NULL GROUP BY _chunk)
SELECT _chunk AS chunk, _n AS n_rows, _ae AS estimated_mae, _sd_ae / sqrt(_n) AS mae_sampling_error,
       sqrt(_se) AS estimated_rmse, _sd_se / sqrt(_n) / (2 * sqrt(_se)) AS rmse_sampling_error
FROM _e ORDER BY _chunk
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
    {"ml_domain_classifier_drift_by",
     {"source", "chunk_col", "id_col", "x_cols", "reference"},
     {{"kind", "'HistGradientBoostingClassifier'"}, {"params", "'{}'"}, {"folds", "5"}, {"threshold", "0.55"}, {"seed", "0"}},
     kDomainClassifierDrift},
    {"ml_dle_estimate_by",
     {"reference", "analysis", "chunk_col", "y_col", "pred_col", "x_cols"},
     {{"kind", "'LgbmRegressor'"}, {"params", "'{}'"}},
     kDle},
};

// Register `CREATE MACRO <name><definition>` as an internal, temporary macro
// of the main schema; the parser handles parameters and defaults.
void RegisterMacro(ExtensionLoader &loader, const string &name, const string &definition) {
    auto parser = compat::MakeParser();
    parser.ParseQuery("CREATE MACRO " + name + definition);
    if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::CREATE_STATEMENT) {
        throw InternalException("%s: expected a single CREATE MACRO", name);
    }
    auto &info = parser.statements[0]->Cast<CreateStatement>().info->Cast<CreateMacroInfo>();
    info.SetSchema(DEFAULT_SCHEMA);
    info.temporary = true;
    info.internal = true;
    loader.RegisterFunction(info);
}

// "(p1, ..., n1 := d1, ...) AS TABLE <sql>"
string TableMacroDefinition(const TableMacro &def) {
    string params;
    for (auto p : def.parameters) {
        params += string(params.empty() ? "" : ", ") + p;
    }
    for (const auto &p : def.named) {
        params += string(params.empty() ? "" : ", ") + p.first + " := " + p.second;
    }
    return "(" + params + ") AS TABLE " + def.sql;
}

// Scalar macros: name, "(parameters) AS expression". Set metrics aggregate
// ml_predict_set's VARCHAR[]; a numeric y matches its label text as a number
// ("1" for 1.0).
const std::pair<const char *, const char *> kScalarMacros[] = {
    {"ml_set_coverage",
     "(y, prediction_set) AS avg(CASE WHEN y IS NULL OR prediction_set IS NULL THEN NULL "
     "WHEN list_contains(prediction_set, y::VARCHAR) OR coalesce(list_contains(list_transform(prediction_set, "
     "lambda _e: TRY_CAST(_e AS DOUBLE)), TRY_CAST(y AS DOUBLE)), false) THEN 1.0 ELSE 0.0 END)"},
    {"ml_mean_set_size", "(prediction_set) AS avg(len(prediction_set))"},
    // Squared Mahalanobis distance to a fitted covariance estimator.
    {"ml_mahalanobis", "(model, x) AS ml_transform(model, x)[1]"},
};

} // anonymous namespace

void RegisterMlMacros(ExtensionLoader &loader) {
    for (const auto &def : kScalarMacros) {
        RegisterMacro(loader, def.first, def.second);
    }
    for (const auto &def : kMacros) {
        RegisterMacro(loader, def.name, TableMacroDefinition(def));
    }
}

} // namespace duckdb
