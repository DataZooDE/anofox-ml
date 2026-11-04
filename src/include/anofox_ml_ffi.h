// C ABI of the anofox_ml_ffi Rust crate (crates/anofox-ml-ffi).
// Every function returns NULL on success or an error message that must be
// released with anofox_ml_free_string.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *data;
    size_t len;
} AnofoxMlBytes;

typedef struct {
    double *data;
    size_t len;
} AnofoxMlDoubles;

typedef struct {
    char **data;
    size_t len;
} AnofoxMlStrings;

// Fit `kind` with JSON `params` (may be NULL) on row-major x (n_rows x n_cols);
// y and w hold n_rows doubles or are NULL. The fit uses up to `threads`
// threads of a shared pool (0 or 1 = sequential). `out` receives the BLOB.
char *anofox_ml_fit(const char *kind, const char *params, const double *x, size_t n_rows, size_t n_cols,
                    const double *y, const double *w, size_t threads, AnofoxMlBytes *out);

// anofox_ml_fit for a classifier on string labels y (n_rows strings); the
// labels are stored in the BLOB.
char *anofox_ml_fit_labels(const char *kind, const char *params, const double *x, size_t n_rows, size_t n_cols,
                           const char *const *y, const double *w, size_t threads, AnofoxMlBytes *out);

typedef struct {
    char *kind;
    size_t n_features;
    AnofoxMlDoubles classes; // data is NULL when the model has no classes
    char *capabilities;      // supported methods separated by '\n'
    char *params;            // JSON
    uint32_t format_version;
    char *crate_version;
    AnofoxMlStrings labels; // string class labels (len 0 unless fitted on labels)
} AnofoxMlModelInfo;

// Predict n_rows rows of x; `out` receives one class label per row (the
// number as text for models without string labels).
char *anofox_ml_predict_labels(const uint8_t *model, size_t model_len, const double *x, size_t n_rows, size_t n_cols,
                               AnofoxMlStrings *out);

// Predict n_rows rows of x with a model BLOB; `out` receives n_rows doubles.
char *anofox_ml_predict(const uint8_t *model, size_t model_len, const double *x, size_t n_rows, size_t n_cols,
                        AnofoxMlDoubles *out);

// Apply `method` (predict, predict_proba, decision_function, transform,
// inverse_transform, score_samples) to n_rows rows of x; `out` receives
// n_rows * out_cols doubles, row-major.
char *anofox_ml_apply(const uint8_t *model, size_t model_len, const char *method, const double *x, size_t n_rows,
                      size_t n_cols, AnofoxMlDoubles *out, size_t *out_cols);

// Fit `kind` (with `y`, `w` when non-NULL) and run the first supported of the
// comma-separated `methods` (e.g. "labels,predict") on the training rows; the
// model is not kept.
char *anofox_ml_fit_apply(const char *kind, const char *params, const double *x, size_t n_rows, size_t n_cols,
                          const double *y, const double *w, const char *methods, size_t threads,
                          AnofoxMlDoubles *out, size_t *out_cols);

// Metric `name` (r2, f1, roc_auc, ...) with JSON `options` (may be NULL) on n
// rows: y_true holds n doubles, y_pred n * n_cols (row-major), w n or NULL.
char *anofox_ml_metric(const char *name, const char *options, const double *y_true, const double *y_pred, size_t n,
                       size_t n_cols, const double *w, double *out);

// Cross-validation splits of n rows (in the caller's order): every
// (row, split, test) entry. y (NaN = missing) and groups (UINT64_MAX =
// missing) may be NULL when the method doesn't use them.
typedef struct {
    uint64_t *row;
    uint32_t *split;
    uint8_t *test;
    int32_t *path; // backtest path of a combinatorial-purged test entry, else -1
    size_t len;
} AnofoxMlSplits;
char *anofox_ml_cv_splits(const char *method, const char *options, size_t n, const double *y, const uint64_t *groups,
                          AnofoxMlSplits *out);
// Purged splits (purged_kfold, combinatorial_purged) of n rows in time order
// whose labels span [t_start[i], t_end[i]].
char *anofox_ml_purged_cv_splits(const char *method, const char *options, size_t n, const double *t_start,
                                 const double *t_end, AnofoxMlSplits *out);
// Spatial splits (spatial_block_kfold, kfold, buffered_loo; optional buffer)
// of n rows with n x d row-major coordinates.
char *anofox_ml_spatial_cv_splits(const char *method, const char *options, size_t n, size_t d, const double *coords,
                                  AnofoxMlSplits *out);
void anofox_ml_free_splits(AnofoxMlSplits s);

// anofox_ml_metric on string labels (y_true, y_pred hold n strings each).
char *anofox_ml_metric_labels(const char *name, const char *options, const char *const *y_true,
                              const char *const *y_pred, size_t n, const double *w, double *out);

// Split conformal prediction: calibrate a model BLOB on held-out rows (y as
// doubles, or string labels for a model fitted on labels); `out` receives the
// calibrated BLOB. Intervals: n_rows x 2 doubles (lower, upper). Sets: the
// members of each row's set one row after the other, and each set's size.
char *anofox_ml_conformalize(const uint8_t *model, size_t model_len, const double *x, size_t n_rows, size_t n_cols,
                             const double *y, AnofoxMlBytes *out);
char *anofox_ml_conformalize_labels(const uint8_t *model, size_t model_len, const double *x, size_t n_rows,
                                    size_t n_cols, const char *const *y, AnofoxMlBytes *out);
char *anofox_ml_predict_interval(const uint8_t *model, size_t model_len, const double *x, size_t n_rows,
                                 size_t n_cols, double alpha, AnofoxMlDoubles *out);
char *anofox_ml_predict_set(const uint8_t *model, size_t model_len, const double *x, size_t n_rows, size_t n_cols,
                            double alpha, AnofoxMlStrings *labels, AnofoxMlDoubles *lengths);

// Permutation importance of a model on n_rows rows (y as doubles, or y_labels
// for a model fitted on string labels; the other NULL). `out` receives
// n_cols x 2 doubles: mean and std of the metric drop per feature.
char *anofox_ml_permutation_importance(const uint8_t *model, size_t model_len, const double *x, size_t n_rows,
                                       size_t n_cols, const double *y, const char *const *y_labels,
                                       const char *options, AnofoxMlDoubles *out);

// scikit-learn HashingVectorizer of one document into n_features buckets:
// dense -> n_features values; sparse -> the nnz buckets (0-based) then their values.
char *anofox_ml_hashing_vectorize(const char *text, size_t n_features, const char *options, bool dense,
                                  AnofoxMlDoubles *out);

// Categorical encoders (OneHotEncoder, OrdinalEncoder) on VARCHAR features:
// fit on n_rows x n_cols strings (row-major), encode into n_rows x out_cols.
char *anofox_ml_fit_encoder(const char *kind, const char *params, const char *const *x, size_t n_rows,
                            size_t n_cols, AnofoxMlBytes *out);
char *anofox_ml_encode(const uint8_t *model, size_t model_len, const char *const *x, size_t n_rows, size_t n_cols,
                       AnofoxMlDoubles *out, size_t *out_cols);

// Table statistics of n x d row-major x: "mutual_info" (with y doubles or
// y_labels strings; d values), "covariance" (d * d values + shrinkage),
// "bandwidth" (1 value).
char *anofox_ml_stat(const char *name, const char *options, const double *x, size_t n, size_t d, const double *y,
                     const char *const *y_labels, AnofoxMlDoubles *out);
// Multiple imputation (Hmisc aregImpute) of n_rows x n_cols row-major x
// (NaN = missing): `out` receives n_impute completed copies one after another.
char *anofox_ml_multiple_impute(const char *options, const double *x, size_t n_rows, size_t n_cols,
                                AnofoxMlDoubles *out, size_t *n_impute);
// Hyperparameter search (grid / random) with k-fold CV; y as doubles or
// y_labels as strings (the other NULL). Release with anofox_ml_free_tune_result.
typedef struct {
    char *best_params; // JSON
    double best_score;
    AnofoxMlBytes model; // refitted on all rows
    char *trials;        // JSON array
} AnofoxMlTuneResult;
char *anofox_ml_tune(const char *kind, const char *space, const char *options, const double *x, size_t n_rows,
                     size_t n_cols, const double *y, const char *const *y_labels, size_t threads,
                     AnofoxMlTuneResult *out);
void anofox_ml_free_tune_result(AnofoxMlTuneResult r);

char *anofox_ml_model_info(const uint8_t *model, size_t model_len, AnofoxMlModelInfo *out);

char *anofox_ml_kinds(char **out);                                   // kinds separated by '\n'
char *anofox_ml_default_params(const char *kind, char **out);        // JSON
char *anofox_ml_estimators_json(char **out);                         // [{kind, default_params}]
char *anofox_ml_model_kind(const uint8_t *model, size_t model_len, char **out);
char *anofox_ml_model_fitted(const uint8_t *model, size_t model_len, char **out); // fitted state as JSON
const char *anofox_ml_ffi_version(void);

void anofox_ml_free_string(char *s);
void anofox_ml_free_bytes(AnofoxMlBytes b);
void anofox_ml_free_doubles(AnofoxMlDoubles d);
void anofox_ml_free_strings(AnofoxMlStrings s);
void anofox_ml_free_model_info(AnofoxMlModelInfo info);

#ifdef __cplusplus
}
#endif
