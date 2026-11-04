// C ABI of the anofox_ml_ffi Rust crate (crates/anofox-ml-ffi).
// Every function returns NULL on success or an error message that must be
// released with anofox_ml_free_string.
#pragma once

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

// Fit `kind` with JSON `params` (may be NULL) on row-major x (n_rows x n_cols);
// y and w hold n_rows doubles or are NULL. `out` receives the model BLOB.
char *anofox_ml_fit(const char *kind, const char *params, const double *x, size_t n_rows, size_t n_cols,
                    const double *y, const double *w, AnofoxMlBytes *out);

typedef struct {
    char *kind;
    size_t n_features;
    AnofoxMlDoubles classes; // data is NULL when the model has no classes
    char *capabilities;      // supported methods separated by '\n'
    char *params;            // JSON
    uint32_t format_version;
    char *crate_version;
} AnofoxMlModelInfo;

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
                          const double *y, const double *w, const char *methods, AnofoxMlDoubles *out,
                          size_t *out_cols);

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
    size_t len;
} AnofoxMlSplits;
char *anofox_ml_cv_splits(const char *method, const char *options, size_t n, const double *y, const uint64_t *groups,
                          AnofoxMlSplits *out);
void anofox_ml_free_splits(AnofoxMlSplits s);

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
void anofox_ml_free_model_info(AnofoxMlModelInfo info);

#ifdef __cplusplus
}
#endif
