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

// Predict n_rows rows of x with a model BLOB; `out` receives n_rows doubles.
char *anofox_ml_predict(const uint8_t *model, size_t model_len, const double *x, size_t n_rows, size_t n_cols,
                        AnofoxMlDoubles *out);

char *anofox_ml_kinds(char **out);                                   // kinds separated by '\n'
char *anofox_ml_default_params(const char *kind, char **out);        // JSON
char *anofox_ml_estimators_json(char **out);                         // [{kind, default_params}]
char *anofox_ml_model_kind(const uint8_t *model, size_t model_len, char **out);
const char *anofox_ml_ffi_version(void);

void anofox_ml_free_string(char *s);
void anofox_ml_free_bytes(AnofoxMlBytes b);
void anofox_ml_free_doubles(AnofoxMlDoubles d);

#ifdef __cplusplus
}
#endif
