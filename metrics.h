/**
 * metrics.h - Thread-safe seam between a training run and the UI
 *
 * The training code publishes numbers here; the UI reads snapshots of them.
 * Nothing in this file knows anything about terminals, which is what keeps all
 * presentation confined to `tui/`. `train.c` includes this header and never
 * includes anything from `tui/`.
 *
 * Only scalars and a bounded history live here - never the model matrices - so
 * the UI thread never touches memory that the OpenMP regions are working on.
 *
 * History is stored as two series with very different sample rates:
 *
 * - train accuracy is dense and self-decimating: a fixed number of buckets that
 *   halve in place when full, so a 3,000,000-iteration run costs the same memory
 *   and the same per-frame plotting work as a 6,000-iteration one
 * - test accuracy is sparse and exact: one point per epoch boundary, at most
 *   `num_epochs + 1` of them, each a real measurement
 */

#ifndef _METRICS_H_
#define _METRICS_H_

#include <stdbool.h>

// buckets in the dense train-accuracy history
#define HIST_CAP 1024

// points in the sparse test-accuracy series
// note: one per epoch boundary plus a final one, and epochs are capped at 40
#define TEST_HIST_CAP 64

typedef enum {
    M_IDLE,      // no run has started
    M_RUNNING,   // worker is training
    M_PAUSED,    // worker is parked in `metrics_wait_if_paused`
    M_STOPPING,  // stop requested, worker has not returned yet
    M_DONE       // run finished, either completed or stopped early
} Metrics_state;

// one bucket of the dense train-accuracy series
typedef struct {
    float x;     // fractional epoch at the end of the bucket
    float mean;  // mean accuracy across the bucket
    float min;   // minimum accuracy in the bucket
    float max;   // maximum accuracy in the bucket
} Hist_bucket;

// one exact test-accuracy measurement
typedef struct {
    float x;    // fractional epoch
    float acc;  // test accuracy
} Test_point;

// consistent view of the run's scalars, copied out under the lock
typedef struct {
    int num_epochs;
    int epoch_iters;
    int total_iters;

    int iter;
    int epoch;

    float train_acc;  // most recent batch accuracy
    float test_acc;   // most recent epoch-boundary test accuracy

    double elapsed_s;   // wall time the worker measured, excluding pauses
    double paused_ms;   // total time spent paused so far

    Metrics_state state;
    bool stopped_early;

    unsigned generation;  // bumped on every history push
} TrainSnapshot;

// copy of both history series
typedef struct {
    Hist_bucket train[HIST_CAP + 1];  // +1 for the live partial bucket
    int train_len;

    Test_point test[TEST_HIST_CAP];
    int test_len;

    unsigned generation;
} MetricsHistory;

////////////////////////////////////////////////////////////////////////////////
// WORKER SIDE
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Reports the size of the run, once the dataset has been loaded
 *
 * - Until this is called the run's `total_iters` is 0, which is how the UI knows
 * the dataset is still being read
 *
 * @param[in] num_epochs  Number of epochs in the run
 * @param[in] epoch_iters Iterations per epoch
 * @param[in] total_iters Total iterations in the run
 */
void metrics_set_totals(int num_epochs, int epoch_iters, int total_iters);

/**
 * @brief Starts the run's clock
 *
 * - Call immediately before the first iteration, so that dataset loading is not
 * counted as training time
 *
 * - This is the single origin for elapsed time: the live clock the UI draws and
 * the total reported at the end are the same measurement, so they cannot
 * disagree
 */
void metrics_mark_start(void);

/**
 * @brief Publishes the run's current scalars
 *
 * @param[in] iter       Current iteration
 * @param[in] epoch      Current epoch
 * @param[in] train_acc  Most recent batch accuracy
 */
void metrics_publish(int iter, int epoch, float train_acc);

/**
 * @brief Adds a train-accuracy sample to the dense history
 *
 * - Amortized O(1); the buffer decimates itself instead of growing
 *
 * @param[in] x   Fractional epoch of the sample
 * @param[in] acc Batch accuracy
 */
void metrics_push_train(float x, float acc);

/**
 * @brief Adds an exact test-accuracy measurement to the sparse history
 *
 * - Also updates the `test_acc` scalar
 *
 * @param[in] x   Fractional epoch of the measurement
 * @param[in] acc Test accuracy
 */
void metrics_push_test(float x, float acc);

/**
 * @brief Marks the run as finished and freezes its clock
 *
 * @param[in] stopped_early Whether the run ended because a stop was requested
 * @param[in] test_acc      Final test accuracy
 */
void metrics_finish(bool stopped_early, float test_acc);

/**
 * @brief Blocks while the run is paused
 *
 * - Returns immediately if not paused or if a stop has been requested
 *
 * - Time spent parked here is accumulated so that it can be excluded from the
 * reported training time, which matters because those timings are the baseline
 * the FPGA accelerator is compared against
 */
void metrics_wait_if_paused(void);

/**
 * @brief Whether the UI has asked the run to stop
 *
 * - Checked once per iteration by the training loop
 */
bool metrics_should_stop(void);

////////////////////////////////////////////////////////////////////////////////
// UI SIDE
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Clears all metrics and marks a run as starting
 *
 * - Called by the UI before it launches the worker, so that a stop requested
 * while the dataset is still loading cannot be cleared by the worker
 */
void metrics_reset(void);

/**
 * @brief Copies out a consistent view of the run's scalars
 *
 * - `elapsed_s` is live while the run is going and frozen once it has finished
 *
 * @param[out] out Snapshot to fill
 */
void metrics_snapshot(TrainSnapshot *out);

/**
 * @brief Copies out both history series
 *
 * - The live partial bucket is appended to the train series, so the tip of the
 * curve stays current instead of lagging by a whole bucket
 *
 * @param[out] out History to fill
 */
void metrics_copy_history(MetricsHistory *out);

/**
 * @brief Asks the run to stop at its next iteration boundary
 *
 * - Also wakes a paused worker so that a stop is never swallowed by a pause
 */
void metrics_request_stop(void);

/**
 * @brief Toggles the run between paused and running
 */
void metrics_toggle_pause(void);

#endif
