/**
 * metrics.c - Thread-safe seam between a training run and the UI
 */

#include "metrics.h"

#include <pthread.h>
#include <string.h>
#include <time.h>

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

// run scalars
static int num_epochs = 0;
static int epoch_iters = 0;
static int total_iters = 0;
static int cur_iter = 0;
static int cur_epoch = 0;
static float train_acc = 0.f;
static float test_acc = 0.f;

// the run's clock: one origin for both the live display and the final total
static double start_ms = 0.0;          // 0 until the first iteration
static double paused_ms = 0.0;         // time parked in completed pauses, excluded
static double pause_start_ms = 0.0;    // when the worker parked, 0 if it is running
static double final_elapsed_s = -1.0;  // negative while the run is still going

// lifecycle
static Metrics_state state = M_IDLE;
static bool stop_req = false;
static bool stopped_early = false;
static unsigned generation = 0;

// dense train-accuracy history
static Hist_bucket train_hist[HIST_CAP];
static int train_len = 0;
static int merge_factor = 1;  // raw samples that go into one committed bucket

// in-flight partial bucket, not yet committed to `train_hist`
static float pend_sum = 0.f;
static float pend_min = 0.f;
static float pend_max = 0.f;
static float pend_x = 0.f;
static int pend_n = 0;

// sparse test-accuracy history
static Test_point test_hist[TEST_HIST_CAP];
static int test_len = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/**
 * @brief Keeps a bucket's mean inside its own min and max
 *
 * - Averaging in float can land a rounding unit outside the range of the values
 * averaged, and a bucket whose mean sits outside its own bounds would be a trap
 * for anything that later draws a spread band from them
 */
static float clamp_to_bucket(float mean, float lo, float hi) {
    if (mean < lo) return lo;
    if (mean > hi) return hi;

    return mean;
}

/**
 * @brief Elapsed training time, excluding pauses
 *
 * - Live while the run is going, frozen at the value it had when the run ended
 *
 * - Caller must hold the lock
 */
static double elapsed_locked(void) {
    if (final_elapsed_s >= 0.0) return final_elapsed_s;
    if (start_ms == 0.0) return 0.0;

    double parked = paused_ms;

    // a pause is only credited to `paused_ms` once the worker wakes up, so the
    // one currently in progress has to be counted here or the clock would keep
    // running while the run is stopped
    if (pause_start_ms != 0.0) parked += now_ms() - pause_start_ms;

    return (now_ms() - start_ms - parked) / 1e3;
}

////////////////////////////////////////////////////////////////////////////////
// WORKER SIDE
////////////////////////////////////////////////////////////////////////////////

void metrics_set_totals(int epochs, int iters_per_epoch, int iters_total) {
    pthread_mutex_lock(&mtx);

    num_epochs = epochs;
    epoch_iters = iters_per_epoch;
    total_iters = iters_total;

    pthread_mutex_unlock(&mtx);
}

void metrics_mark_start(void) {
    pthread_mutex_lock(&mtx);

    start_ms = now_ms();
    paused_ms = 0.0;
    pause_start_ms = 0.0;
    final_elapsed_s = -1.0;

    pthread_mutex_unlock(&mtx);
}

void metrics_publish(int iter, int epoch, float acc) {
    pthread_mutex_lock(&mtx);

    cur_iter = iter;
    cur_epoch = epoch;
    train_acc = acc;

    pthread_mutex_unlock(&mtx);
}

/**
 * @brief Halves the dense history in place, doubling the samples per bucket
 *
 * - Every committed bucket holds exactly `merge_factor` raw samples, so merging
 * a pair by plain averaging is exact rather than an approximation
 */
static void halve_train_hist(void) {
    int merged_len = HIST_CAP / 2;

    for (int i = 0; i < merged_len; i++) {
        Hist_bucket *a = &train_hist[2 * i];
        Hist_bucket *b = &train_hist[2 * i + 1];

        // the bucket's x is its right edge, so the merged bucket takes b's
        train_hist[i].x = b->x;
        train_hist[i].min = (a->min < b->min) ? a->min : b->min;
        train_hist[i].max = (a->max > b->max) ? a->max : b->max;
        train_hist[i].mean = clamp_to_bucket(0.5f * (a->mean + b->mean),
                                             train_hist[i].min, train_hist[i].max);
    }

    train_len = merged_len;
    merge_factor *= 2;
}

void metrics_push_train(float x, float acc) {
    pthread_mutex_lock(&mtx);

    // accumulate into the partial bucket
    if (pend_n == 0) {
        pend_sum = acc;
        pend_min = acc;
        pend_max = acc;
    } else {
        pend_sum += acc;
        if (acc < pend_min) pend_min = acc;
        if (acc > pend_max) pend_max = acc;
    }
    pend_x = x;
    pend_n++;

    // commit the bucket once it holds a full set of samples
    if (pend_n >= merge_factor) {
        train_hist[train_len].x = pend_x;
        train_hist[train_len].min = pend_min;
        train_hist[train_len].max = pend_max;
        train_hist[train_len].mean = clamp_to_bucket(pend_sum / (float)pend_n,
                                                     pend_min, pend_max);
        train_len++;

        pend_n = 0;

        if (train_len >= HIST_CAP) halve_train_hist();
    }

    generation++;

    pthread_mutex_unlock(&mtx);
}

void metrics_push_test(float x, float acc) {
    pthread_mutex_lock(&mtx);

    if (test_len < TEST_HIST_CAP) {
        test_hist[test_len].x = x;
        test_hist[test_len].acc = acc;
        test_len++;
    } else {
        // cannot happen while epochs are capped at 50, but never write past the end
        test_hist[TEST_HIST_CAP - 1].x = x;
        test_hist[TEST_HIST_CAP - 1].acc = acc;
    }

    test_acc = acc;
    generation++;

    pthread_mutex_unlock(&mtx);
}

void metrics_finish(bool early, float final_test_acc) {
    pthread_mutex_lock(&mtx);

    // freeze the clock before switching state, so the total cannot drift
    final_elapsed_s = elapsed_locked();

    state = M_DONE;
    stopped_early = early;
    test_acc = final_test_acc;
    generation++;

    pthread_mutex_unlock(&mtx);
}

void metrics_wait_if_paused(void) {
    pthread_mutex_lock(&mtx);

    if (state == M_PAUSED && !stop_req) {
        // note: the clock stops from the moment the worker actually parks, not
        //       from the moment the key was pressed - the iteration still running
        //       at that point is real training time
        pause_start_ms = now_ms();

        while (state == M_PAUSED && !stop_req) {
            pthread_cond_wait(&cv, &mtx);
        }

        paused_ms += now_ms() - pause_start_ms;
        pause_start_ms = 0.0;
    }

    pthread_mutex_unlock(&mtx);
}

bool metrics_should_stop(void) {
    pthread_mutex_lock(&mtx);
    bool req = stop_req;
    pthread_mutex_unlock(&mtx);

    return req;
}

////////////////////////////////////////////////////////////////////////////////
// UI SIDE
////////////////////////////////////////////////////////////////////////////////

void metrics_reset(void) {
    pthread_mutex_lock(&mtx);

    num_epochs = 0;
    epoch_iters = 0;
    total_iters = 0;

    cur_iter = 0;
    cur_epoch = 0;
    train_acc = 0.f;
    test_acc = 0.f;

    start_ms = 0.0;
    paused_ms = 0.0;
    pause_start_ms = 0.0;
    final_elapsed_s = -1.0;

    state = M_RUNNING;
    stop_req = false;
    stopped_early = false;

    train_len = 0;
    merge_factor = 1;
    pend_n = 0;
    test_len = 0;

    generation++;

    pthread_mutex_unlock(&mtx);
}

void metrics_snapshot(TrainSnapshot *out) {
    pthread_mutex_lock(&mtx);

    out->num_epochs = num_epochs;
    out->epoch_iters = epoch_iters;
    out->total_iters = total_iters;
    out->iter = cur_iter;
    out->epoch = cur_epoch;
    out->train_acc = train_acc;
    out->test_acc = test_acc;
    out->elapsed_s = elapsed_locked();
    out->paused_ms = paused_ms;
    out->state = state;
    out->stopped_early = stopped_early;
    out->generation = generation;

    pthread_mutex_unlock(&mtx);
}

void metrics_copy_history(MetricsHistory *out) {
    pthread_mutex_lock(&mtx);

    memcpy(out->train, train_hist, (size_t)train_len * sizeof(Hist_bucket));
    out->train_len = train_len;

    // append the live partial bucket so the tip of the curve is current
    if (pend_n > 0) {
        out->train[out->train_len].x = pend_x;
        out->train[out->train_len].min = pend_min;
        out->train[out->train_len].max = pend_max;
        out->train[out->train_len].mean = clamp_to_bucket(pend_sum / (float)pend_n,
                                                          pend_min, pend_max);
        out->train_len++;
    }

    memcpy(out->test, test_hist, (size_t)test_len * sizeof(Test_point));
    out->test_len = test_len;

    out->generation = generation;

    pthread_mutex_unlock(&mtx);
}

void metrics_request_stop(void) {
    pthread_mutex_lock(&mtx);

    stop_req = true;
    if (state == M_RUNNING || state == M_PAUSED) state = M_STOPPING;

    // wake the worker if it is parked in a pause, so a stop is never swallowed
    pthread_cond_broadcast(&cv);

    pthread_mutex_unlock(&mtx);
}

void metrics_toggle_pause(void) {
    pthread_mutex_lock(&mtx);

    if (state == M_RUNNING) {
        state = M_PAUSED;
    } else if (state == M_PAUSED) {
        state = M_RUNNING;
        pthread_cond_broadcast(&cv);
    }

    pthread_mutex_unlock(&mtx);
}
