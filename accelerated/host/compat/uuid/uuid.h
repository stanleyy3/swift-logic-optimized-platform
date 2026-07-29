/**
 * compat/uuid/uuid.h - Minimal stand-in for libuuid's header
 *
 * XRT's xclbin.h includes <uuid/uuid.h> solely to typedef its own xuid_t:
 *
 *     typedef uuid_t xuid_t;
 *
 * That header ships in the `uuid-dev` package, which is absent on a stock
 * Kria image (only the runtime libuuid.so.1 is installed). Since uuid_t is a
 * plain 16-byte array with no layout subtleties, this reproduces it exactly
 * and is ABI-identical to the real thing.
 *
 * The Makefile only puts this directory on the include path when the real
 * <uuid/uuid.h> cannot be found, so installing `uuid-dev` transparently takes
 * precedence over this file.
 *
 * The functions below are declared but deliberately not defined. Nothing in
 * this project calls them, and XRT's C API passes xuid_t around as an opaque
 * token. If some XRT header ever does call one, the result is an undefined
 * reference at link time -- a loud, obvious failure rather than a silent
 * misbehavior -- fixed by linking the runtime library that is already present
 * on the board, with -l:libuuid.so.1
 */

#ifndef _UUID_UUID_H
#define _UUID_UUID_H

typedef unsigned char uuid_t[16];

void uuid_clear(uuid_t uu);
int uuid_compare(const uuid_t uu1, const uuid_t uu2);
void uuid_copy(uuid_t dst, const uuid_t src);
void uuid_generate(uuid_t out);
void uuid_generate_random(uuid_t out);
void uuid_generate_time(uuid_t out);
int uuid_is_null(const uuid_t uu);
int uuid_parse(const char *in, uuid_t uu);
void uuid_unparse(const uuid_t uu, char *out);
void uuid_unparse_lower(const uuid_t uu, char *out);
void uuid_unparse_upper(const uuid_t uu, char *out);

#endif