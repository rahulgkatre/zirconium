const std = @import("std");

const CACHE_LINE = std.atomic.cache_line;
const SIMD_WIDTH = 128; // std.simd.suggestVectorLength(f32) orelse 8;
const MATRIX_SIZE = 1024;
const BLOCK_SIZE = 512;
const NBLOCKS = MATRIX_SIZE / BLOCK_SIZE;
const NSIMD = BLOCK_SIZE / SIMD_WIDTH;

const SIMD_ALIGN = @alignOf(@Vector(SIMD_WIDTH, f32));

const ALIGN = @max(SIMD_ALIGN, CACHE_LINE);

const Matrix = *align(ALIGN) [MATRIX_SIZE][MATRIX_SIZE]f32;
const InMatrix = *align(ALIGN) const [MATRIX_SIZE][MATRIX_SIZE]f32;

const BlockedMatrix = *align(ALIGN) [NBLOCKS][BLOCK_SIZE][NBLOCKS][BLOCK_SIZE]f32;
const InBlockedMatrix = *align(ALIGN) const [NBLOCKS][BLOCK_SIZE][NBLOCKS][BLOCK_SIZE]f32;

const SimdBlockedMatrix = *align(ALIGN) [NSIMD][SIMD_WIDTH][NSIMD][SIMD_WIDTH]f32;
const InSimdBlockedMatrix = *align(ALIGN) const [NSIMD][SIMD_WIDTH][NSIMD][SIMD_WIDTH]f32;

const Block = *align(ALIGN) [BLOCK_SIZE][BLOCK_SIZE]f32;

const InBlock = *align(ALIGN) const [BLOCK_SIZE][BLOCK_SIZE]f32;

pub fn init(buffer: *[MATRIX_SIZE * MATRIX_SIZE]f32) void {
    var prng = std.rand.DefaultPrng.init(0);
    for (0..MATRIX_SIZE * MATRIX_SIZE) |i| {
        buffer[i] = prng.random().floatNorm(f32);
    }
}

pub fn naive_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    for (0..MATRIX_SIZE) |i| {
        for (0..MATRIX_SIZE) |j| {
            c[i][j] = 0.0;
            for (0..MATRIX_SIZE) |k| {
                c[i][j] = @mulAdd(f32, a[i][k], b[k][j], c[i][j]);
            }
        }
    }
}

pub fn blocked_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    @memset(@as(*[MATRIX_SIZE * MATRIX_SIZE]f32, @ptrCast(c)), 0.0);

    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b: InBlockedMatrix = @ptrCast(b);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        blocked_c[bi][ii][bj][jj] = 0.0;
                        for (0..BLOCK_SIZE) |kk| {
                            blocked_c[bi][ii][bj][jj] = @mulAdd(f32, blocked_a[bi][ii][bk][kk], blocked_b[bk][kk][bj][jj], blocked_c[bi][ii][bj][jj]);
                        }
                    }
                }
            }
        }
    }
}

pub fn transposed_blocked_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    @memset(@as(*[MATRIX_SIZE * MATRIX_SIZE]f32, @ptrCast(c)), 0.0);

    var b_t_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    var b_t: Matrix = @ptrCast(&b_t_raw);
    for (0..MATRIX_SIZE) |i| {
        for (0..MATRIX_SIZE) |j| {
            b_t[i][j] = b[j][i];
        }
    }
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b_t: InBlockedMatrix = @ptrCast(b_t);
    var blocked_c: BlockedMatrix = @ptrCast(c);
    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        for (0..BLOCK_SIZE) |kk| {
                            blocked_c[bi][ii][bj][jj] = @mulAdd(f32, blocked_a[bi][ii][bk][kk], blocked_b_t[bj][jj][bk][kk], blocked_c[bi][ii][bj][jj]);
                        }
                    }
                }
            }
        }
    }
}

pub fn transposed_blocked_simd_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    @memset(@as(*[MATRIX_SIZE * MATRIX_SIZE]f32, @ptrCast(c)), 0.0);

    var b_t_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    var b_t: Matrix = @ptrCast(&b_t_raw);
    for (0..MATRIX_SIZE) |i| {
        for (0..MATRIX_SIZE) |j| {
            b_t[i][j] = b[j][i];
        }
    }
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b_t: InBlockedMatrix = @ptrCast(b_t);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    var simd_a: @Vector(BLOCK_SIZE, f32) = undefined;
    var simd_b: @Vector(BLOCK_SIZE, f32) = undefined;

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        simd_a = blocked_a[bi][ii][bk];
                        simd_b = blocked_b_t[bj][jj][bk];
                        blocked_c[bi][ii][bj][jj] += @reduce(.Add, simd_a * simd_b);
                    }
                }
            }
        }
    }
}

pub fn transposed_blocked_local_simd_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    @memset(@as(*[MATRIX_SIZE * MATRIX_SIZE]f32, @ptrCast(c)), 0.0);

    var b_t_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    var b_t: Matrix = @ptrCast(&b_t_raw);
    for (0..MATRIX_SIZE) |i| {
        for (0..MATRIX_SIZE) |j| {
            b_t[i][j] = b[j][i];
        }
    }
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b_t: InBlockedMatrix = @ptrCast(b_t);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    var local_a: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_b: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_c: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;

    var simd_a: @Vector(BLOCK_SIZE, f32) = undefined;
    var simd_b: @Vector(BLOCK_SIZE, f32) = undefined;

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..BLOCK_SIZE) |ii| {
                for (0..BLOCK_SIZE) |jj| {
                    local_c[ii][jj] = 0.0;
                }
            }
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    local_a[ii] = blocked_a[bi][ii][bk];
                    local_b[ii] = blocked_b_t[bj][ii][bk];
                }
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        simd_a = local_a[ii];
                        simd_b = local_b[jj];
                        local_c[ii][jj] += @reduce(.Add, simd_a * simd_b);
                    }
                }
            }
            for (0..BLOCK_SIZE) |ii| {
                blocked_c[bi][ii][bj] = local_c[ii];
            }
        }
    }
}

pub fn blocked_local_simd_fma_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b: InBlockedMatrix = @ptrCast(b);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    var local_a: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_b: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_c: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;

    var simd_b: @Vector(BLOCK_SIZE, f32) = undefined;
    var simd_c: @Vector(BLOCK_SIZE, f32) = undefined;

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..BLOCK_SIZE) |ii| {
                for (0..BLOCK_SIZE) |jj| {
                    local_c[ii][jj] = 0.0;
                }
            }
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        local_a[ii][jj] = blocked_a[bi][ii][bk][jj];
                        local_b[ii][jj] = blocked_b[bk][ii][bj][jj];
                    }
                }
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |kk| {
                        simd_b = local_b[kk];
                        simd_c = local_c[ii];
                        simd_c = @mulAdd(@Vector(BLOCK_SIZE, f32), @splat(local_a[ii][kk]), simd_b, simd_c);
                        local_c[ii] = simd_c;
                    }
                }
            }
            for (0..BLOCK_SIZE) |ii| {
                blocked_c[bi][ii][bj] = local_c[ii];
            }
        }
    }
}

pub fn blocked_local_blocked_simd_fma_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b: InBlockedMatrix = @ptrCast(b);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    var local_a: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_b: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_c: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;

    const simd_block_a: InSimdBlockedMatrix = @ptrCast(&local_a);
    const simd_block_b: InSimdBlockedMatrix = @ptrCast(&local_b);
    const simd_block_c: SimdBlockedMatrix = @ptrCast(&local_c);

    var local_simd_b: [SIMD_WIDTH]@Vector(SIMD_WIDTH, f32) align(ALIGN) = undefined;
    var local_simd_c: [SIMD_WIDTH]@Vector(SIMD_WIDTH, f32) align(ALIGN) = undefined;

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            @memset(@as(*align(ALIGN) [BLOCK_SIZE * BLOCK_SIZE]f32, @ptrCast(simd_block_c)), 0.0);
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    local_a[ii] = blocked_a[bi][ii][bk];
                    local_b[ii] = blocked_b[bk][ii][bj];
                }
                for (0..NSIMD) |bbi| {
                    for (0..NSIMD) |bbj| {
                        for (0..SIMD_WIDTH) |bii| {
                            local_simd_c[bii] = simd_block_c[bbi][bii][bbj];
                        }
                        for (0..NSIMD) |bbk| {
                            for (0..SIMD_WIDTH) |bkk| {
                                local_simd_b[bkk] = simd_block_b[bbk][bkk][bbj];
                            }
                            for (0..SIMD_WIDTH) |bii| {
                                for (0..SIMD_WIDTH) |bkk| {
                                    local_simd_c[bii] = @mulAdd(@Vector(SIMD_WIDTH, f32), @splat(simd_block_a[bbi][bii][bbk][bkk]), local_simd_b[bkk], local_simd_c[bii]);
                                }
                            }
                        }
                        for (0..SIMD_WIDTH) |bii| {
                            simd_block_c[bbi][bii][bbj] = local_simd_c[bii];
                        }
                    }
                }
            }
            for (0..BLOCK_SIZE) |ii| {
                blocked_c[bi][ii][bj] = local_c[ii];
            }
        }
    }
}

pub fn parallel_blocked_local_blocked_simd_fma_gemm(a: InMatrix, b: InMatrix, c: Matrix) void {
    const blocked_a: InBlockedMatrix = @ptrCast(a);
    const blocked_b: InBlockedMatrix = @ptrCast(b);
    var blocked_c: BlockedMatrix = @ptrCast(c);

    var local_a: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_b: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;
    var local_c: [BLOCK_SIZE][BLOCK_SIZE]f32 align(ALIGN) = undefined;

    const simd_block_a: InSimdBlockedMatrix = @ptrCast(&local_a);
    const simd_block_b: InSimdBlockedMatrix = @ptrCast(&local_b);
    const simd_block_c: SimdBlockedMatrix = @ptrCast(&local_c);

    var local_simd_b: [SIMD_WIDTH]@Vector(SIMD_WIDTH, f32) align(ALIGN) = undefined;
    var local_simd_c: [SIMD_WIDTH]@Vector(SIMD_WIDTH, f32) align(ALIGN) = undefined;

    for (0..NBLOCKS) |bi| {
        for (0..NBLOCKS) |bj| {
            for (0..BLOCK_SIZE) |ii| {
                for (0..BLOCK_SIZE) |jj| {
                    local_c[ii][jj] = 0.0;
                }
            }
            for (0..NBLOCKS) |bk| {
                for (0..BLOCK_SIZE) |ii| {
                    for (0..BLOCK_SIZE) |jj| {
                        local_a[ii][jj] = blocked_a[bi][ii][bk][jj];
                        local_b[ii][jj] = blocked_b[bk][ii][bj][jj];
                    }
                }
                for (0..NSIMD) |bbi| {
                    for (0..NSIMD) |bbj| {
                        for (0..SIMD_WIDTH) |bii| {
                            local_simd_c[bii] = simd_block_c[bbi][bii][bbj];
                        }
                        for (0..NSIMD) |bbk| {
                            for (0..SIMD_WIDTH) |bkk| {
                                local_simd_b[bkk] = simd_block_b[bbk][bkk][bbj];
                            }
                            for (0..SIMD_WIDTH) |bii| {
                                for (0..SIMD_WIDTH) |bkk| {
                                    local_simd_c[bii] = @mulAdd(@Vector(SIMD_WIDTH, f32), @splat(simd_block_a[bbi][bii][bbk][bkk]), local_simd_b[bkk], local_simd_c[bii]);
                                }
                            }
                        }
                        for (0..SIMD_WIDTH) |bii| {
                            simd_block_c[bbi][bii][bbj] = local_simd_c[bii];
                        }
                    }
                }
            }
            for (0..BLOCK_SIZE) |ii| {
                for (0..BLOCK_SIZE) |jj| {
                    blocked_c[bi][ii][bj][jj] = local_c[ii][jj];
                }
            }
        }
    }
}

pub fn main() !void {
    @setFloatMode(.optimized);
    // const NCPU = (std.Thread.getCpuCount() catch unreachable) / 2;
    const NUMBER = 30;

    // const gemm_mp_x_np = @import("gemm2.zig").gemm_mp_x_np;

    var a_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    var b_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    var c_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;
    // var expected_raw: [MATRIX_SIZE * MATRIX_SIZE]f32 align(ALIGN) = undefined;

    const a: Matrix = @ptrCast(&a_raw);
    const b: Matrix = @ptrCast(&b_raw);
    const c: Matrix = @ptrCast(&c_raw);
    // const expected: Matrix = @ptrCast(&expected_raw);

    init(&a_raw);
    init(&b_raw);

    std.debug.print("starting benchmark run...\n", .{});

    var total_nanos: i128 = 0.0;
    for (0..NUMBER) |_| {
        const start = std.time.nanoTimestamp();
        // gemm_mp_x_np(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE)(a, b, c);
        blocked_local_blocked_simd_fma_gemm(a, b, c);
        const end = std.time.nanoTimestamp();
        total_nanos += end - start;
    }

    // std.debug.print("complete, validating result...\n", .{});
    // blocked_local_blocked_simd_fma_gemm(a, b, expected);
    // var total_error: f32 = 0.0;
    // for (0..MATRIX_SIZE * MATRIX_SIZE) |i| {
    // try std.testing.expectApproxEqAbs(expected_raw[i], c_raw[i], 0.001);
    // total_error += @abs(expected_raw[i] - c_raw[i]);
    // }
    // std.debug.print("total error {}\n", .{total_error});
    // std.debug.print("results are valid!\n", .{});

    const total_nanos_float: f64 = @floatFromInt(total_nanos);
    const total_time: f64 = total_nanos_float / 1_000_000_000.0;
    const gflops: f64 = @as(f64, @floatFromInt(NUMBER * 2 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE)) / total_nanos_float;
    std.debug.print("elapsed time was {d} seconds\n", .{total_time});
    std.debug.print("performance estimate is {d} GFLOPS\n", .{gflops});
}
