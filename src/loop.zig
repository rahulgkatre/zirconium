const std = @import("std");
const utils = @import("utils.zig");
const func = @import("func.zig");

const AllocatedBuffer = @import("buffer.zig").AllocatedBuffer;
const IterationSpace = @import("iterspace.zig").IterationSpace;

pub fn Nest(comptime In: type, comptime InOut: type, comptime idx_ndims: u8) type {
    return struct {
        const CurrentNest = @This();
        const IterLogicWrapper = struct {
            eval: fn (anytype, anytype, [idx_ndims]usize) callconv(.Inline) void,
        };

        // common interface here is that both have an eval() function
        const LoopBodyItem = union(enum) { loop: Loop, iter_logic: IterLogicWrapper };

        pub const Loop = struct {
            lower: usize = 0,
            upper: usize,

            step_size: usize = 1,
            unrolled: bool = false,
            vector: bool = false,
            block_info: utils.BlockInfo,
            cpu_parallel: bool = false,

            body: []const LoopBodyItem = &.{},

            inline fn eval(
                comptime loop: *const Loop,
                in: anytype,
                inout: anytype,
                base_idx: [idx_ndims]usize,
            ) void {
                var idx = base_idx;
                const base_for_dim = base_idx[loop.block_info.orig_dim];
                var loopvar = loop.lower;
                const n_steps = comptime @divFloor(loop.upper - loop.lower, loop.step_size);
                var threads: [n_steps]std.Thread = undefined;

                inline for (loop.body) |item| {
                    switch (item) {
                        inline else => |inner| {
                            if (comptime loop.unrolled) {
                                comptime std.debug.assert(@mod(loop.upper - loop.lower, loop.step_size) == 0);
                                inline for (0..n_steps) |i| {
                                    idx[loop.block_info.orig_dim] += loop.block_info.block_size * loopvar;
                                    if (comptime loop.cpu_parallel) {
                                        threads[i] = std.Thread.spawn(.{}, inner.eval, .{ in, inout, idx }) catch unreachable;
                                    } else {
                                        inner.eval(in, inout, idx);
                                    }
                                    idx[loop.block_info.orig_dim] = base_for_dim;
                                    loopvar += loop.step_size;
                                }
                            } else {
                                for (0..@divFloor(loop.upper - loop.lower, loop.step_size)) |i| {
                                    idx[loop.block_info.orig_dim] += loop.block_info.block_size * loopvar;
                                    if (comptime loop.cpu_parallel) {
                                        threads[i] = std.Thread.spawn(.{}, inner.eval, .{ in, inout, idx }) catch unreachable;
                                    } else {
                                        inner.eval(in, inout, idx);
                                    }
                                    idx[loop.block_info.orig_dim] = base_for_dim;
                                    loopvar += loop.step_size;
                                }
                            }
                            if (comptime loop.cpu_parallel) {
                                inline for (threads) |t| {
                                    t.join();
                                }
                            }
                        },
                    }
                }
            }
        };

        body: []const Loop,
        idx_ndims: u8,

        fn buildLoop(
            comptime iter_space: anytype,
            comptime dim: u8,
            body: []const LoopBodyItem,
        ) Loop {
            const IterSpace: type = @TypeOf(iter_space.*);
            const vector = iter_space.vector and dim == IterSpace.ndims - 1;

            return .{
                .lower = 0,
                .upper = IterSpace.shape[dim],
                .body = body,
                .block_info = iter_space.block_info[dim],
                .vector = vector,
                .unrolled = iter_space.unrolled_dims[dim],
                .step_size = if (vector) IterSpace.shape[IterSpace.ndims - 1] else 1,
                .cpu_parallel = iter_space.parallel_dims[dim],
            };
        }

        pub fn init(
            comptime iter_space: anytype,
            comptime iter_logic: func.IterationLogic(In, InOut, idx_ndims),
        ) @This() {
            const IterSpace: type = @TypeOf(iter_space.*);
            if (IterSpace.ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }

            const iter_logic_wrapper: IterLogicWrapper = .{
                .eval = struct {
                    inline fn wrapped_iter_logic(in: anytype, inout: anytype, idx: [idx_ndims]usize) void {
                        if (comptime iter_space.vector) {
                            const vec_len = IterSpace.shape[IterSpace.ndims - 1];
                            const VecLogic: type = func.VectorizedLogic(In, InOut, idx_ndims, vec_len);
                            const vectorized_logic = comptime @as(*const VecLogic, @ptrCast(&iter_logic));
                            const vectorized_logic_args = @as(*const std.meta.ArgsTuple(VecLogic), @ptrCast(&(in ++ inout ++ .{idx}))).*;
                            @call(.always_inline, vectorized_logic, vectorized_logic_args);
                        } else {
                            @call(.always_inline, iter_logic, in ++ inout ++ .{idx});
                        }
                    }
                }.wrapped_iter_logic,
            };

            const body: []const Loop = comptime blk: {
                var body: []const LoopBodyItem = &.{LoopBodyItem{ .iter_logic = iter_logic_wrapper }};
                for (0..IterSpace.ndims) |dim| {
                    body = &.{
                        LoopBodyItem{
                            .loop = buildLoop(
                                iter_space,
                                IterSpace.ndims - dim - 1,
                                body,
                            ),
                        },
                    };
                }
                break :blk &.{body[0].loop};
            };

            return .{
                .idx_ndims = iter_space.idx_ndims,
                .body = body,
            };
        }

        pub fn evalFn(comptime nest: *const @This()) fn (anytype, anytype) void {
            return comptime struct {
                fn eval(in: anytype, inout: anytype) void {
                    const idx: [nest.idx_ndims]usize = .{0} ** nest.idx_ndims;
                    inline for (nest.body) |loop| {
                        loop.eval(in, inout, idx);
                    }
                }
            }.eval;
        }

        pub inline fn eval(
            comptime nest: *const @This(),
            in: anytype,
            inout: anytype,
        ) void {
            const eval_fn = comptime nest.evalFn();
            eval_fn(in, inout);
        }

        // fuse will need to fuse the args of two different nests
        // to build a new type and appropriately pass in the args into the functions
        // which require the originally defined arg types, this will be tricky!
        // pub fn fuse() {}
    };
}

const s = IterationSpace([16][8]bool).init();
const B = [16][8]bool;
const Args = struct {
    b: B,
};

const test_logic: func.IterationLogic(void, Args, 2) = struct {
    // for gpu execution inline this function into a surrounding GPU kernel.
    // Unraveling method would require to be bound to GPU thread / group ids
    inline fn iter_logic(b: *AllocatedBuffer(B), idx: [2]usize) void {
        std.testing.expectEqual(b.constant(false), b.load(idx)) catch unreachable;
        b.store(b.constant(true), idx);
    }
}.iter_logic;

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const nest = s.nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 16,
        .block_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 8,
                    .block_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .block_size = 1,
                    },
                    .body = &.{
                        .{
                            .iter_logic = nest.body[0].body[0].loop.body[0].iter_logic,
                        },
                    },
                },
            },
        },
    };

    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);

    var b = try AllocatedBuffer(B).alloc(arena.allocator());

    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}

test "unroll nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const us = comptime s.unroll(1);
    const nest = us.nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 16,
        .block_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 8,
                    .block_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .block_size = 1,
                    },
                    .unrolled = true,
                    .body = &.{
                        .{
                            .iter_logic = nest.body[0].body[0].loop.body[0].iter_logic,
                        },
                    },
                },
            },
        },
    };

    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}

test "vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vs = comptime s.vectorize();
    try std.testing.expect(@TypeOf(vs).Vec == @Vector(8, bool));
    const nest = comptime vs.nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 16,
        .block_info = .{
            .block_size = 1,
            .num_blocks = 16,
            .orig_dim = 0,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 8,
                    .vector = true,
                    .step_size = 8,
                    .block_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .block_size = 1,
                    },
                    .body = &.{
                        .{
                            .iter_logic = nest.body[0].body[0].loop.body[0].iter_logic,
                        },
                    },
                },
            },
        },
    };

    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}

test "reorder nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const rs = comptime s.reorder(.{ 1, 0 });
    const nest = comptime rs.nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 8,
        .block_info = .{
            .num_blocks = 8,
            .orig_dim = 1,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 16,
                    .block_info = .{
                        .block_size = 1,
                        .num_blocks = 16,
                        .orig_dim = 0,
                    },
                    .body = &.{
                        .{
                            .iter_logic = nest.body[0].body[0].loop.body[0].iter_logic,
                        },
                    },
                },
            },
        },
    };

    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}

test "parallel nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const nest = comptime s.parallel(1).nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 16,
        .block_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 8,
                    .cpu_parallel = true,
                    .block_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .block_size = 1,
                    },
                    .body = &.{
                        .{
                            .iter_logic = nest.body[0].body[0].loop.body[0].iter_logic,
                        },
                    },
                },
            },
        },
    };

    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);

    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}

test "split split vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const ssv = comptime s
        .split(1, 4)
        .split(0, 4)
        .vectorize();
    try comptime std.testing.expectEqual(@TypeOf(ssv), IterationSpace([4][4][2]@Vector(4, bool)));

    const nest = comptime ssv.nest(void, Args, test_logic);
    const expected = comptime Nest(void, Args, 2).Loop{
        .upper = 4,
        .block_info = .{
            .num_blocks = 4,
            .block_size = 4,
            .orig_dim = 0,
        },
        .body = &.{
            .{
                .loop = .{
                    .upper = 4,
                    .block_info = .{
                        .num_blocks = 4,
                        .block_size = 1,
                        .orig_dim = 0,
                    },
                    .body = &.{
                        .{
                            .loop = .{
                                .upper = 2,
                                .block_info = .{
                                    .num_blocks = 2,
                                    .block_size = 4,
                                    .orig_dim = 1,
                                },
                                .body = &.{
                                    .{
                                        .loop = .{
                                            .upper = 4,
                                            .step_size = 4,
                                            .vector = true,
                                            .block_info = .{
                                                .num_blocks = 4,
                                                .block_size = 1,
                                                .orig_dim = 1,
                                            },
                                            .body = &.{
                                                .{
                                                    .iter_logic = nest.body[0].body[0].loop.body[0].loop.body[0].loop.body[0].iter_logic,
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    };
    try comptime std.testing.expectEqualDeep(expected, nest.body[0]);
    var b = try AllocatedBuffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.raw[0..128]);
}
