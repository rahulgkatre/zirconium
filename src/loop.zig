const std = @import("std");
const utils = @import("utils.zig");
const func = @import("func.zig");

const Buffer = @import("buffer.zig").Buffer;
const IterationSpace = @import("iterspace.zig").IterationSpace;

pub fn Nest(comptime Args: type, comptime iter_space: anytype) type {
    return struct {
        const idx_ndims = iter_space.idx_ndims;
        const CurrentNest = @This();
        const IterLogicWrapper = struct {
            eval: fn (anytype, anytype, [idx_ndims]usize) callconv(.Inline) void,
        };

        // common interface here is that both have an eval() function
        const LoopBodyItem = union(enum) {
            loop: Loop,
            iter_logic: IterLogicWrapper,
        };

        pub const Loop = struct {
            idx_min: usize = 0,
            idx_max: usize,

            // step_size: usize = 1,
            unrolled: bool = false,
            idx_info: utils.IndexInfo,
            cpu_parallel: bool = false,

            body: []const LoopBodyItem = &.{},

            inline fn eval(
                comptime loop: *const Loop,
                in: anytype,
                inout: anytype,
                base_idx: [idx_ndims]usize,
            ) void {
                var idx = base_idx;
                const base_for_dim = base_idx[loop.idx_info.orig_dim];
                var threads: [loop.idx_info.num_blocks]std.Thread = undefined;

                inline for (loop.body) |item| {
                    switch (item) {
                        inline else => |inner| {
                            if (comptime loop.unrolled) {
                                comptime var unroll_loopvar = loop.idx_min;
                                comptime var i = 0;
                                inline while (i < loop.idx_info.num_blocks) : ({
                                    unroll_loopvar += loop.idx_info.block_size;
                                    idx[loop.idx_info.orig_dim] = base_for_dim + unroll_loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop.cpu_parallel) {
                                        threads[i] = std.Thread.spawn(.{}, inner.eval, .{ in, inout, idx }) catch unreachable;
                                    } else {
                                        inner.eval(in, inout, idx);
                                    }
                                }
                            } else {
                                var loopvar = loop.idx_min;
                                var i: usize = 0;
                                while (i < loop.idx_info.num_blocks) : ({
                                    loopvar += loop.idx_info.block_size;
                                    idx[loop.idx_info.orig_dim] = base_for_dim + loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop.cpu_parallel) {
                                        threads[i] = std.Thread.spawn(.{}, inner.eval, .{ in, inout, idx }) catch unreachable;
                                    } else {
                                        inner.eval(in, inout, idx);
                                    }
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
            comptime dim: u8,
            body: []const LoopBodyItem,
        ) Loop {
            return .{
                .idx_min = 0,
                .idx_max = iter_space.iter_shape[dim],
                .body = body,
                .idx_info = iter_space.idx_info[dim],
                .unrolled = iter_space.unrolled_dims[dim],
                .cpu_parallel = iter_space.parallel_dims[dim],
            };
        }

        pub fn init(
            comptime iter_logic: func.Logic(Args, iter_space.Indices()),
        ) @This() {
            if (iter_space.iter_ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }

            const iter_logic_wrapper: IterLogicWrapper = .{
                .eval = struct {
                    inline fn wrapped_iter_logic(in: anytype, inout: anytype, idx: [idx_ndims]usize) void {
                        const IterSpaceLogic: type = func.IterSpaceLogic(Args, iter_space);
                        const iter_space_logic = comptime @as(*const IterSpaceLogic, @ptrCast(&iter_logic));
                        const iter_space_logic_args = @as(*const std.meta.ArgsTuple(IterSpaceLogic), @ptrCast(&(in ++ inout ++ .{idx}))).*;
                        @call(.always_inline, iter_space_logic, iter_space_logic_args);
                    }
                }.wrapped_iter_logic,
            };

            const body: []const Loop = comptime blk: {
                var body: []const LoopBodyItem = &.{LoopBodyItem{ .iter_logic = iter_logic_wrapper }};
                for (0..iter_space.iter_ndims) |dim| {
                    body = &.{
                        LoopBodyItem{
                            .loop = buildLoop(
                                iter_space.iter_ndims - dim - 1,
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

const test_iter_space = IterationSpace([16][8]bool).init(.{ "dim0", "dim1" });
const B = [16][8]bool;
const TestArgs = struct {
    b: B,
};

const TestIndices = test_iter_space.Indices();

const test_logic: func.Logic(TestArgs, TestIndices) = struct {
    // for gpu execution inline this function into a surrounding GPU kernel.
    // Unraveling method would require to be bound to GPU thread / group ids
    inline fn iter_logic(b: *Buffer(B), idx: [2]usize) void {
        std.testing.expectEqual(b.constant(false), b.load(.{ .dim0, .dim1 }, idx)) catch unreachable;
        b.store(.{ .dim0, .dim1 }, b.constant(true), idx);
    }
}.iter_logic;

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const nest = test_iter_space.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, test_iter_space).Loop{
        .idx_max = 16,
        .idx_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .vector = false,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 8,
                    .idx_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .vector = false,
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

    var b = try Buffer(B).alloc(arena.allocator());

    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "unroll nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const us = comptime test_iter_space.unroll(1);
    const nest = us.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, us).Loop{
        .idx_max = 16,
        .idx_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .vector = false,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 8,
                    .idx_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .vector = false,
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

    var b = try Buffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vs = comptime test_iter_space.vectorize();
    try std.testing.expect(@TypeOf(vs).Vec == @Vector(8, bool));
    const nest = comptime vs.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, vs).Loop{
        .idx_max = 16,
        .idx_info = .{
            .block_size = 1,
            .num_blocks = 16,
            .vector = false,
            .orig_dim = 0,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 8,
                    .idx_info = .{
                        .num_blocks = 1,
                        .orig_dim = 1,
                        .block_size = 8,
                        .vector = true,
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

    var b = try Buffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "reorder nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const rs = comptime test_iter_space.reorder(.{ 1, 0 });
    const nest = comptime rs.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, rs).Loop{
        .idx_max = 8,
        .idx_info = .{
            .num_blocks = 8,
            .orig_dim = 1,
            .vector = false,
            .block_size = 1,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 16,
                    .idx_info = .{
                        .block_size = 1,
                        .num_blocks = 16,
                        .orig_dim = 0,
                        .vector = false,
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

    var b = try Buffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "parallel nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const ps = comptime test_iter_space.parallel(1);
    const nest = comptime ps.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, ps).Loop{
        .idx_max = 16,
        .idx_info = .{
            .num_blocks = 16,
            .orig_dim = 0,
            .block_size = 1,
            .vector = false,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 8,
                    .cpu_parallel = true,
                    .idx_info = .{
                        .num_blocks = 8,
                        .orig_dim = 1,
                        .block_size = 1,
                        .vector = false,
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

    var b = try Buffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "split split vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const ssv = comptime test_iter_space
        .split(1, 4)
        .split(0, 4)
        .vectorize();
    try comptime std.testing.expectEqual(@TypeOf(ssv), IterationSpace([4][4][2]@Vector(4, bool)));

    const nest = comptime ssv.nest(TestArgs, test_logic);
    const expected = comptime Nest(TestArgs, ssv).Loop{
        .idx_max = 4,
        .idx_info = .{
            .num_blocks = 4,
            .block_size = 4,
            .orig_dim = 0,
            .vector = false,
        },
        .body = &.{
            .{
                .loop = .{
                    .idx_max = 4,
                    .idx_info = .{
                        .num_blocks = 4,
                        .block_size = 1,
                        .orig_dim = 0,
                        .vector = false,
                    },
                    .body = &.{
                        .{
                            .loop = .{
                                .idx_max = 2,
                                .idx_info = .{
                                    .num_blocks = 2,
                                    .block_size = 4,
                                    .orig_dim = 1,
                                    .vector = false,
                                },
                                .body = &.{
                                    .{
                                        .loop = .{
                                            .idx_max = 4,
                                            .idx_info = .{
                                                .num_blocks = 1,
                                                .block_size = 4,
                                                .orig_dim = 1,
                                                .vector = true,
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
    var b = try Buffer(B).alloc(arena.allocator());
    nest.eval(.{}, .{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}
