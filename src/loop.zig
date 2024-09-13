const std = @import("std");
const utils = @import("utils.zig");
const func = @import("func.zig");

const buffer = @import("buffer.zig");
const IterationSpace = @import("iterspace.zig").IterationSpace;

pub fn Nest(comptime Args: type, comptime iter_space: anytype) type {
    return struct {
        const idx_ndims = iter_space.idx_ndims;
        const CurrentNest = @This();

        fn buildLogic(comptime iter_logic: func.Logic(Args, iter_space.Ind)) EvalItem {
            return .{
                .eval = struct {
                    const IterSpaceLogic: type = func.IterSpaceLogic(Args, iter_space);
                    const iter_space_logic = @as(*const IterSpaceLogic, @ptrCast(&iter_logic));
                    /// Explanation of what is going on here:
                    /// Buffers are initially created with a default iter_space in the type, corresponding to the array type,
                    /// not the one that has been transformed by the programmer.
                    /// Both buffers are identical but the additional type information from the iter_space being used will change
                    /// This will allow the usage of the indices enum to correctly identify vector loads and stores
                    inline fn wrapped_iter_logic(args: anytype, idx: [idx_ndims]usize) void {
                        const iter_space_logic_args = @as(*const std.meta.ArgsTuple(IterSpaceLogic), @ptrCast(&(args ++ .{idx}))).*;
                        @call(.always_inline, iter_space_logic, iter_space_logic_args);
                    }
                }.wrapped_iter_logic,
            };
        }

        fn buildLoop(comptime dim: u8, body: []const EvalItem) EvalItem {
            const loop_info = iter_space.loop_info[dim];
            return .{
                .eval = struct {
                    inline fn eval(args: anytype, base_idx: [idx_ndims]usize) void {
                        var idx = base_idx;
                        const base_for_dim = base_idx[loop_info.idx_dim];
                        var threads: [loop_info.num_blocks]std.Thread = undefined;
                        inline for (body) |item| {
                            if (comptime loop_info.unrolled) {
                                comptime var unroll_loopvar = loop_info.idx_min;
                                comptime var i = 0;
                                inline while (i < loop_info.num_blocks) : ({
                                    unroll_loopvar += loop_info.block_size;
                                    idx[loop_info.idx_dim] = base_for_dim + unroll_loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop_info.parallel) {
                                        threads[i] = std.Thread.spawn(.{}, item.eval, .{ args, idx }) catch unreachable;
                                    } else {
                                        item.eval(args, idx);
                                    }
                                }
                            } else {
                                var loopvar = loop_info.idx_min;
                                var i: usize = 0;
                                while (i < loop_info.num_blocks) : ({
                                    loopvar += loop_info.block_size;
                                    idx[loop_info.idx_dim] = base_for_dim + loopvar;
                                    i += 1;
                                }) {
                                    if (comptime loop_info.parallel) {
                                        threads[i] = std.Thread.spawn(.{}, item.eval, .{ args, idx }) catch unreachable;
                                    } else {
                                        item.eval(args, idx);
                                    }
                                }
                            }
                            if (comptime loop_info.parallel) {
                                inline for (threads) |t| {
                                    t.join();
                                }
                            }
                        }
                    }
                }.eval,
            };
        }

        const EvalItem = struct {
            eval: fn (anytype, [idx_ndims]usize) callconv(.Inline) void,
        };

        body: []const EvalItem,

        pub fn init(
            comptime iter_logic: func.Logic(Args, iter_space.Ind),
        ) @This() {
            if (iter_space.ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }

            const body: []const EvalItem = comptime blk: {
                var body: []const EvalItem = &.{buildLogic(iter_logic)};
                for (0..iter_space.ndims) |dim| {
                    body = &.{buildLoop(iter_space.ndims - dim - 1, body)};
                }
                break :blk &.{body[0]};
            };

            return .{
                .body = body,
            };
        }

        pub fn build(comptime nest: *const @This()) fn (anytype) void {
            return comptime struct {
                fn eval(args: anytype) void {
                    const idx: [idx_ndims]usize = .{0} ** idx_ndims;
                    inline for (nest.body) |item| {
                        item.eval(args, idx);
                    }
                }
            }.eval;
        }

        pub inline fn eval(
            comptime nest: *const @This(),
            args: anytype,
        ) void {
            const eval_fn = comptime nest.build();
            eval_fn(args);
        }

        pub fn alloc(
            comptime _: *const @This(),
            comptime Array: type,
            allocator: std.mem.Allocator,
        ) !buffer.IterSpaceBuffer(Array, iter_space) {
            return buffer.IterSpaceBuffer(Array, iter_space).alloc(allocator);
        }

        // fuse will need to fuse the args of two different nests
        // to build a new type and appropriately pass in the args into the functions
        // which require the originally defined arg types, this will be tricky!
        // alternatively, have an execution wrapper that manages arg passing and executes loops / logic manually, and keep nests separate
    };
}

const TestIndices = enum { i, j };

const test_iter_space = IterationSpace([16][8]bool, TestIndices).init();
const B = [16][8]bool;
const TestArgs = struct {
    b: B,
};

const test_logic: func.Logic(TestArgs, TestIndices) = struct {
    // for gpu execution inline this function into a surrounding GPU kernel.
    // Unraveling method would require to be bound to GPU thread / group ids
    inline fn iter_logic(b: *buffer.Buffer(B, TestIndices), idx: [2]usize) void {
        std.testing.expectEqual(b.constant(.{ .i, .j }, false), b.load(.{ .i, .j }, idx)) catch unreachable;
        b.store(.{ .i, .j }, b.constant(.{ .i, .j }, true), idx);
    }
}.iter_logic;

test "nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const nest = test_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .num_blocks = 16,
        .idx_dim = 0,
        .vector = false,
        .block_size = 1,
        .unrolled = false,
        .parallel = false,
    }, .{
        .num_blocks = 8,
        .idx_dim = 1,
        .vector = false,
        .block_size = 1,
        .unrolled = false,
        .parallel = false,
    } }, test_iter_space.loop_info);
    var b = try nest.alloc(B, arena.allocator());

    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "unroll nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const unrolled_iter_space = comptime test_iter_space.unroll(1);
    const nest = unrolled_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{
        .{
            .num_blocks = 16,
            .idx_dim = 0,
            .vector = false,
            .block_size = 1,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .num_blocks = 8,
            .idx_dim = 1,
            .vector = false,
            .block_size = 1,
            .unrolled = true,
            .parallel = false,
        },
    }, unrolled_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vectorized_iter_space = comptime test_iter_space.vectorize(1);
    try std.testing.expect(@TypeOf(vectorized_iter_space).Vec == @Vector(8, bool));
    const nest = comptime vectorized_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .block_size = 1,
        .num_blocks = 16,
        .vector = false,
        .idx_dim = 0,
        .unrolled = false,
        .parallel = false,
    }, .{
        .num_blocks = 1,
        .idx_dim = 1,
        .block_size = 8,
        .vector = true,
        .unrolled = false,
        .parallel = false,
    } }, vectorized_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "reorder nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const reordered_iter_space = comptime test_iter_space.reorder(.{ 1, 0 });
    const nest = comptime reordered_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .num_blocks = 8,
        .idx_dim = 1,
        .vector = false,
        .block_size = 1,
        .unrolled = false,
        .parallel = false,
    }, .{
        .block_size = 1,
        .num_blocks = 16,
        .idx_dim = 0,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    } }, reordered_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "parallel nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const parallel_iter_space = comptime test_iter_space.parallel(1);
    const nest = comptime parallel_iter_space.nest(TestArgs, test_logic);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .block_size = 1,
        .num_blocks = 16,
        .vector = false,
        .idx_dim = 0,
        .unrolled = false,
        .parallel = false,
    }, .{
        .num_blocks = 8,
        .idx_dim = 1,
        .block_size = 1,
        .vector = false,
        .unrolled = false,
        .parallel = true,
    } }, parallel_iter_space.loop_info);

    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}

test "split split vector nest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const transformed_iter_space = comptime test_iter_space
        .split(1, 4)
        .split(0, 4)
        .vectorize(3);
    try comptime std.testing.expectEqual(@TypeOf(transformed_iter_space), IterationSpace([4][4][2][4]bool, TestIndices));

    const nest = comptime transformed_iter_space.nest(TestArgs, test_logic);
    // const expected = comptime Nest(TestArgs, ssv).Loop{
    //     .idx_max = 4,
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{
        .{
            .num_blocks = 4,
            .block_size = 4,
            .idx_dim = 0,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .num_blocks = 4,
            .block_size = 1,
            .idx_dim = 0,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .num_blocks = 2,
            .block_size = 4,
            .idx_dim = 1,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .num_blocks = 1,
            .block_size = 4,
            .idx_dim = 1,
            .vector = true,
            .unrolled = false,
            .parallel = false,
        },
    }, transformed_iter_space.loop_info);
    var b = try nest.alloc(B, arena.allocator());
    nest.eval(.{&b});
    try std.testing.expectEqualSlices(bool, &(.{true} ** 128), b.data[0..128]);
}
