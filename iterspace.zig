const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const AllocatedBuffer = @import("buffer.zig").AllocatedBuffer;

pub fn IterationSpace(comptime Array: type) type {
    return struct {
        const Self = @This();
        const dtype: type = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape: [ndims]usize = utils.extractShape(Array);
        const ThisUnit = utils.Unit(Array);
        const ThisVectorized = utils.Vectorized(Array);
        const default_block_info = blk: {
            var splits: [ndims]utils.BlockInfo = undefined;
            for (0..ndims) |d| {
                splits[d] = .{
                    .orig_dim = d,
                    .block_size = 1,
                    .num_blocks = shape[d],
                };
            }
            break :blk splits;
        };

        orig_ndims: u8,
        block_info: *const [ndims]utils.BlockInfo = &default_block_info,
        vector: bool = (ThisUnit == Vec),

        pub fn init() Self {
            return .{
                .orig_ndims = ndims,
            };
        }

        fn Split(comptime dim: u8, comptime block_size: u8) type {
            std.debug.assert(dim < ndims);
            // TODO: Support splits that don't divide evenly
            // Give the option of how to evaluate the uneven part
            // - Pad it to evenly divide
            // - Unfuse into two separate loops (gives more control for unrolling)
            std.debug.assert(@mod(shape[dim], block_size) == 0);

            if (block_size == 1 or block_size == shape[dim]) {
                return Self;
            }

            const num_blocks = @divExact(shape[dim], block_size);

            const pre = if (dim > 0) shape[0..dim] else .{};
            const post = if (dim < ndims - 1) shape[dim + 1 .. ndims] else .{};
            return IterationSpace(utils.ShapeToArray(dtype, ndims + 1, pre ++ .{ num_blocks, block_size } ++ post));
        }

        pub fn split(comptime b: *const Self, comptime dim: u8, comptime block_size: u8) Split(dim, block_size) {
            if (Split(dim, block_size) == Self) {
                return b.*;
            }

            const block_info1: utils.BlockInfo = .{
                .orig_dim = b.block_info[dim].orig_dim,
                .num_blocks = Split(dim, block_size).shape[dim],
                .block_size = block_size,
            };

            const block_info2: utils.BlockInfo = .{
                .orig_dim = b.block_info[dim].orig_dim,
                .num_blocks = Split(dim, block_size).shape[dim + 1],
                .block_size = b.block_info[dim].block_size,
            };

            return .{
                .orig_ndims = b.orig_ndims,
                .block_info = b.block_info[0..dim] ++ .{ block_info1, block_info2 } ++ b.block_info[dim + 1 .. ndims],
            };
        }

        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr = Array;

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        const NonVectorized = if (ThisVectorized != void) Self else void;
        pub fn vectorize(b: *const NonVectorized) IterationSpace(ThisVectorized) {
            return .{ .block_info = b.block_info, .orig_ndims = b.orig_ndims };
        }

        fn buildLoop(comptime b: *const Self, comptime dim: u8, inner: ?*const loop.Nest.Loop) ?loop.Nest.Loop {
            if (dim >= ndims) {
                return null;
            }

            return .{
                // .lower = b.layout.skew[dim],
                .lower = 0,
                .upper = shape[dim],
                .inner = inner,
                .block_info = b.block_info[dim],
                .vector = (ThisUnit == Vec and dim == ndims - 1),
                .step_size = if (ThisUnit == Vec and dim == ndims - 1) shape[ndims - 1] else 1,
            };
        }

        pub fn nest(
            comptime b: *const Self,
            comptime float_mode: std.builtin.FloatMode,
        ) loop.Nest {
            if (ndims == 0) {
                @compileError("cannot generate loop nest for 0 dimensional iteration space");
            }

            const loop_nest: ?loop.Nest.Loop = comptime blk: {
                var curr: ?loop.Nest.Loop = null;
                for (0..ndims) |dim| {
                    if (curr) |curr_loop| {
                        curr = b.buildLoop(ndims - dim - 1, &curr_loop);
                    } else {
                        curr = b.buildLoop(ndims - dim - 1, null);
                    }
                }
                break :blk curr;
            };

            if (loop_nest) |top_loop| {
                return .{
                    .orig_ndims = b.orig_ndims,
                    .float_mode = float_mode,
                    .loop = &top_loop,
                };
            } else {
                return .{
                    .orig_ndims = b.orig_ndims,
                    .float_mode = float_mode,
                    .loop = null,
                };
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const t = comptime IterationSpace([16][8]f32).init();
    var d = try AllocatedBuffer(@TypeOf(t).Arr).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&d.multi[0]) == @intFromPtr(&d.raw[0]));
    try std.testing.expect(@intFromPtr(&d.multi[1][7]) == @intFromPtr(&d.raw[15]));
}

test "split" {
    const st = comptime IterationSpace([16][8]f32).init().split(0, 4);
    try std.testing.expect(@TypeOf(st) == IterationSpace([4][4][8]f32));
    try std.testing.expectEqualSlices(utils.BlockInfo, &.{
        .{
            .orig_dim = 0,
            .block_size = 4,
            .num_blocks = 4,
        },
        .{
            .orig_dim = 0,
            .block_size = 1,
            .num_blocks = 4,
        },
        .{
            .orig_dim = 1,
            .block_size = 1,
            .num_blocks = 8,
        },
    }, st.block_info);
}

test "split_split" {
    const sst = comptime IterationSpace([16][8]f32).init().split(0, 4).split(0, 2);
    try std.testing.expect(@TypeOf(sst) == IterationSpace([2][2][4][8]f32));
    try std.testing.expectEqualSlices(utils.BlockInfo, &.{ .{
        .orig_dim = 0,
        .block_size = 2,
        .num_blocks = 2,
    }, .{
        .orig_dim = 0,
        .block_size = 4,
        .num_blocks = 2,
    }, .{
        .orig_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
    }, .{
        .orig_dim = 1,
        .num_blocks = 8,
        .block_size = 1,
    } }, sst.block_info);
}

test "vectorize" {
    const t = comptime IterationSpace([16][8]f32).init();
    const vt = t.vectorize();
    try std.testing.expect(@TypeOf(vt) == IterationSpace([16]@Vector(8, f32)));
}

test "split_vectorize" {
    const svt = comptime IterationSpace([16][8]f32).init()
        .split(0, 4)
        .vectorize();
    try std.testing.expect(@TypeOf(svt) == IterationSpace([4][4]@Vector(8, f32)));
    try std.testing.expectEqualSlices(utils.BlockInfo, &.{ .{
        .orig_dim = 0,
        .block_size = 4,
        .num_blocks = 4,
    }, .{
        .orig_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
    }, .{
        .orig_dim = 1,
        .num_blocks = 8,
        .block_size = 1,
    } }, svt.block_info);
}
