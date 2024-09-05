const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const func = @import("func.zig");
const AllocatedBuffer = @import("buffer.zig").AllocatedBuffer;

pub fn IterationSpace(comptime Array: type) type {
    return struct {
        const Self = @This();
        pub const dtype: type = utils.Datatype(Array);
        pub const ndims = utils.extractNdims(Array);
        pub const shape: [ndims]usize = utils.extractShape(Array);
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

        idx_ndims: u8,
        block_info: *const [ndims]utils.BlockInfo = &default_block_info,
        vector: bool = (ThisUnit == Vec),

        pub fn init() Self {
            return .{
                .idx_ndims = ndims,
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
                .idx_ndims = b.idx_ndims,
                .block_info = b.block_info[0..dim] ++ .{ block_info1, block_info2 } ++ b.block_info[dim + 1 .. ndims],
            };
        }

        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr = Array;

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        const NonVectorized = if (ThisVectorized != void) Self else void;
        pub fn vectorize(b: *const NonVectorized) IterationSpace(ThisVectorized) {
            return .{ .block_info = b.block_info, .idx_ndims = b.idx_ndims };
        }

        pub fn nest(
            comptime self: *const Self,
            comptime In: type,
            comptime InOut: type,
            comptime iter_logic: func.IterationLogic(In, InOut, self.idx_ndims),
        ) loop.Nest(In, InOut, self.idx_ndims) {
            return loop.Nest(In, InOut, self.idx_ndims).init(self, iter_logic);
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
