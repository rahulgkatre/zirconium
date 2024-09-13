const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const func = @import("func.zig");
const AllocatedBuffer = @import("buffer.zig").Buffer;

pub fn IterationSpace(comptime Array: type, comptime Indices: type) type {
    return struct {
        const Self = @This();
        pub const dtype: type = utils.Datatype(Array);
        pub const ndims = utils.extractNdims(Array);
        pub const shape: [ndims]usize = utils.extractShape(Array);
        const default_info = blk: {
            var splits: [ndims]utils.LoopInfo = undefined;
            for (0..ndims) |d| {
                splits[d] = .{
                    .idx_dim = d,
                    .block_size = 1,
                    .num_blocks = shape[d],
                    .vector = false,
                    .unrolled = false,
                    .parallel = false,
                };
            }
            break :blk splits;
        };

        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        comptime Ind: type = Indices,
        comptime Arr: type = Array,

        idx_ndims: u8,
        loop_info: *const [ndims]utils.LoopInfo = &default_info,

        pub fn init() Self {
            return .{ .idx_ndims = ndims };
        }

        fn Split(
            comptime dim: u8,
            comptime block_size: usize,
        ) type {
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

            const pre = shape[0..dim];
            const post = shape[dim + 1 .. ndims];
            return IterationSpace(utils.ShapeToArray(dtype, ndims + 1, pre ++ .{ num_blocks, block_size } ++ post), Indices);
        }

        pub fn split(
            comptime b: *const Self,
            comptime dim: u8,
            comptime block_size: usize,
        ) Split(dim, block_size) {
            if (Split(dim, block_size) == Self) {
                return b.*;
            }

            const num_blocks = @divExact(shape[dim], block_size);

            const loop_info1: utils.LoopInfo = .{
                .idx_dim = b.loop_info[dim].idx_dim,
                .num_blocks = num_blocks,
                .block_size = block_size,
                .vector = false,
                .unrolled = false,
                .parallel = false,
            };

            const loop_info2: utils.LoopInfo = .{
                .idx_dim = b.loop_info[dim].idx_dim,
                .num_blocks = block_size,
                .block_size = b.loop_info[dim].block_size,
                .vector = b.loop_info[dim].vector,
                .unrolled = b.loop_info[dim].unrolled,
                .parallel = b.loop_info[dim].parallel,
            };

            return .{
                .idx_ndims = b.idx_ndims,
                .loop_info = b.loop_info[0..dim] ++ .{ loop_info1, loop_info2 } ++ b.loop_info[dim + 1 .. ndims],
            };
        }

        fn TileHelper(
            comptime dim_offset: u8,
            comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize }),
        ) type {
            const dim, const block_size = sorted_tile_config[0];
            if (sorted_tile_config.len > 1) {
                return Split(dim + dim_offset, block_size)
                    .TileHelper(dim_offset + 1, sorted_tile_config[1..]);
            }
            return Split(dim + dim_offset, block_size);
        }

        fn tileHelper(
            comptime b: *const Self,
            comptime dim_offset: u8,
            comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize }),
        ) TileHelper(dim_offset, sorted_tile_config) {
            const dim, const block_size = sorted_tile_config[0];
            if (sorted_tile_config.len > 1) {
                // dim offset is needed because as we process each split, a new dim is added.
                // TODO: sort tile_config by dim before processing it
                return b.split(dim, block_size)
                    .tileHelper(dim_offset + 1, sorted_tile_config[1..]);
            }
            return b.split(dim + dim_offset, block_size);
        }

        fn tileReorder(
            comptime tile_ndims: u8,
            comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize }),
        ) []const u8 {
            var new_order: [tile_ndims]u8 = undefined;
            var added_dims: [tile_ndims]bool = .{false} ** tile_ndims;

            for (sorted_tile_config, 0..) |tile_cfg, count| {
                const idx_dim, _ = tile_cfg;
                const new_dim = idx_dim + count;
                new_order[count + sorted_tile_config[0][0]] = new_dim;
                added_dims[new_dim] = true;
            }

            const inner_dims_offset = sorted_tile_config[0][0] + sorted_tile_config.len;
            var inner_dims_idx = 0;
            for (added_dims, 0..) |is_added, dim| {
                if (!is_added) {
                    if (dim < sorted_tile_config[0][0]) {
                        new_order[dim] = dim;
                    } else {
                        new_order[inner_dims_idx + inner_dims_offset] = dim;
                        inner_dims_idx += 1;
                    }
                }
            }

            return &new_order;
        }

        fn Tile(
            comptime tile_config: []const std.meta.Tuple(&.{ u8, usize }),
        ) type {
            var sorted_tile_config = tile_config[0..tile_config.len].*;
            std.sort.insertion(std.meta.Tuple(&.{ u8, usize }), &sorted_tile_config, {}, struct {
                fn lessThan(_: void, lhs: std.meta.Tuple(&.{ u8, usize }), rhs: std.meta.Tuple(&.{ u8, usize })) bool {
                    return lhs[0] < rhs[0];
                }
            }.lessThan);
            const Tiled = TileHelper(0, &sorted_tile_config);
            return Tiled.Reorder(tileReorder(Tiled.ndims, &sorted_tile_config)[0..Tiled.ndims].*);
        }

        pub fn tile(
            comptime b: *const Self,
            comptime tile_config: []const std.meta.Tuple(&.{ u8, usize }),
        ) Tile(tile_config) {
            // tile helper creates the extra splits, but it still needs to be reordered
            var sorted_tile_config = tile_config[0..tile_config.len].*;
            std.sort.insertion(std.meta.Tuple(&.{ u8, usize }), &sorted_tile_config, {}, struct {
                fn lessThan(_: void, lhs: std.meta.Tuple(&.{ u8, usize }), rhs: std.meta.Tuple(&.{ u8, usize })) bool {
                    return lhs[0] < rhs[0];
                }
            }.lessThan);
            const tile_ndims = Tile(tile_config).ndims;
            return b.tileHelper(0, &sorted_tile_config).reorder(tileReorder(tile_ndims, &sorted_tile_config)[0..tile_ndims].*);
        }

        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr = Array;
        pub const Ind = Indices;

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        pub fn vectorize(
            comptime b: *const Self,
            comptime dim: u8,
        ) Self {
            const new_loop_info = blk: {
                var orig_info = b.loop_info.*;
                orig_info[dim].vector = true;
                orig_info[dim].block_size = shape[ndims - 1];
                orig_info[dim].num_blocks = 1;
                orig_info[dim].parallel = false;
                orig_info[dim].unrolled = false;
                break :blk orig_info;
            };
            return .{
                .loop_info = &new_loop_info,
                .idx_ndims = b.idx_ndims,
            };
        }

        pub fn unroll(
            comptime b: *const Self,
            comptime dim: u8,
        ) Self {
            const new_loop_info = blk: {
                var orig_info = b.loop_info.*;
                orig_info[dim].parallel = false;
                orig_info[dim].unrolled = true;
                orig_info[dim].vector = false;
                break :blk orig_info;
            };
            return .{
                .loop_info = &new_loop_info,
                .idx_ndims = b.idx_ndims,
            };
        }

        fn Reorder(comptime new_order: [ndims]u8) type {
            const new_shape = utils.arrayPermute(usize, ndims, shape, new_order);
            return IterationSpace(utils.ShapeToArray(dtype, ndims, &new_shape), Indices);
        }
        pub fn reorder(
            comptime b: *const Self,
            comptime new_order: [ndims]u8,
        ) Reorder(new_order) {
            return .{
                .loop_info = &comptime utils.arrayPermute(utils.LoopInfo, ndims, b.loop_info.*, new_order),
                .idx_ndims = b.idx_ndims,
            };
        }

        pub fn parallel(
            comptime b: *const Self,
            comptime dim: u8,
        ) Self {
            const new_loop_info = blk: {
                var orig_info = b.loop_info.*;
                orig_info[dim].parallel = true;
                orig_info[dim].unrolled = false;
                orig_info[dim].vector = false;
                break :blk orig_info;
            };
            return .{
                .loop_info = &new_loop_info,
                .idx_ndims = b.idx_ndims,
            };
        }

        pub fn nest(
            comptime self: Self,
            comptime Args: type,
            comptime iter_logic: func.Logic(Args, Indices),
        ) loop.Nest(Args, self) {
            std.debug.assert(std.meta.activeTag(@typeInfo(Indices)) == .Enum and @typeInfo(Indices).Enum.fields.len == self.idx_ndims);
            return loop.Nest(Args, self).init(iter_logic);
        }
    };
}

test "tile" {
    const Ind = enum { i, j, k };
    const tiled = comptime IterationSpace([32][16][8]f32, Ind).init().tile(&.{ .{ 1, 8 }, .{ 2, 4 } });
    try std.testing.expectEqual(@TypeOf(tiled), IterationSpace([32][2][2][8][4]f32, Ind));
}

test "split" {
    const Ind = enum { i, j };
    const st = comptime IterationSpace([16][8]f32, Ind).init().split(0, 4);
    try std.testing.expect(@TypeOf(st) == IterationSpace([4][4][8]f32, Ind));
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{
        .{
            .idx_dim = 0,
            .block_size = 4,
            .num_blocks = 4,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .idx_dim = 0,
            .block_size = 1,
            .num_blocks = 4,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
        .{
            .idx_dim = 1,
            .block_size = 1,
            .num_blocks = 8,
            .vector = false,
            .unrolled = false,
            .parallel = false,
        },
    }, st.loop_info);
}

test "split_split" {
    const Ind = enum { i, j };
    const sst = comptime IterationSpace([16][8]f32, Ind).init().split(0, 4).split(0, 2);
    try std.testing.expect(@TypeOf(sst) == IterationSpace([2][2][4][8]f32, Ind));
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 2,
        .num_blocks = 2,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 0,
        .block_size = 4,
        .num_blocks = 2,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 1,
        .num_blocks = 8,
        .block_size = 1,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    } }, sst.loop_info);
}

test "vectorize" {
    const Ind = enum { i, j };
    const t = comptime IterationSpace([16][8]f32, Ind).init();
    const vt = t.vectorize(1);
    try std.testing.expect(@TypeOf(vt) == IterationSpace([16][8]f32, Ind));
}

test "split_vectorize" {
    const Ind = enum { i, j };
    const svt = comptime IterationSpace([16][8]f32, Ind).init()
        .split(0, 4)
        .vectorize(2);
    try std.testing.expect(@TypeOf(svt) == IterationSpace([4][4][8]f32, Ind));
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 4,
        .num_blocks = 4,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 0,
        .block_size = 1,
        .num_blocks = 4,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 1,
        .num_blocks = 1,
        .block_size = 8,
        .vector = true,
        .unrolled = false,
        .parallel = false,
    } }, svt.loop_info);
}
