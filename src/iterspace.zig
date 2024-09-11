const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const func = @import("func.zig");
const AllocatedBuffer = @import("buffer.zig").Buffer;

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

        iter_ndims: u8 = ndims,
        iter_shape: [ndims]usize = shape,
        idx_ndims: u8,
        block_info: *const [ndims]utils.BlockInfo = &default_block_info,
        unrolled_dims: *const [ndims]bool = &(.{false} ** ndims),
        parallel_dims: *const [ndims]bool = &(.{false} ** ndims),
        vector: bool = (ThisUnit == Vec),

        pub fn init() Self {
            return .{
                .idx_ndims = ndims,
            };
        }

        pub fn Indices(comptime self: Self, comptime names: [self.idx_ndims][:0]const u8) type {
            const idx_ndims = self.idx_ndims;
            var fields: [idx_ndims]std.builtin.Type.EnumField = undefined;
            for (names, 0..) |name, i| {
                fields[i].name = name;
                fields[i].value = @intCast(i);
            }
            return @Type(std.builtin.Type{ .Enum = .{
                .decls = &.{},
                .fields = &fields,
                .is_exhaustive = true,
                .tag_type = u8,
            } });
        }

        fn Split(comptime dim: u8, comptime block_size: usize) type {
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
            return IterationSpace(utils.ShapeToArray(dtype, ndims + 1, pre ++ .{ num_blocks, block_size } ++ post));
        }

        pub fn split(comptime b: *const Self, comptime dim: u8, comptime block_size: usize) Split(dim, block_size) {
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
                .unrolled_dims = b.unrolled_dims[0..dim] ++ .{false} ++ b.unrolled_dims[dim..ndims],
                .parallel_dims = b.parallel_dims[0..dim] ++ .{false} ++ b.parallel_dims[dim..ndims],
            };
        }

        fn TileHelper(comptime dim_offset: u8, comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize })) type {
            const dim, const block_size = sorted_tile_config[0];
            if (sorted_tile_config.len > 1) {
                return Split(dim + dim_offset, block_size)
                    .TileHelper(dim_offset + 1, sorted_tile_config[1..]);
            }
            return Split(dim + dim_offset, block_size);
        }

        fn tileHelper(comptime b: *const Self, comptime dim_offset: u8, comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize })) TileHelper(dim_offset, sorted_tile_config) {
            const dim, const block_size = sorted_tile_config[0];
            if (sorted_tile_config.len > 1) {
                // dim offset is needed because as we process each split, a new dim is added.
                // TODO: sort tile_config by dim before processing it
                return b.split(dim, block_size)
                    .tileHelper(dim_offset + 1, sorted_tile_config[1..]);
            }
            return b.split(dim + dim_offset, block_size);
        }

        fn tileReorder(comptime tile_ndims: u8, comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize })) []const u8 {
            var new_order: [tile_ndims]u8 = undefined;
            var added_dims: [tile_ndims]bool = .{false} ** tile_ndims;

            for (sorted_tile_config, 0..) |tile_cfg, count| {
                const orig_dim, _ = tile_cfg;
                const new_dim = orig_dim + count;
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

        fn Tile(comptime tile_config: []const std.meta.Tuple(&.{ u8, usize })) type {
            var sorted_tile_config = tile_config[0..tile_config.len].*;
            std.sort.insertion(std.meta.Tuple(&.{ u8, usize }), &sorted_tile_config, {}, struct {
                fn lessThan(_: void, lhs: std.meta.Tuple(&.{ u8, usize }), rhs: std.meta.Tuple(&.{ u8, usize })) bool {
                    return lhs[0] < rhs[0];
                }
            }.lessThan);
            const Tiled = TileHelper(0, &sorted_tile_config);
            return Tiled.Reorder(tileReorder(Tiled.ndims, &sorted_tile_config)[0..Tiled.ndims].*);
        }

        pub fn tile(comptime b: *const Self, comptime tile_config: []const std.meta.Tuple(&.{ u8, usize })) Tile(tile_config) {
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

        // By setting NonVectorized to void if it is already vectorized
        // the following function is removed from the namespace of this type
        const NonVectorized = if (ThisVectorized != void) Self else void;
        pub fn vectorize(b: *const NonVectorized) IterationSpace(ThisVectorized) {
            return .{
                .block_info = b.block_info,
                .idx_ndims = b.idx_ndims,
                .unrolled_dims = b.unrolled_dims,
                .parallel_dims = b.parallel_dims,
            };
        }

        pub fn unroll(comptime b: *const Self, comptime dim: u8) Self {
            const new_unrolled_dims: *const [ndims]bool = &comptime blk: {
                var orig_unrolled_dims = b.unrolled_dims.*;
                std.debug.assert(!orig_unrolled_dims[dim]);
                orig_unrolled_dims[dim] = true;
                break :blk orig_unrolled_dims;
            };
            return .{
                .block_info = b.block_info,
                .idx_ndims = b.idx_ndims,
                .unrolled_dims = new_unrolled_dims,
                .parallel_dims = b.parallel_dims,
            };
        }

        fn Reorder(comptime new_order: [ndims]u8) type {
            const new_shape = utils.arrayPermute(usize, ndims, shape, new_order);
            return IterationSpace(utils.ShapeToArray(dtype, ndims, &new_shape));
        }
        pub fn reorder(comptime b: *const Self, comptime new_order: [ndims]u8) Reorder(new_order) {
            return .{
                .block_info = &comptime utils.arrayPermute(utils.BlockInfo, ndims, b.block_info.*, new_order),
                .idx_ndims = b.idx_ndims,
                .unrolled_dims = &comptime utils.arrayPermute(bool, ndims, b.unrolled_dims.*, new_order),
                .parallel_dims = &comptime utils.arrayPermute(bool, ndims, b.parallel_dims.*, new_order),
            };
        }

        pub fn parallel(comptime b: *const Self, comptime dim: u8) Self {
            const new_parallel_dims: *const [ndims]bool = &comptime blk: {
                var orig_parallel_dims = b.parallel_dims.*;
                std.debug.assert(!orig_parallel_dims[dim]);
                orig_parallel_dims[dim] = true;
                break :blk orig_parallel_dims;
            };
            return .{
                .block_info = b.block_info,
                .idx_ndims = b.idx_ndims,
                .unrolled_dims = b.unrolled_dims,
                .parallel_dims = new_parallel_dims,
            };
        }

        pub fn nest(
            comptime self: Self,
            comptime Args: type,
            comptime idx_names: [self.idx_ndims][:0]const u8,
            comptime iter_logic: func.Logic(Args, Indices(self, idx_names)),
        ) loop.Nest(Args, self) {
            return loop.Nest(Args, self).init(Indices(self, idx_names), iter_logic);
        }
    };
}

test "tile" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const tiled = comptime IterationSpace([32][16][8]f32).init().tile(&.{ .{ 1, 8 }, .{ 2, 4 } });
    try std.testing.expectEqual(@TypeOf(tiled), IterationSpace([32][2][2][8][4]f32));
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
