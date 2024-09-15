const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const func = @import("func.zig");
const AllocatedBuffer = @import("buffer.zig").Buffer;

const IterSpace = @This();

DataIndex: type,
LoopVar: type,
loop_info: []const utils.LoopInfo,

pub inline fn ndims(is: *const IterSpace) u8 {
    return is.loop_info.len;
}

pub inline fn indexName(is: *const IterSpace, dim: u8) [:0]const u8 {
    return @typeInfo(is.DataIndex).Enum.fields[dim].name;
}

pub inline fn numDataIndices(is: *const IterSpace) u8 {
    return @typeInfo(is.DataIndex).Enum.fields.len;
}

pub inline fn numLoopVars(is: *const IterSpace) u8 {
    return @typeInfo(is.LoopVar).Enum.fields.len;
}

pub inline fn size(is: *const IterSpace, dim: u8) ?usize {
    return is.loop_info[dim].num_blocks;
}

pub fn init(
    comptime Shape: type,
    comptime DataIndex: type,
) IterSpace {
    const _ndims = utils.extractNdims(Shape);
    const _shape = utils.extractShape(Shape);
    const default_loop_info = comptime blk: {
        var loop_info: [_ndims]utils.LoopInfo = undefined;
        for (0.._ndims) |d| {
            loop_info[d] = .{
                .idx_dim = d,
                .num_blocks = _shape[d],
            };
        }
        break :blk loop_info;
    };
    return .{
        .DataIndex = DataIndex,
        .LoopVar = DataIndex,
        .loop_info = &default_loop_info,
    };
}

fn splitDimInt(
    comptime is: *const IterSpace,
    comptime dim: u8,
    comptime block_size: usize,
) IterSpace {
    if (block_size == 1) {
        return is.*;
    }
    if (is.loop_info[dim].num_blocks) |num_blocks| {
        if (num_blocks == block_size) {
            return is.*;
        }
    }

    // Outer num_blocks can be null (runtime defined) if splitting on a runtime defined size dimension
    const num_blocks: ?usize = if (is.loop_info[dim].num_blocks) |num_blocks| @divExact(num_blocks, block_size) else null;

    // Outer loop can be made parallel, but cannot be unrolled or vectorized
    const outer_info: utils.LoopInfo = .{
        .idx_dim = is.loop_info[dim].idx_dim,
        .num_blocks = num_blocks,
        .block_size = block_size,
        .vector = false,
        .unrolled = false,
        .parallel = is.loop_info[dim].parallel,
    };
    // Technically inner loop could also be made parallel but 2 levels of threading is crazy
    // block_size is fixed, so it can be used to vectorize or unroll
    const inner_info: utils.LoopInfo = .{
        .idx_dim = is.loop_info[dim].idx_dim,
        .num_blocks = block_size,
        .block_size = is.loop_info[dim].block_size,
        .vector = is.loop_info[dim].vector,
        .unrolled = is.loop_info[dim].unrolled,
        .parallel = null,
    };

    var newIndicesFields: [is.ndims() + 1]std.builtin.Type.EnumField = undefined;
    var d = 0;
    for (0..dim) |_| {
        newIndicesFields[d] = @typeInfo(is.LoopVar).Enum.fields[d];
        d += 1;
    }
    newIndicesFields[d] =
        std.builtin.Type.EnumField{
        .name = std.meta.fieldNames(is.LoopVar)[dim] ++ std.meta.fieldNames(is.DataIndex)[is.loop_info[dim].idx_dim],
        .value = @as(comptime_int, @intCast(dim)),
    };
    d += 1;
    for (dim..is.ndims()) |_| {
        newIndicesFields[d] =
            std.builtin.Type.EnumField{
            .name = std.meta.fieldNames(is.LoopVar)[d - 1],
            .value = @as(comptime_int, @intCast(d)),
        };
        d += 1;
    }
    const NewLoopVar = @Type(.{
        .Enum = std.builtin.Type.Enum{
            .decls = &.{},
            .fields = &newIndicesFields,
            .is_exhaustive = true,
            .tag_type = u8,
        },
    });

    return .{
        .DataIndex = is.DataIndex,
        .LoopVar = NewLoopVar,
        .loop_info = is.loop_info[0..dim] ++ .{ outer_info, inner_info } ++ is.loop_info[dim + 1 .. is.ndims()],
    };
}

pub fn split(
    comptime is: *const IterSpace,
    comptime dim: is.LoopVar,
    comptime block_size: usize,
) IterSpace {
    return is.splitDimInt(@intFromEnum(dim), block_size);
}

fn tileHelper(
    comptime is: *const IterSpace,
    comptime dim_offset: u8,
    comptime sorted_tiling: []const std.meta.Tuple(&.{ u8, usize }),
) IterSpace {
    const dim, const block_size = sorted_tiling[0];
    if (sorted_tiling.len > 1) {
        // dim offset is needed because as we process each split, a new dim is added.
        return is.splitDimInt(dim, block_size)
            .tileHelper(dim_offset + 1, sorted_tiling[1..]);
    }
    return is.splitDimInt(dim + dim_offset, block_size);
}

/// Splits and reorders as per tiling
///
/// Example:
///
/// original: `[..., i, ..., j, ..., k]`
///
/// splits result in `[..., ii, i, ..., jj, j, ..., kk, k]`
///
/// reorder results in `[..., ii, jj, kk, i, ..., j, ..., k]`
pub fn tile(
    comptime is: *const IterSpace,
    comptime tiling: []const std.meta.Tuple(&.{ is.LoopVar, usize }),
) IterSpace {
    // tile helper creates the extra splits, but it still needs to be reordered
    // also convert from enums to ints
    var sorted_tiling: [tiling.len]std.meta.Tuple(&.{ u8, usize }) = undefined;
    for (tiling, 0..) |t, i| {
        sorted_tiling[i] = .{ @intFromEnum(t[0]), t[1] };
    }
    std.sort.insertion(std.meta.Tuple(&.{ u8, usize }), &sorted_tiling, {}, struct {
        fn lessThan(_: void, lhs: std.meta.Tuple(&.{ u8, usize }), rhs: std.meta.Tuple(&.{ u8, usize })) bool {
            return lhs[0] < rhs[0];
        }
    }.lessThan);
    const tiled = is.tileHelper(0, &sorted_tiling);

    // compute the new dimension order
    var new_order: [tiled.ndims()]u8 = undefined;
    var added_dims: [tiled.ndims()]bool = .{false} ** tiled.ndims();
    for (sorted_tiling, 0..) |tile_cfg, count| {
        const idx_dim, _ = tile_cfg;
        const new_dim = idx_dim + count;
        new_order[count + sorted_tiling[0][0]] = new_dim;
        added_dims[new_dim] = true;
    }

    const inner_dims_offset = sorted_tiling[0][0] + sorted_tiling.len;
    var inner_dims_idx = 0;
    for (added_dims, 0..) |is_added, dim| {
        if (!is_added) {
            if (dim < sorted_tiling[0][0]) {
                new_order[dim] = dim;
            } else {
                new_order[inner_dims_idx + inner_dims_offset] = dim;
                inner_dims_idx += 1;
            }
        }
    }

    return tiled.reorder(new_order);
}

pub fn reorder(
    comptime is: *const IterSpace,
    comptime new_order: [is.ndims()]u8,
) IterSpace {
    var newIndicesFields: [is.ndims()]std.builtin.Type.EnumField = utils.arrayPermute(
        std.builtin.Type.EnumField,
        is.ndims(),
        @typeInfo(is.LoopVar).Enum.fields[0..is.ndims()].*,
        new_order,
    );
    for (0..is.ndims()) |d| {
        newIndicesFields[d].value = d;
    }
    const NewLoopVar = @Type(.{
        .Enum = std.builtin.Type.Enum{
            .decls = &.{},
            .fields = &newIndicesFields,
            .is_exhaustive = true,
            .tag_type = u8,
        },
    });
    return .{
        .loop_info = &comptime utils.arrayPermute(utils.LoopInfo, is.ndims(), is.loop_info[0..is.ndims()].*, new_order),
        .DataIndex = is.DataIndex,
        .LoopVar = NewLoopVar,
    };
}

// By setting NonVectorized to void if it is already vectorized
// the following function is removed from the namespace of this type
pub fn vectorize(
    comptime is: *const IterSpace,
    comptime dim: is.LoopVar,
) IterSpace {
    const d = @intFromEnum(dim);
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[d].vector = true;
        loop_info[d].block_size = is.loop_info[d].num_blocks.?;
        loop_info[d].num_blocks = 1;
        loop_info[d].parallel = null;
        loop_info[d].unrolled = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .DataIndex = is.DataIndex,
        .LoopVar = is.LoopVar,
    };
}

pub fn unroll(
    comptime is: *const IterSpace,
    comptime dim: is.LoopVar,
) IterSpace {
    const d = @intFromEnum(dim);
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[d].parallel = null;
        loop_info[d].unrolled = true;
        loop_info[d].vector = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .DataIndex = is.DataIndex,
        .LoopVar = is.LoopVar,
    };
}

pub fn parallel(
    comptime is: *const IterSpace,
    comptime dim: is.LoopVar,
    comptime n_threads: ?u8,
) IterSpace {
    const d = @intFromEnum(dim);
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[d].parallel = n_threads orelse is.loop_info[d].num_blocks;
        loop_info[d].unrolled = false;
        loop_info[d].vector = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .DataIndex = is.DataIndex,
        .LoopVar = is.LoopVar,
    };
}

pub fn nest(
    comptime is: IterSpace,
    comptime Args: type,
    comptime iter_logic: func.Logic(Args, is.DataIndex),
) loop.Nest(Args, is) {
    std.debug.assert(std.meta.activeTag(@typeInfo(is.DataIndex)) == .Enum and @typeInfo(is.DataIndex).Enum.fields.len == is.numDataIndices());
    return loop.Nest(Args, is).init(iter_logic);
}

test tile {
    const DataIndex = enum { i, j, k };
    const tiled_iter_space = comptime IterSpace.init([32][16][8]f32, DataIndex).tile(&.{ .{ .j, 8 }, .{ .k, 4 } });
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4 }, &.{
        @intFromEnum(tiled_iter_space.LoopVar.i),
        @intFromEnum(tiled_iter_space.LoopVar.jj),
        @intFromEnum(tiled_iter_space.LoopVar.kk),
        @intFromEnum(tiled_iter_space.LoopVar.j),
        @intFromEnum(tiled_iter_space.LoopVar.k),
    });
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .num_blocks = 32,
    }, .{
        .idx_dim = 1,
        .block_size = 8,
        .num_blocks = 2,
    }, .{
        .idx_dim = 2,
        .block_size = 4,
        .num_blocks = 2,
    }, .{
        .idx_dim = 1,
        .num_blocks = 8,
    }, .{
        .idx_dim = 2,
        .num_blocks = 4,
    } }, tiled_iter_space.loop_info);
}

test split {
    const DataIndex = enum { i, j };
    const split_iter_space = comptime IterSpace.init([16][8]f32, DataIndex).split(.i, 4);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2 }, &.{
        @intFromEnum(split_iter_space.LoopVar.ii),
        @intFromEnum(split_iter_space.LoopVar.i),
        @intFromEnum(split_iter_space.LoopVar.j),
    });
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 4,
        .num_blocks = 4,
    }, .{
        .idx_dim = 0,
        .num_blocks = 4,
    }, .{
        .idx_dim = 1,
        .num_blocks = 8,
    } }, split_iter_space.loop_info);
}

test "split split" {
    const DataIndex = enum { i, j };
    const split_split_iter_space = comptime IterSpace
        .init([16][8]f32, DataIndex)
        .split(.i, 4)
        .split(.ii, 2);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 2,
        .num_blocks = 2,
    }, .{
        .idx_dim = 0,
        .block_size = 4,
        .num_blocks = 2,
    }, .{
        .idx_dim = 0,
        .num_blocks = 4,
    }, .{
        .idx_dim = 1,
        .num_blocks = 8,
    } }, split_split_iter_space.loop_info);
}

test "vectorize" {
    const DataIndex = enum { i, j };
    const t = comptime IterSpace.init([16][8]f32, DataIndex);
    _ = t.vectorize(.j);
}

test "split vectorize" {
    const DataIndex = enum { i, j };
    const split_vectorized_iter_space = comptime IterSpace.init([16][8]f32, DataIndex)
        .split(.i, 4)
        .vectorize(.j);
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 4,
        .num_blocks = 4,
    }, .{
        .idx_dim = 0,
        .num_blocks = 4,
    }, .{
        .idx_dim = 1,
        .num_blocks = 1,
        .block_size = 8,
        .vector = true,
    } }, split_vectorized_iter_space.loop_info);
}
