const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");
const func = @import("func.zig");
const AllocatedBuffer = @import("buffer.zig").Buffer;

const IterSpace = @This();

Indices: type,
// Arr: type,

idx_ndims: u8,
loop_info: []const utils.LoopInfo,

pub inline fn ndims(is: *const IterSpace) u8 {
    return is.loop_info.len;
}

pub inline fn size(is: *const IterSpace, dim: u8) ?usize {
    return is.loop_info[dim].num_blocks;
}

pub fn init(
    comptime iterspace_shape: anytype,
    comptime Indices: type,
) IterSpace {
    const _ndims = iterspace_shape.len;
    const default_loop_info = comptime blk: {
        var loop_info: [_ndims]utils.LoopInfo = undefined;
        for (0.._ndims) |d| {
            loop_info[d] = .{
                .idx_dim = d,
                .num_blocks = iterspace_shape[d],
            };
        }
        break :blk loop_info;
    };
    return .{
        .idx_ndims = _ndims,
        .Indices = Indices,
        .loop_info = &default_loop_info,
    };
}

pub fn split(
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
        .parallel = false,
    };

    return .{
        .Indices = is.Indices,
        .idx_ndims = is.idx_ndims,
        .loop_info = is.loop_info[0..dim] ++ .{ outer_info, inner_info } ++ is.loop_info[dim + 1 .. is.ndims()],
    };
}

fn tileHelper(
    comptime is: *const IterSpace,
    comptime dim_offset: u8,
    comptime sorted_tile_config: []const std.meta.Tuple(&.{ u8, usize }),
) IterSpace {
    const dim, const block_size = sorted_tile_config[0];
    if (sorted_tile_config.len > 1) {
        // dim offset is needed because as we process each split, a new dim is added.
        // TODO: sort tile_config by dim before processing it
        return is.split(dim, block_size)
            .tileHelper(dim_offset + 1, sorted_tile_config[1..]);
    }
    return is.split(dim + dim_offset, block_size);
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

pub fn tile(
    comptime is: *const IterSpace,
    comptime tile_config: []const std.meta.Tuple(&.{ u8, usize }),
) IterSpace {
    // tile helper creates the extra splits, but it still needs to be reordered
    var sorted_tile_config = tile_config[0..tile_config.len].*;
    std.sort.insertion(std.meta.Tuple(&.{ u8, usize }), &sorted_tile_config, {}, struct {
        fn lessThan(_: void, lhs: std.meta.Tuple(&.{ u8, usize }), rhs: std.meta.Tuple(&.{ u8, usize })) bool {
            return lhs[0] < rhs[0];
        }
    }.lessThan);
    const tiled = is.tileHelper(0, &sorted_tile_config);
    const tiled_ndims = tiled.loop_info.len;
    return tiled.reorder(tileReorder(tiled_ndims, &sorted_tile_config)[0..tiled_ndims].*);
}

pub fn reorder(
    comptime is: *const IterSpace,
    comptime new_order: [is.ndims()]u8,
) IterSpace {
    return .{
        .loop_info = &comptime utils.arrayPermute(utils.LoopInfo, is.ndims(), is.loop_info[0..is.ndims()].*, new_order),
        .idx_ndims = is.idx_ndims,
        .Indices = is.Indices,
    };
}

// By setting NonVectorized to void if it is already vectorized
// the following function is removed from the namespace of this type
pub fn vectorize(
    comptime is: *const IterSpace,
    comptime dim: u8,
) IterSpace {
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[dim].vector = true;
        loop_info[dim].block_size = is.loop_info[dim].num_blocks.?;
        loop_info[dim].num_blocks = 1;
        loop_info[dim].parallel = false;
        loop_info[dim].unrolled = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .idx_ndims = is.idx_ndims,
        .Indices = is.Indices,
    };
}

pub fn unroll(
    comptime is: *const IterSpace,
    comptime dim: u8,
) IterSpace {
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[dim].parallel = false;
        loop_info[dim].unrolled = true;
        loop_info[dim].vector = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .idx_ndims = is.idx_ndims,
        .Indices = is.Indices,
    };
}

pub fn parallel(
    comptime is: *const IterSpace,
    comptime dim: u8,
) IterSpace {
    const new_info = blk: {
        var loop_info = is.loop_info[0..is.ndims()].*;
        loop_info[dim].parallel = true;
        loop_info[dim].unrolled = false;
        loop_info[dim].vector = false;
        break :blk loop_info;
    };
    return .{
        .loop_info = &new_info,
        .idx_ndims = is.idx_ndims,
        .Indices = is.Indices,
    };
}

pub fn nest(
    comptime is: IterSpace,
    comptime Args: type,
    comptime iter_logic: func.Logic(Args, is.Indices),
) loop.Nest(Args, is) {
    std.debug.assert(std.meta.activeTag(@typeInfo(is.Indices)) == .Enum and @typeInfo(is.Indices).Enum.fields.len == is.idx_ndims);
    return loop.Nest(Args, is).init(iter_logic);
}

test tile {
    const Indices = enum { i, j, k };
    const tiled_iter_space = comptime IterSpace.init(.{ 32, 16, 8 }, Indices).tile(&.{ .{ 1, 8 }, .{ 2, 4 } });
    try std.testing.expectEqualSlices(utils.LoopInfo, &.{ .{
        .idx_dim = 0,
        .block_size = 1,
        .num_blocks = 32,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 1,
        .block_size = 8,
        .num_blocks = 2,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 2,
        .block_size = 4,
        .num_blocks = 2,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 1,
        .block_size = 1,
        .num_blocks = 8,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    }, .{
        .idx_dim = 2,
        .block_size = 1,
        .num_blocks = 4,
        .vector = false,
        .unrolled = false,
        .parallel = false,
    } }, tiled_iter_space.loop_info);
}

test split {
    const Indices = enum { i, j };
    const split_iter_space = comptime IterSpace.init(.{ 16, 8 }, Indices).split(0, 4);
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
    const Indices = enum { i, j };
    const split_split_iter_space = comptime IterSpace
        .init(.{ 16, 8 }, Indices)
        .split(0, 4)
        .split(0, 2);
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
    }, .{
        .idx_dim = 0,
        .num_blocks = 4,
    }, .{
        .idx_dim = 1,
        .num_blocks = 8,
    } }, split_split_iter_space.loop_info);
}

test "vectorize" {
    const Indices = enum { i, j };
    const t = comptime IterSpace.init(.{ 16, 8 }, Indices);
    _ = t.vectorize(1);
}

test "split vectorize" {
    const Indices = enum { i, j };
    const split_vectorized_iter_space = comptime IterSpace.init(.{ 16, 8 }, Indices)
        .split(0, 4)
        .vectorize(2);
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
