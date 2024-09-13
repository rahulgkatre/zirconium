const std = @import("std");
const loop = @import("loop.zig");
const iterspace = @import("iterspace.zig");
const utils = @import("utils.zig");

pub fn Buffer(comptime Array: type, comptime Indices: type) type {
    const iter_space = iterspace.init(utils.extractShape(Array), Indices);
    return IterSpaceBuffer(Array, iter_space);
}

pub fn IterSpaceBuffer(comptime Array: type, comptime iter_space: iterspace) type {
    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;

        pub const dtype = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape = utils.extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Arr: type = Array;

        layout: utils.Layout,
        data: [*]dtype align(CACHE_LINE),

        const numel: ?comptime_int = blk: {
            var n = 1;
            for (shape) |len| {
                n *= len;
            }
            break :blk n;
        };

        const contiguous_strides: [ndims]usize = blk: {
            if (ndims == 0) {
                break :blk .{};
            }
            var offset: usize = 1;
            var strides: [ndims]usize = undefined;
            for (0..ndims - 1) |d| {
                const stride = shape[ndims - d - 1] * offset;
                strides[ndims - d - 2] = stride;
                offset = stride;
            }
            strides[ndims - 1] = 1;
            for (0..ndims) |d| {
                if (shape[d] == 0 or shape[d] == 1) {
                    strides[d] = 0;
                }
            }
            break :blk strides;
        };

        pub fn init(slice: []align(CACHE_LINE) dtype) IterSpaceBuffer(Arr, iter_space) {
            return IterSpaceBuffer(Arr, iter_space){
                .layout = .{
                    .ndims = ndims,
                    .shape = &shape,
                    .strides = &contiguous_strides,
                },
                .data = @alignCast(@ptrCast(slice)),
            };
        }

        /// Allocate the array for this buffer.
        pub fn alloc(allocator: std.mem.Allocator) !IterSpaceBuffer(Arr, iter_space) {
            const slice: []align(CACHE_LINE) dtype = try allocator.alignedAlloc(dtype, CACHE_LINE, numel.?);
            if (dtype == bool) {
                @memset(slice, false);
            } else {
                @memset(slice, 0);
            }
            return init(slice);
        }

        pub fn constant(_: Self, comptime indices: [ndims]iter_space.Indices, val: dtype) BlockData(indices) {
            if (BlockData(indices) == dtype) {
                return val;
            } else {
                return @splat(val);
            }
        }

        /// Use the strides to convert an n-dimensional index to a flat index
        pub inline fn unravel(b: Self, idx: [ndims]usize) usize {
            var i: usize = 0;
            inline for (0..ndims) |d| {
                i += idx[d] * b.layout.strides[d];
            }
            return i;
        }

        fn BlockData(comptime indices: [ndims]iter_space.Indices) type {
            for (indices, 0..) |ind, d| {
                const dim = @intFromEnum(ind);
                for (iter_space.loop_info) |loop_info| {
                    if (loop_info.idx_dim == dim and loop_info.vector and d == ndims - 1) {
                        return @Vector(loop_info.block_size, dtype);
                    }
                }
            }
            return dtype;
        }

        /// TODO: Using indices + iter_space, determine the type of Unit.
        /// e.g. if accessing a vectorized index, Unit should be a vector
        pub inline fn load(
            b: Self,
            /// Specify which indices of the iteration space will be used to index into the array
            comptime indices: [ndims]iter_space.Indices,
            idx: [iter_space.idx_ndims]usize,
        ) BlockData(indices) {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(BlockData(indices))) {
                .Vector => |info| {
                    const len = info.len;
                    const ptr: *const BlockData(indices) align(@alignOf(BlockData(indices))) = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
                    return ptr.*;
                },
                .Array => |_| {
                    @compileError("Multidimensional tiles (e.g. for WMMA) not yet supported");
                },
                // TODO: splat if one dimension is a vector but the contiguous dimension is not
                else => return b.data[b.unravel(selected)],
            }
        }

        pub inline fn store(
            b: Self,
            comptime indices: [ndims]iter_space.Indices,
            // idea; make val anytype, store reduction dimension info in idx type along with reduction op
            // if val is a vector, reduce it with the op, otherwise proceed with storing the val directly
            // if both are vectors, no reduction is needed
            val: BlockData(indices),
            idx: [iter_space.idx_ndims]usize,
        ) void {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(BlockData(indices))) {
                .Vector => |info| {
                    const len = info.len;
                    const dst: *[len]dtype align(@alignOf([len]dtype)) = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
                    dst.* = val;
                },
                .Array => |_| {
                    @compileError("Multidimensional tiles (e.g. for WMMA) not yet supported");
                },
                else => b.data[b.unravel(selected)] = val,
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var d = try Buffer([16][8]f32, enum { i, j }).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[0][0]) == @intFromPtr(&d.data[0]));
    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[1][7]) == @intFromPtr(&d.data[15]));
}
