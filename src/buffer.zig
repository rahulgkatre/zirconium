const std = @import("std");
const loop = @import("loop.zig");
const iterspace = @import("iterspace.zig");
const utils = @import("utils.zig");

pub fn Buffer(comptime Array: type) type {
    return IterSpaceBuffer(Array, null);
}

pub fn IterSpaceBuffer(comptime Array: type, comptime raw_iter_space: anytype) type {
    var default_idx_names: [utils.extractNdims(Array)][:0]const u8 = undefined;
    for (0..default_idx_names.len) |dim| {
        default_idx_names[dim] = std.fmt.comptimePrint("dim{d}", .{dim});
    }
    const IterSpace = if (@TypeOf(raw_iter_space) == @TypeOf(null)) iterspace.IterationSpace(Array) else iterspace.IterationSpace(@TypeOf(raw_iter_space).Arr);
    const iter_space: IterSpace = if (@TypeOf(raw_iter_space) == @TypeOf(null)) iterspace.IterationSpace(Array).init(default_idx_names) else @as(*const IterSpace, @ptrCast(&raw_iter_space)).*;

    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;

        const Tile = @Vector(iter_space.iter_shape[iter_space.iter_ndims - 1], dtype);

        const TILE_ALIGN = @alignOf(Tile);

        pub const dtype = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape = utils.extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Arr: type = Array;

        pub const Unit = if (iter_space.idx_info[iter_space.iter_ndims - 1].vector) Tile else dtype;

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
            return Buffer(Arr){
                .layout = .{
                    .ndims = ndims,
                    .shape = &shape,
                    .strides = &contiguous_strides,
                    // .skew = &(.{0} ** ndims),
                    // .unrolled = &(.{false} ** ndims),
                    // .parallelized = &(.{false} ** ndims),
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

        pub fn constant(_: Self, val: dtype) Unit {
            if (Unit == dtype) {
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

        /// TODO: Using indices + iter_space, determine the type of Unit.
        /// e.g. if accessing a vectorized index, Unit should be a vector
        pub inline fn load(
            b: Self,
            comptime indices: [ndims]iter_space.Indices(),
            idx: [iter_space.idx_ndims]usize,
        ) Unit {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(Unit)) {
                .Vector => |info| {
                    const len = info.len;
                    const val: *const Unit = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
                    return val.*;
                },
                .Array => |_| {
                    @compileError("Multidimensional tiles (e.g. for WMMA) not yet supported");
                },
                else => return b.data[b.unravel(selected)],
            }
        }

        pub inline fn store(
            b: Self,
            comptime indices: [ndims]iter_space.Indices(),
            val: Unit,
            idx: [iter_space.idx_ndims]usize,
        ) void {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(Unit)) {
                .Vector => |info| {
                    const len = info.len;
                    const dst: *[len]dtype = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
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
    var d = try Buffer([16][8]f32).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[0][0]) == @intFromPtr(&d.data[0]));
    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[1][7]) == @intFromPtr(&d.data[15]));
}
