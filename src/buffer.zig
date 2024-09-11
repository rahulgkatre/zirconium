const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");

pub fn Buffer(comptime Array: type, comptime Indices: type) type {
    return TiledBuffer(Array, Indices, void);
}

pub fn TiledBuffer(comptime Array: type, comptime Indices: type, comptime Tile: type) type {
    switch (@typeInfo(Tile)) {
        .Vector, .Array, .Void => {},
        else => @compileError("Unsupported tile type: " ++ @typeName(Tile)),
    }
    const nindices: u8 = switch (@typeInfo(Indices)) {
        .Enum => |info| @intCast(info.fields.len),
        else => @compileError("Indices type must be enum, received " ++ @typeName(Indices)),
    };

    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;
        const TILE_ALIGN = @alignOf(Tile);
        pub const alignment = @max(TILE_ALIGN, CACHE_LINE);

        layout: utils.Layout,
        data: [*]dtype align(alignment),

        pub const dtype = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape = utils.extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Arr: type = Array;

        pub const Unit = if (Tile != void) Tile else dtype;

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

        pub fn init(slice: []align(alignment) dtype) Buffer(Arr, Indices) {
            return Buffer(Arr, Indices){
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

        pub fn alloc(allocator: std.mem.Allocator) !Buffer(Arr, Indices) {
            const slice: []align(alignment) dtype = try allocator.alignedAlloc(dtype, alignment, numel.?);
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

        pub inline fn unravel(b: Self, idx: [ndims]usize) usize {
            var i: usize = 0;
            inline for (0..ndims) |d| {
                i += idx[d] * b.layout.strides[d];
            }
            return i;
        }

        pub inline fn load(
            b: Self,
            comptime indices: [ndims]Indices,
            idx: [nindices]usize,
        ) Unit {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(Tile)) {
                .Vector => |info| {
                    const len = info.len;
                    const val: *const Unit = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
                    return val.*;
                },
                .Array => |_| {
                    @compileError("Multidimensional tiles (e.g. for WMMA) not yet supported");
                },
                .Void => return b.data[b.unravel(selected)],
                else => unreachable,
            }
        }

        pub inline fn store(
            b: Self,
            comptime indices: [ndims]Indices,
            val: Unit,
            idx: [nindices]usize,
        ) void {
            var selected: [ndims]usize = undefined;
            inline for (indices, 0..) |ind, d| {
                selected[d] = idx[@intFromEnum(ind)];
            }
            switch (@typeInfo(Tile)) {
                .Vector => |info| {
                    const len = info.len;
                    const dst: *[len]dtype = @alignCast(@ptrCast(b.data[b.unravel(selected)..][0..len]));
                    dst.* = val;
                },
                .Array => |_| {
                    @compileError("Multidimensional tiles (e.g. for WMMA) not yet supported");
                },
                .Void => b.data[b.unravel(selected)] = val,
                else => unreachable,
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var d = try Buffer([16][8]f32, enum {}).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[0][0]) == @intFromPtr(&d.data[0]));
    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[1][7]) == @intFromPtr(&d.data[15]));
}
