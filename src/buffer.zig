const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");

pub fn AllocatedBuffer(comptime Array: type) type {
    return AllocatedVecBuffer(Array, null);
}

pub fn AllocatedVecBuffer(comptime Array: type, comptime vec_len: ?usize) type {
    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;
        const SIMD_ALIGN = @alignOf(Vec);
        pub const alignment = @max(SIMD_ALIGN, CACHE_LINE);

        layout: utils.Layout,
        data: [*]dtype align(alignment),

        pub const dtype = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape = utils.extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Vec: type = if (vec_len) |len| @Vector(len, dtype) else void;
        pub const Arr: type = Array;

        const Unit = if (Vec != void) Vec else dtype;

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

        pub fn init(slice: []align(alignment) dtype) AllocatedBuffer(Arr) {
            return AllocatedBuffer(Arr){
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

        pub fn alloc(allocator: std.mem.Allocator) !AllocatedBuffer(Arr) {
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

        pub inline fn load(b: Self, idx: [ndims]usize) Unit {
            if (Unit == Vec) {
                const len = @typeInfo(Vec).Vector.len;
                const val: *const Unit align(alignment) = @alignCast(@ptrCast(b.data[b.unravel(idx)..][0..len]));
                return val.*;
            } else {
                return b.data[b.unravel(idx)];
            }
        }

        pub inline fn store(b: Self, val: Unit, idx: [ndims]usize) void {
            if (Unit == Vec) {
                const len = @typeInfo(Vec).Vector.len;
                const dst: *[len]dtype align(alignment) = @alignCast(@ptrCast(b.data[b.unravel(idx)..][0..len]));
                dst.* = val;
            } else {
                b.data[b.unravel(idx)] = val;
            }
        }
    };
}

test "init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    var d = try AllocatedBuffer([16][8]f32).alloc(arena.allocator());
    defer arena.deinit();

    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[0][0]) == @intFromPtr(&d.data[0]));
    try std.testing.expect(@intFromPtr(&@as(*const [16][8]f32, @ptrCast(d.data))[1][7]) == @intFromPtr(&d.data[15]));
}
