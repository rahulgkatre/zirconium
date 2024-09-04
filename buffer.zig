const std = @import("std");
const loop = @import("loop.zig");
const utils = @import("utils.zig");

pub fn AllocatedBuffer(comptime Array: type) type {
    return struct {
        const Self = @This();
        const CACHE_LINE = std.atomic.cache_line;
        const SIMD_ALIGN = @alignOf(Vec);
        const alignment = @max(SIMD_ALIGN, CACHE_LINE);

        layout: utils.Layout,
        multi: *Array align(alignment),
        raw: [*]dtype align(alignment),

        const dtype = utils.Datatype(Array);
        const ndims = utils.extractNdims(Array);
        const shape = utils.extractShape(Array);
        /// Vectorized contiguous dim of the buffer.
        pub const Vec: type = @Vector(shape[ndims - 1], dtype);
        pub const Arr: type = Array;

        const ThisUnit = utils.Unit(Array);
        const ThisVectorized = utils.Vectorized(Array);

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

        pub fn alloc(allocator: std.mem.Allocator) !AllocatedBuffer(Arr) {
            const slice: []align(alignment) dtype = try allocator.alignedAlloc(dtype, alignment, numel.?);
            if (dtype == bool) {
                @memset(slice, false);
            } else {
                @memset(slice, 0);
            }

            return AllocatedBuffer(Arr){
                .layout = .{
                    .ndims = ndims,
                    .shape = &shape,
                    .strides = &contiguous_strides,
                    // .skew = &(.{0} ** ndims),
                    // .unrolled = &(.{false} ** ndims),
                    // .parallelized = &(.{false} ** ndims),
                },
                .multi = @alignCast(@ptrCast(slice)),
                .raw = @alignCast(@ptrCast(slice)),
            };
        }

        pub fn constant(_: Self, val: dtype) ThisUnit {
            if (ThisUnit == dtype) {
                return val;
            } else {
                return @splat(val);
            }
        }

        pub inline fn unravel(b: Self, idx: [ndims]usize) usize {
            var i: usize = 0;
            inline for (0..ndims) |d| {
                comptime if (ThisUnit != dtype and d == ndims - 1) break;
                i += idx[d] * b.layout.strides[d];
            }
            return i;
        }

        pub inline fn load(b: Self, idx: [ndims]usize) ThisUnit {
            if (comptime ThisUnit != dtype) {
                const val: *const ThisUnit align(alignment) = @alignCast(@ptrCast(b.raw[b.unravel(idx)..][0..shape[ndims - 1]]));
                return @as(ThisUnit, val.*);
            } else {
                return b.raw[b.unravel(idx)];
            }
        }

        pub inline fn store(b: Self, val: ThisUnit, idx: [ndims]usize) void {
            if (comptime ThisUnit != dtype) {
                const dst: [*]ThisUnit align(alignment) = @alignCast(@ptrCast(b.raw[b.unravel(idx)..][0..shape[ndims - 1]]));
                // TODO: Need to verify what asm this generates, needs to be vmovaps
                @memcpy(@as([*]dtype, @ptrCast(dst)), &@as([shape[ndims - 1]]dtype, val));
            } else {
                b.raw[b.unravel(idx)] = val;
            }
        }
    };
}
